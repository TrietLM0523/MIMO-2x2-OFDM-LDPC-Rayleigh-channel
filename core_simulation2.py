import numpy as np
import tensorflow as tf

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper

try:
    from sionna.phy.utils import ebnodb2no
except Exception:
    from sionna.utils import ebnodb2no


class AlamoutiSystem:
    """
    2x2 MIMO Alamouti/STBC system with:
    - 2 transmit antennas
    - 2 receive antennas
    - 64-QAM
    - LDPC 5G
    - Rayleigh fading + AWGN
    - BER/SER evaluation

    Important:
    This is a spatial diversity model, not spatial multiplexing.
    It transmits one STBC-coded stream over two transmit antennas.
    """

    def __init__(self, code_rate=0.5):
        # ======================================================
        # 1. System parameters
        # ======================================================
        self.num_tx_ant = 2
        self.num_rx_ant = 2
        self.bits_per_symbol = 6  # 64-QAM

        # Match the old OFDM grid size approximately:
        # old rg: 14 OFDM symbols x 64 subcarriers = 896 data symbols
        # Alamouti needs an even number of symbols
        self.num_data_symbols = 14 * 64

        if self.num_data_symbols % 2 != 0:
            self.num_data_symbols -= 1

        self.num_pairs = self.num_data_symbols // 2

        # n: coded bits
        self.n = self.num_data_symbols * self.bits_per_symbol

        # k: information bits
        self.k = int(self.n * code_rate)

        # Actual code rate after rounding
        self.code_rate = self.k / self.n

        # ======================================================
        # 2. LDPC encoder / decoder
        # ======================================================
        self.encoder = LDPC5GEncoder(self.k, self.n)

        self.decoder = LDPC5GDecoder(
            self.encoder,
            num_iter=50,
            hard_out=True
        )

        # ======================================================
        # 3. 64-QAM mapper / demapper
        # ======================================================
        self.mapper = Mapper("qam", self.bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.bits_per_symbol)

    def compute_noise_variance(self, ebno_db, coderate):
        """
        Convert Eb/N0 to noise variance.

        For this Alamouti core, we do not use Sionna ResourceGrid directly,
        so this is the standard Eb/N0 -> No conversion.
        """

        ebno_db = tf.cast(ebno_db, tf.float32)

        no = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.bits_per_symbol,
            coderate=coderate
        )

        return tf.cast(no, tf.float32)

    def compute_ser_from_bits(self, bits_tx, llr_rx):
        """
        SER at demapper/modulation layer.

        64-QAM:
            1 symbol = 6 bits

        A symbol is wrong if at least one of its 6 bits is wrong.
        """

        bits_rx = tf.cast(tf.math.greater(llr_rx, 0.0), tf.float32)

        bits_tx_sym = tf.reshape(bits_tx, [-1, self.bits_per_symbol])
        bits_rx_sym = tf.reshape(bits_rx, [-1, self.bits_per_symbol])

        sym_err_bool = tf.reduce_any(
            tf.not_equal(bits_tx_sym, bits_rx_sym),
            axis=1
        )

        num_sym_err = tf.reduce_sum(tf.cast(sym_err_bool, tf.float32))
        num_symbols = tf.cast(tf.shape(bits_tx_sym)[0], tf.float32)

        return num_sym_err, num_symbols

    def compute_block6_error_after_decoder(self, bits_tx, bits_rx):
        """
        Optional metric:
        6-bit block error after LDPC decoder.

        This is NOT modulation SER.
        It is just the error rate of groups of 6 information bits
        after LDPC decoding.
        """

        usable_len = (self.k // self.bits_per_symbol) * self.bits_per_symbol

        bits_tx_cut = bits_tx[..., :usable_len]
        bits_rx_cut = bits_rx[..., :usable_len]

        bits_tx_blk = tf.reshape(bits_tx_cut, [-1, self.bits_per_symbol])
        bits_rx_blk = tf.reshape(bits_rx_cut, [-1, self.bits_per_symbol])

        blk_err_bool = tf.reduce_any(
            tf.not_equal(bits_tx_blk, bits_rx_blk),
            axis=1
        )

        num_blk_err = tf.reduce_sum(tf.cast(blk_err_bool, tf.float32))
        num_blk = tf.cast(tf.shape(bits_tx_blk)[0], tf.float32)

        return num_blk_err, num_blk

    def alamouti_channel_and_combine(self, x_symbols, no):
        """
        Apply 2Tx-2Rx Alamouti STBC over Rayleigh + AWGN.

        Input:
            x_symbols shape:
                [batch, 1, 1, num_data_symbols]

        Output:
            x_hat shape:
                [batch, 1, 1, num_data_symbols]

            no_eff shape:
                [batch, 1, 1, num_data_symbols]
        """

        # Flatten to [batch, num_data_symbols]
        x = tf.reshape(x_symbols, [-1, self.num_data_symbols])
        batch_size = tf.shape(x)[0]

        # Pair symbols: [s1, s2]
        x_pair = tf.reshape(x, [batch_size, self.num_pairs, 2])
        s1 = x_pair[:, :, 0]
        s2 = x_pair[:, :, 1]

        # Rayleigh channel for each Alamouti pair
        # h shape: [batch, num_pairs, num_rx_ant, num_tx_ant]
        h_real = tf.random.normal(
            [batch_size, self.num_pairs, self.num_rx_ant, self.num_tx_ant],
            dtype=tf.float32
        )

        h_imag = tf.random.normal(
            [batch_size, self.num_pairs, self.num_rx_ant, self.num_tx_ant],
            dtype=tf.float32
        )

        h = tf.complex(h_real, h_imag) / tf.sqrt(tf.constant(2.0, tf.complex64))

        h1 = h[:, :, :, 0]  # channel from Tx1 to each Rx
        h2 = h[:, :, :, 1]  # channel from Tx2 to each Rx

        # Expand symbols for receive antenna dimension
        s1_e = tf.expand_dims(s1, axis=-1)  # [batch, pair, 1]
        s2_e = tf.expand_dims(s2, axis=-1)

        # Alamouti transmission:
        # time slot 1: Tx1 = s1,       Tx2 = s2
        # time slot 2: Tx1 = -conj(s2), Tx2 = conj(s1)
        y_t1_clean = h1 * s1_e + h2 * s2_e
        y_t2_clean = -h1 * tf.math.conj(s2_e) + h2 * tf.math.conj(s1_e)

        # AWGN
        no_c = tf.cast(no, tf.complex64)
        noise_std = tf.sqrt(no_c / tf.constant(2.0, tf.complex64))

        n1 = noise_std * tf.complex(
            tf.random.normal(tf.shape(y_t1_clean), dtype=tf.float32),
            tf.random.normal(tf.shape(y_t1_clean), dtype=tf.float32)
        )

        n2 = noise_std * tf.complex(
            tf.random.normal(tf.shape(y_t2_clean), dtype=tf.float32),
            tf.random.normal(tf.shape(y_t2_clean), dtype=tf.float32)
        )

        y1 = y_t1_clean + n1
        y2 = y_t2_clean + n2

        # Alamouti combining over receive antennas
        # s1_hat = sum_r conj(h1)*y1 + h2*conj(y2)
        # s2_hat = sum_r conj(h2)*y1 - h1*conj(y2)
        s1_comb = tf.reduce_sum(
            tf.math.conj(h1) * y1 + h2 * tf.math.conj(y2),
            axis=2
        )

        s2_comb = tf.reduce_sum(
            tf.math.conj(h2) * y1 - h1 * tf.math.conj(y2),
            axis=2
        )

        denom = tf.reduce_sum(
            tf.abs(h1) ** 2 + tf.abs(h2) ** 2,
            axis=2
        )

        denom = tf.maximum(denom, tf.constant(1e-12, dtype=tf.float32))

        s1_hat = s1_comb / tf.cast(denom, tf.complex64)
        s2_hat = s2_comb / tf.cast(denom, tf.complex64)

        # Effective noise variance after combining
        no_eff_pair = no / denom

        # Interleave s1_hat, s2_hat back to original symbol order
        x_hat_pair = tf.stack([s1_hat, s2_hat], axis=-1)
        x_hat = tf.reshape(x_hat_pair, [batch_size, self.num_data_symbols])

        no_eff_pair2 = tf.stack([no_eff_pair, no_eff_pair], axis=-1)
        no_eff = tf.reshape(no_eff_pair2, [batch_size, self.num_data_symbols])

        x_hat = tf.reshape(x_hat, [batch_size, 1, 1, self.num_data_symbols])
        no_eff = tf.reshape(no_eff, [batch_size, 1, 1, self.num_data_symbols])

        return x_hat, no_eff

    @tf.function
    def process_batch(self, batch_size, ebno_db):
        # ======================================================
        # Noise variance
        # ======================================================
        no_coded = self.compute_noise_variance(
            ebno_db,
            coderate=self.code_rate
        )

        no_uncoded = self.compute_noise_variance(
            ebno_db,
            coderate=1.0
        )

        # ======================================================
        # Branch 1: LDPC coded + Alamouti/STBC
        # ======================================================
        bits = tf.random.uniform(
            [batch_size, 1, 1, self.k],
            minval=0,
            maxval=2,
            dtype=tf.int32
        )

        bits_float = tf.cast(bits, tf.float32)

        coded_bits = self.encoder(bits_float)

        x_coded = self.mapper(coded_bits)

        x_hat_coded, no_eff_coded = self.alamouti_channel_and_combine(
            x_coded,
            no_coded
        )

        llr_coded = self.demapper(x_hat_coded, no_eff_coded)

        bits_est_coded = self.decoder(llr_coded)

        err_coded = tf.reduce_sum(
            tf.cast(tf.not_equal(bits_float, bits_est_coded), tf.float32)
        )

        num_bits_coded = tf.cast(tf.size(bits_float), tf.float32)

        # SER at demapper, before LDPC decoder
        sym_err_pre_ldpc, num_sym_pre_ldpc = self.compute_ser_from_bits(
            coded_bits,
            llr_coded
        )

        # Optional: 6-bit block error after decoder
        blk6_err_after_dec, num_blk6_after_dec = self.compute_block6_error_after_decoder(
            bits_float,
            bits_est_coded
        )

        # ======================================================
        # Branch 2: Uncoded + Alamouti/STBC
        # ======================================================
        bits_u = tf.random.uniform(
            [batch_size, 1, 1, self.n],
            minval=0,
            maxval=2,
            dtype=tf.int32
        )

        bits_u_float = tf.cast(bits_u, tf.float32)

        x_uncoded = self.mapper(bits_u_float)

        x_hat_uncoded, no_eff_uncoded = self.alamouti_channel_and_combine(
            x_uncoded,
            no_uncoded
        )

        llr_uncoded = self.demapper(x_hat_uncoded, no_eff_uncoded)

        bits_est_uncoded = tf.cast(
            tf.math.greater(llr_uncoded, 0.0),
            tf.float32
        )

        err_uncoded = tf.reduce_sum(
            tf.cast(tf.not_equal(bits_u_float, bits_est_uncoded), tf.float32)
        )

        num_bits_uncoded = tf.cast(tf.size(bits_u_float), tf.float32)

        sym_err_uncoded, num_sym_uncoded = self.compute_ser_from_bits(
            bits_u_float,
            llr_uncoded
        )

        return (
            err_coded,
            num_bits_coded,
            sym_err_pre_ldpc,
            num_sym_pre_ldpc,
            blk6_err_after_dec,
            num_blk6_after_dec,
            err_uncoded,
            num_bits_uncoded,
            sym_err_uncoded,
            num_sym_uncoded
        )

    def run_monte_carlo(
        self,
        ebno_db,
        min_errors=500,
        max_bits=5e6,
        batch_size=64
    ):
        total_err_coded = 0.0
        total_bits_coded = 0.0

        total_sym_err_pre_ldpc = 0.0
        total_sym_pre_ldpc = 0.0

        total_blk6_err_after_dec = 0.0
        total_blk6_after_dec = 0.0

        total_err_uncoded = 0.0
        total_bits_uncoded = 0.0

        total_sym_err_uncoded = 0.0
        total_sym_uncoded = 0.0

        while (
            (total_err_coded < min_errors or total_err_uncoded < min_errors)
            and total_bits_coded < max_bits
        ):
            (
                e_c,
                b_c,
                se_pre,
                ns_pre,
                be6,
                nb6,
                e_u,
                b_u,
                se_u,
                ns_u
            ) = self.process_batch(
                int(batch_size),
                tf.constant(float(ebno_db), dtype=tf.float32)
            )

            total_err_coded += float(e_c.numpy())
            total_bits_coded += float(b_c.numpy())

            total_sym_err_pre_ldpc += float(se_pre.numpy())
            total_sym_pre_ldpc += float(ns_pre.numpy())

            total_blk6_err_after_dec += float(be6.numpy())
            total_blk6_after_dec += float(nb6.numpy())

            total_err_uncoded += float(e_u.numpy())
            total_bits_uncoded += float(b_u.numpy())

            total_sym_err_uncoded += float(se_u.numpy())
            total_sym_uncoded += float(ns_u.numpy())

        ber_coded = (
            total_err_coded / total_bits_coded
            if total_err_coded > 0
            else 1.0 / total_bits_coded
        )

        ser_pre_ldpc = (
            total_sym_err_pre_ldpc / total_sym_pre_ldpc
            if total_sym_err_pre_ldpc > 0
            else 1.0 / total_sym_pre_ldpc
        )

        block6_after_decoder = (
            total_blk6_err_after_dec / total_blk6_after_dec
            if total_blk6_err_after_dec > 0
            else 1.0 / total_blk6_after_dec
        )

        ber_uncoded = (
            total_err_uncoded / total_bits_uncoded
            if total_err_uncoded > 0
            else 1.0 / total_bits_uncoded
        )

        ser_uncoded = (
            total_sym_err_uncoded / total_sym_uncoded
            if total_sym_err_uncoded > 0
            else 1.0 / total_sym_uncoded
        )

        return (
            ber_coded,
            ser_pre_ldpc,
            block6_after_decoder,
            ber_uncoded,
            ser_uncoded
        )