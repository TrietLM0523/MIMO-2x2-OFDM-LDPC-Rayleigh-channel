import numpy as np
import tensorflow as tf

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.channel import RayleighBlockFading, OFDMChannel, AWGN
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LMMSEEqualizer

# Nếu dòng này lỗi do khác version Sionna, đổi thành:
# from sionna.utils import ebnodb2no
from sionna.phy.utils import ebnodb2no


class MIMOSystem:
    def __init__(self, code_rate=0.5):
        # ======================================================
        # 1. System parameters
        # ======================================================
        self.num_tx_ant = 2
        self.num_rx_ant = 2

        # 64-QAM = 6 bits per modulation symbol
        self.bits_per_symbol = 6

        # ======================================================
        # 2. OFDM resource grid
        # ======================================================
        self.rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=15e3,
            num_tx=1,
            num_streams_per_tx=self.num_tx_ant,
            cyclic_prefix_length=16
        )

        self.num_data_symbols = int(self.rg.num_data_symbols)

        # n: number of coded bits per stream
        self.n = self.num_data_symbols * self.bits_per_symbol

        # k: number of information bits per stream
        self.k = int(self.n * code_rate)

        # Actual code rate after integer rounding
        self.code_rate = self.k / self.n

        # ======================================================
        # 3. LDPC encoder and decoder
        # ======================================================
        self.encoder = LDPC5GEncoder(self.k, self.n)

        self.decoder = LDPC5GDecoder(
            self.encoder,
            num_iter=50,
            hard_out=True
        )

        # ======================================================
        # 4. 64-QAM mapper and demapper
        # ======================================================
        self.mapper = Mapper("qam", self.bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.bits_per_symbol)

        # ======================================================
        # 5. OFDM mapper
        # ======================================================
        self.rg_mapper = ResourceGridMapper(self.rg)

        # ======================================================
        # 6. 2x2 Rayleigh block fading channel + AWGN
        # ======================================================
        self.channel_model = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=self.num_rx_ant,
            num_tx=1,
            num_tx_ant=self.num_tx_ant
        )

        self.channel = OFDMChannel(
            self.channel_model,
            self.rg,
            add_awgn=False,
            normalize_channel=True,
            return_channel=True
        )

        self.awgn = AWGN()

        # ======================================================
        # 7. Stream management and LMMSE equalizer
        # ======================================================
        rx_tx_association = np.array([[1]])

        self.stream_management = StreamManagement(
            rx_tx_association,
            self.num_tx_ant
        )

        self.equalizer = LMMSEEqualizer(
            self.rg,
            self.stream_management
        )

    def compute_ser_from_bits(self, bits_tx, llr_rx):
        """
        Compute SER by grouping hard-decoded bits into 64-QAM symbols.

        For 64-QAM:
            1 symbol = 6 bits

        A symbol is considered incorrect if at least one of its 6 bits is wrong.

        Sionna APP demapper convention:
            LLR > 0  -> bit 1
            LLR <= 0 -> bit 0
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

    def no_equalizer_extract_streams(self, y_rg):
        """
        A simple 'without MIMO equalization' baseline.

        y_rg shape from OFDMChannel:
            [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]

        In proper MIMO spatial multiplexing, the receiver needs an equalizer
        to separate streams. Here, for a no-equalizer baseline, we directly
        take the received antenna signals and treat them as streams.

        This is intentionally a weak baseline to show the importance of
        MIMO equalization.
        """

        # Remove num_rx dimension because num_rx = 1
        # Shape: [batch, num_rx_ant, num_ofdm_symbols, fft_size]
        y_ant = y_rg[:, 0, :, :, :]

        # Use the first num_tx_ant receive antennas as if they were streams
        # Shape: [batch, num_tx_ant, num_ofdm_symbols, fft_size]
        y_ant = y_ant[:, :self.num_tx_ant, :, :]

        # Flatten OFDM resource grid back to stream symbols
        # Shape: [batch, 1, num_streams, num_data_symbols]
        y_no_eq = tf.reshape(
            y_ant,
            [-1, 1, self.num_tx_ant, self.num_data_symbols]
        )

        return y_no_eq

    @tf.function
    def process_batch(self, batch_size, ebno_db):
        # ======================================================
        # Convert Eb/N0 to noise variance using Sionna standard
        # ======================================================
        ebno_db = tf.cast(ebno_db, tf.float32)

        no_coded_tensor = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.bits_per_symbol,
            coderate=self.code_rate,
            resource_grid=self.rg
        )

        no_uncoded_tensor = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.bits_per_symbol,
            coderate=1.0,
            resource_grid=self.rg
        )

        # ======================================================
        # Branch 1: LDPC coded MIMO-OFDM + LMMSE
        # bits -> LDPC -> 64-QAM -> OFDM -> Rayleigh + AWGN
        # -> LMMSE -> demapper -> LDPC decoder -> BER
        # ======================================================

        bits = tf.random.uniform(
            [batch_size, 1, self.num_tx_ant, self.k],
            minval=0,
            maxval=2,
            dtype=tf.int32
        )

        bits_float = tf.cast(bits, tf.float32)

        coded_bits = self.encoder(bits_float)

        x_coded = self.mapper(coded_bits)
        x_rg_coded = self.rg_mapper(x_coded)

        y_rg_clean_coded, h_freq_coded = self.channel(x_rg_coded)
        y_rg_coded = self.awgn(y_rg_clean_coded, no_coded_tensor)

        # Perfect CSI assumption
        err_var_coded = tf.zeros_like(h_freq_coded, dtype=tf.float32)

        x_hat_c, no_eff_c = self.equalizer(
            y_rg_coded,
            h_freq_coded,
            err_var_coded,
            no_coded_tensor
        )

        llr_c = self.demapper(x_hat_c, no_eff_c)

        bits_est_c = self.decoder(llr_c)

        err_c = tf.reduce_sum(
            tf.cast(tf.not_equal(bits_float, bits_est_c), tf.float32)
        )

        num_bits_c = tf.cast(tf.size(bits_float), tf.float32)

        # This is not final decoded SER.
        # It is only pre-decoder SER at demapper/modulation layer.
        sym_err_pre_ldpc_c, num_sym_pre_ldpc_c = self.compute_ser_from_bits(
            coded_bits,
            llr_c
        )

        # ======================================================
        # Branch 2: Uncoded MIMO-OFDM + LMMSE reference
        # bits -> 64-QAM -> OFDM -> Rayleigh + AWGN
        # -> LMMSE -> demapper -> BER/SER
        # ======================================================

        bits_u = tf.random.uniform(
            [batch_size, 1, self.num_tx_ant, self.n],
            minval=0,
            maxval=2,
            dtype=tf.int32
        )

        bits_u_float = tf.cast(bits_u, tf.float32)

        x_u = self.mapper(bits_u_float)
        x_rg_u = self.rg_mapper(x_u)

        y_rg_clean_u, h_freq_u = self.channel(x_rg_u)
        y_rg_u = self.awgn(y_rg_clean_u, no_uncoded_tensor)

        # Perfect CSI assumption
        err_var_u = tf.zeros_like(h_freq_u, dtype=tf.float32)

        x_hat_u, no_eff_u = self.equalizer(
            y_rg_u,
            h_freq_u,
            err_var_u,
            no_uncoded_tensor
        )

        llr_u = self.demapper(x_hat_u, no_eff_u)

        bits_est_u = tf.cast(tf.math.greater(llr_u, 0.0), tf.float32)

        err_u = tf.reduce_sum(
            tf.cast(tf.not_equal(bits_u_float, bits_est_u), tf.float32)
        )

        num_bits_u = tf.cast(tf.size(bits_u_float), tf.float32)

        sym_err_u, num_sym_u = self.compute_ser_from_bits(
            bits_u_float,
            llr_u
        )

        # ======================================================
        # Branch 3: Uncoded MIMO-OFDM without MIMO equalization
        # This branch uses the same received signal y_rg_u as Branch 2.
        # It directly demaps received antenna signals without LMMSE.
        # ======================================================

        x_no_eq = self.no_equalizer_extract_streams(y_rg_u)

        no_eff_no_eq = tf.ones_like(
            tf.math.real(x_no_eq),
            dtype=tf.float32
        ) * no_uncoded_tensor

        llr_no_eq = self.demapper(x_no_eq, no_eff_no_eq)

        bits_est_no_eq = tf.cast(
            tf.math.greater(llr_no_eq, 0.0),
            tf.float32
        )

        err_no_eq = tf.reduce_sum(
            tf.cast(
                tf.not_equal(bits_u_float, bits_est_no_eq),
                tf.float32
            )
        )

        num_bits_no_eq = tf.cast(tf.size(bits_u_float), tf.float32)

        sym_err_no_eq, num_sym_no_eq = self.compute_ser_from_bits(
            bits_u_float,
            llr_no_eq
        )

        return (
            err_c,
            num_bits_c,
            sym_err_pre_ldpc_c,
            num_sym_pre_ldpc_c,
            err_u,
            num_bits_u,
            sym_err_u,
            num_sym_u,
            err_no_eq,
            num_bits_no_eq,
            sym_err_no_eq,
            num_sym_no_eq
        )

    def run_monte_carlo(
        self,
        ebno_db,
        min_errors=500,
        max_bits=5e6,
        batch_size=64
    ):
        total_err_c = 0.0
        total_bits_c = 0.0
        total_sym_err_pre_ldpc_c = 0.0
        total_sym_pre_ldpc_c = 0.0

        total_err_u = 0.0
        total_bits_u = 0.0
        total_sym_err_u = 0.0
        total_sym_u = 0.0

        total_err_no_eq = 0.0
        total_bits_no_eq = 0.0
        total_sym_err_no_eq = 0.0
        total_sym_no_eq = 0.0

        while (
            (
                total_err_c < min_errors
                or total_err_u < min_errors
                or total_err_no_eq < min_errors
            )
            and total_bits_c < max_bits
        ):
            (
                e_c,
                b_c,
                se_pre_c,
                ns_pre_c,
                e_u,
                b_u,
                se_u,
                ns_u,
                e_no_eq,
                b_no_eq,
                se_no_eq,
                ns_no_eq
            ) = self.process_batch(
                int(batch_size),
                tf.constant(float(ebno_db), dtype=tf.float32)
            )

            total_err_c += float(e_c.numpy())
            total_bits_c += float(b_c.numpy())
            total_sym_err_pre_ldpc_c += float(se_pre_c.numpy())
            total_sym_pre_ldpc_c += float(ns_pre_c.numpy())

            total_err_u += float(e_u.numpy())
            total_bits_u += float(b_u.numpy())
            total_sym_err_u += float(se_u.numpy())
            total_sym_u += float(ns_u.numpy())

            total_err_no_eq += float(e_no_eq.numpy())
            total_bits_no_eq += float(b_no_eq.numpy())
            total_sym_err_no_eq += float(se_no_eq.numpy())
            total_sym_no_eq += float(ns_no_eq.numpy())

        # If no error is observed, show upper bound 1/N instead of 0
        ber_c = total_err_c / total_bits_c if total_err_c > 0 else 1.0 / total_bits_c

        ser_pre_ldpc_c = (
            total_sym_err_pre_ldpc_c / total_sym_pre_ldpc_c
            if total_sym_err_pre_ldpc_c > 0
            else 1.0 / total_sym_pre_ldpc_c
        )

        ber_u = total_err_u / total_bits_u if total_err_u > 0 else 1.0 / total_bits_u
        ser_u = total_sym_err_u / total_sym_u if total_sym_err_u > 0 else 1.0 / total_sym_u

        ber_no_eq = (
            total_err_no_eq / total_bits_no_eq
            if total_err_no_eq > 0
            else 1.0 / total_bits_no_eq
        )

        ser_no_eq = (
            total_sym_err_no_eq / total_sym_no_eq
            if total_sym_err_no_eq > 0
            else 1.0 / total_sym_no_eq
        )

        return (
            ber_c,
            ser_pre_ldpc_c,
            ber_u,
            ser_u,
            ber_no_eq,
            ser_no_eq
        )