import tensorflow as tf

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper

try:
    from sionna.phy.utils import ebnodb2no
except Exception:
    from sionna.utils import ebnodb2no


class AlamoutiSystem:
    """
    2x2 Alamouti/STBC system:
    - 2 Tx antennas
    - 2 Rx antennas
    - 64-QAM
    - LDPC 5G
    - Rayleigh fading + AWGN

    This is NOT spatial multiplexing.
    Alamouti uses STBC combiner, not LMMSEEqualizer.
    """

    def __init__(self, code_rate=0.5, decoder_iter=25):
        self.num_tx_ant = 2
        self.num_rx_ant = 2
        self.bits_per_symbol = 6  # 64-QAM

        # Match old OFDM grid size approximately: 14 x 64 = 896 QAM symbols
        self.num_data_symbols = 14 * 64
        if self.num_data_symbols % 2 != 0:
            self.num_data_symbols -= 1

        self.num_pairs = self.num_data_symbols // 2

        # LDPC codeword length
        self.n = self.num_data_symbols * self.bits_per_symbol

        # information length
        self.k = int(self.n * code_rate)

        self.code_rate = self.k / self.n

        self.encoder = LDPC5GEncoder(self.k, self.n)

        self.decoder = LDPC5GDecoder(
            self.encoder,
            num_iter=decoder_iter,
            hard_out=True
        )

        self.mapper = Mapper("qam", self.bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.bits_per_symbol)

        # Total transmit power normalization.
        # At each Alamouti time slot, 2 antennas transmit.
        # Scale each antenna by 1/sqrt(2) so total average Tx power stays fixed.
        self.alpha = tf.constant(1.0 / (2.0 ** 0.5), dtype=tf.float32)

    def compute_noise_variance(self, ebno_db, coderate):
        """
        Eb/N0 -> noise variance.

        Assumptions:
        - Sionna QAM constellation is normalized.
        - STBC rate = 1 for Alamouti.
        - Total transmit power is normalized by alpha = 1/sqrt(2).
        """

        ebno_db = tf.cast(ebno_db, tf.float32)

        no = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.bits_per_symbol,
            coderate=coderate
        )

        return tf.cast(no, tf.float32)

    def hard_bits_from_llr(self, llr):
        """
        Sionna convention used here:
        LLR > 0 -> bit 1
        LLR <= 0 -> bit 0
        """

        return tf.cast(tf.math.greater(llr, 0.0), tf.float32)

    def count_bit_errors(self, bits_tx, bits_rx):
        err = tf.reduce_sum(
            tf.cast(tf.not_equal(bits_tx, bits_rx), tf.float32)
        )
        total = tf.cast(tf.size(bits_tx), tf.float32)
        return err, total

    def count_symbol_errors_from_bits(self, bits_tx, bits_rx):
        """
        SER by grouping every 6 bits into one 64-QAM symbol.
        A symbol is wrong if at least one of its 6 bits is wrong.
        """

        bits_tx_sym = tf.reshape(bits_tx, [-1, self.bits_per_symbol])
        bits_rx_sym = tf.reshape(bits_rx, [-1, self.bits_per_symbol])

        sym_err_bool = tf.reduce_any(
            tf.not_equal(bits_tx_sym, bits_rx_sym),
            axis=1
        )

        err = tf.reduce_sum(tf.cast(sym_err_bool, tf.float32))
        total = tf.cast(tf.shape(bits_tx_sym)[0], tf.float32)

        return err, total

    def alamouti_channel(self, x_symbols, no):
        """
        Alamouti 2Tx-2Rx channel.

        Input:
            x_symbols: [batch, 1, 1, num_data_symbols]

        Returns:
            x_hat_comb:
                Alamouti/STBC combined symbols.

            no_eff_comb:
                Effective noise variance after Alamouti combining.

            x_hat_no_comb:
                Weak baseline without STBC combining.

            no_eff_no_comb:
                Noise variance for weak baseline.
        """

        x = tf.reshape(x_symbols, [-1, self.num_data_symbols])
        batch_size = tf.shape(x)[0]

        x_pair = tf.reshape(x, [batch_size, self.num_pairs, 2])
        s1 = x_pair[:, :, 0]
        s2 = x_pair[:, :, 1]

        # Rayleigh channel:
        # h: [batch, pair, rx_ant, tx_ant]
        h_real = tf.random.normal(
            [batch_size, self.num_pairs, self.num_rx_ant, self.num_tx_ant],
            dtype=tf.float32
        )
        h_imag = tf.random.normal(
            [batch_size, self.num_pairs, self.num_rx_ant, self.num_tx_ant],
            dtype=tf.float32
        )

        h = tf.complex(h_real, h_imag) / tf.cast(
            tf.sqrt(tf.constant(2.0, dtype=tf.float32)),
            tf.complex64
        )

        h1 = h[:, :, :, 0]  # Tx1 -> Rx antennas
        h2 = h[:, :, :, 1]  # Tx2 -> Rx antennas

        alpha_c = tf.cast(self.alpha, tf.complex64)

        # Apply total-power normalization
        a1 = alpha_c * h1
        a2 = alpha_c * h2

        s1_e = tf.expand_dims(s1, axis=-1)
        s2_e = tf.expand_dims(s2, axis=-1)

        # Alamouti transmit matrix:
        # time 1: Tx1 = s1,        Tx2 = s2
        # time 2: Tx1 = -conj(s2), Tx2 = conj(s1)
        y1_clean = a1 * s1_e + a2 * s2_e
        y2_clean = -a1 * tf.math.conj(s2_e) + a2 * tf.math.conj(s1_e)

        # Complex AWGN with variance no
        noise_std = tf.sqrt(no / 2.0)
        noise_std_c = tf.cast(noise_std, tf.complex64)

        n1 = noise_std_c * tf.complex(
            tf.random.normal(tf.shape(y1_clean), dtype=tf.float32),
            tf.random.normal(tf.shape(y1_clean), dtype=tf.float32)
        )

        n2 = noise_std_c * tf.complex(
            tf.random.normal(tf.shape(y2_clean), dtype=tf.float32),
            tf.random.normal(tf.shape(y2_clean), dtype=tf.float32)
        )

        y1 = y1_clean + n1
        y2 = y2_clean + n2

        # ======================================================
        # Correct Alamouti/STBC combiner
        # ======================================================
        s1_comb = tf.reduce_sum(
            tf.math.conj(a1) * y1 + a2 * tf.math.conj(y2),
            axis=2
        )

        s2_comb = tf.reduce_sum(
            tf.math.conj(a2) * y1 - a1 * tf.math.conj(y2),
            axis=2
        )

        denom = tf.reduce_sum(
            tf.abs(a1) ** 2 + tf.abs(a2) ** 2,
            axis=2
        )

        denom = tf.maximum(
            denom,
            tf.constant(1e-12, dtype=tf.float32)
        )

        s1_hat = s1_comb / tf.cast(denom, tf.complex64)
        s2_hat = s2_comb / tf.cast(denom, tf.complex64)

        # Effective noise after combining
        no_eff_pair = no / denom

        x_hat_pair = tf.stack([s1_hat, s2_hat], axis=-1)
        x_hat_comb = tf.reshape(
            x_hat_pair,
            [batch_size, self.num_data_symbols]
        )

        no_eff_pair2 = tf.stack([no_eff_pair, no_eff_pair], axis=-1)
        no_eff_comb = tf.reshape(
            no_eff_pair2,
            [batch_size, self.num_data_symbols]
        )

        x_hat_comb = tf.reshape(
            x_hat_comb,
            [batch_size, 1, 1, self.num_data_symbols]
        )

        no_eff_comb = tf.reshape(
            no_eff_comb,
            [batch_size, 1, 1, self.num_data_symbols]
        )

        # ======================================================
        # Weak baseline: without STBC combining
        # ======================================================
        # This is intentionally bad.
        # It directly treats received antenna-0 signal as if it were the transmitted QAM symbol.
        y1_rx0 = y1[:, :, 0]
        y2_rx0 = y2[:, :, 0]

        x_no_pair = tf.stack(
            [y1_rx0, tf.math.conj(y2_rx0)],
            axis=-1
        )

        x_hat_no_comb = tf.reshape(
            x_no_pair,
            [batch_size, self.num_data_symbols]
        )

        x_hat_no_comb = tf.reshape(
            x_hat_no_comb,
            [batch_size, 1, 1, self.num_data_symbols]
        )

        no_eff_no_comb = tf.ones_like(
            tf.math.real(x_hat_no_comb),
            dtype=tf.float32
        ) * no

        return x_hat_comb, no_eff_comb, x_hat_no_comb, no_eff_no_comb

    @tf.function
    def process_batch(self, batch_size, ebno_db):
        no_coded = self.compute_noise_variance(
            ebno_db,
            coderate=self.code_rate
        )

        no_uncoded = self.compute_noise_variance(
            ebno_db,
            coderate=1.0
        )

        # ======================================================
        # Branch 1: LDPC coded + Alamouti
        # ======================================================
        bits_info = tf.random.uniform(
            [batch_size, 1, 1, self.k],
            minval=0,
            maxval=2,
            dtype=tf.int32
        )

        bits_info = tf.cast(bits_info, tf.float32)

        coded_bits = self.encoder(bits_info)
        x_coded = self.mapper(coded_bits)

        (
            x_c_comb,
            no_c_comb,
            x_c_no,
            no_c_no
        ) = self.alamouti_channel(x_coded, no_coded)

        # With STBC combiner
        llr_c_comb = self.demapper(x_c_comb, no_c_comb)
        coded_bits_hat_comb = self.hard_bits_from_llr(llr_c_comb)

        pre_bit_err_c_comb, pre_bit_total_c_comb = self.count_bit_errors(
            coded_bits,
            coded_bits_hat_comb
        )

        ser_err_c_comb, ser_total_c_comb = self.count_symbol_errors_from_bits(
            coded_bits,
            coded_bits_hat_comb
        )

        bits_info_hat_comb = self.decoder(llr_c_comb)

        post_bit_err_c_comb, post_bit_total_c_comb = self.count_bit_errors(
            bits_info,
            bits_info_hat_comb
        )

        # Without STBC combiner
        llr_c_no = self.demapper(x_c_no, no_c_no)
        coded_bits_hat_no = self.hard_bits_from_llr(llr_c_no)

        pre_bit_err_c_no, pre_bit_total_c_no = self.count_bit_errors(
            coded_bits,
            coded_bits_hat_no
        )

        ser_err_c_no, ser_total_c_no = self.count_symbol_errors_from_bits(
            coded_bits,
            coded_bits_hat_no
        )

        bits_info_hat_no = self.decoder(llr_c_no)

        post_bit_err_c_no, post_bit_total_c_no = self.count_bit_errors(
            bits_info,
            bits_info_hat_no
        )

        # ======================================================
        # Branch 2: Uncoded + Alamouti
        # ======================================================
        bits_u = tf.random.uniform(
            [batch_size, 1, 1, self.n],
            minval=0,
            maxval=2,
            dtype=tf.int32
        )

        bits_u = tf.cast(bits_u, tf.float32)

        x_u = self.mapper(bits_u)

        (
            x_u_comb,
            no_u_comb,
            x_u_no,
            no_u_no
        ) = self.alamouti_channel(x_u, no_uncoded)

        # Uncoded with STBC combiner
        llr_u_comb = self.demapper(x_u_comb, no_u_comb)
        bits_u_hat_comb = self.hard_bits_from_llr(llr_u_comb)

        bit_err_u_comb, bit_total_u_comb = self.count_bit_errors(
            bits_u,
            bits_u_hat_comb
        )

        ser_err_u_comb, ser_total_u_comb = self.count_symbol_errors_from_bits(
            bits_u,
            bits_u_hat_comb
        )

        # Uncoded without STBC combiner
        llr_u_no = self.demapper(x_u_no, no_u_no)
        bits_u_hat_no = self.hard_bits_from_llr(llr_u_no)

        bit_err_u_no, bit_total_u_no = self.count_bit_errors(
            bits_u,
            bits_u_hat_no
        )

        ser_err_u_no, ser_total_u_no = self.count_symbol_errors_from_bits(
            bits_u,
            bits_u_hat_no
        )

        return (
            # LDPC + combiner
            post_bit_err_c_comb,
            post_bit_total_c_comb,
            pre_bit_err_c_comb,
            pre_bit_total_c_comb,
            ser_err_c_comb,
            ser_total_c_comb,

            # LDPC no combiner
            post_bit_err_c_no,
            post_bit_total_c_no,
            pre_bit_err_c_no,
            pre_bit_total_c_no,
            ser_err_c_no,
            ser_total_c_no,

            # Uncoded + combiner
            bit_err_u_comb,
            bit_total_u_comb,
            ser_err_u_comb,
            ser_total_u_comb,

            # Uncoded no combiner
            bit_err_u_no,
            bit_total_u_no,
            ser_err_u_no,
            ser_total_u_no
        )

    def _metric(self, errors, total):
        errors = float(errors)
        total = float(total)

        if total <= 0:
            return {
                "value": float("nan"),
                "errors": errors,
                "total": total,
                "upper_bound": True
            }

        if errors <= 0:
            return {
                "value": 1.0 / total,
                "errors": errors,
                "total": total,
                "upper_bound": True
            }

        return {
            "value": errors / total,
            "errors": errors,
            "total": total,
            "upper_bound": False
        }

    def run_monte_carlo(
        self,
        ebno_db,
        min_errors=500,
        max_bits=5e6,
        batch_size=64
    ):
        acc = {
            "post_err_c_comb": 0.0,
            "post_total_c_comb": 0.0,
            "pre_err_c_comb": 0.0,
            "pre_total_c_comb": 0.0,
            "ser_err_c_comb": 0.0,
            "ser_total_c_comb": 0.0,

            "post_err_c_no": 0.0,
            "post_total_c_no": 0.0,
            "pre_err_c_no": 0.0,
            "pre_total_c_no": 0.0,
            "ser_err_c_no": 0.0,
            "ser_total_c_no": 0.0,

            "bit_err_u_comb": 0.0,
            "bit_total_u_comb": 0.0,
            "ser_err_u_comb": 0.0,
            "ser_total_u_comb": 0.0,

            "bit_err_u_no": 0.0,
            "bit_total_u_no": 0.0,
            "ser_err_u_no": 0.0,
            "ser_total_u_no": 0.0,
        }

        while (
            (
                acc["post_err_c_comb"] < min_errors
                or acc["bit_err_u_comb"] < min_errors
            )
            and acc["post_total_c_comb"] < max_bits
        ):
            result = self.process_batch(
                int(batch_size),
                tf.constant(float(ebno_db), dtype=tf.float32)
            )

            keys = list(acc.keys())

            for key, value in zip(keys, result):
                acc[key] += float(value.numpy())

        return {
            # Main BER
            "ber_ldpc_comb": self._metric(
                acc["post_err_c_comb"],
                acc["post_total_c_comb"]
            ),
            "ber_ldpc_no_comb": self._metric(
                acc["post_err_c_no"],
                acc["post_total_c_no"]
            ),
            "ber_uncoded_comb": self._metric(
                acc["bit_err_u_comb"],
                acc["bit_total_u_comb"]
            ),
            "ber_uncoded_no_comb": self._metric(
                acc["bit_err_u_no"],
                acc["bit_total_u_no"]
            ),

            # Debug pre-decoder coded BER
            "pre_ber_ldpc_comb": self._metric(
                acc["pre_err_c_comb"],
                acc["pre_total_c_comb"]
            ),
            "pre_ber_ldpc_no_comb": self._metric(
                acc["pre_err_c_no"],
                acc["pre_total_c_no"]
            ),

            # SER
            "ser_ldpc_comb": self._metric(
                acc["ser_err_c_comb"],
                acc["ser_total_c_comb"]
            ),
            "ser_ldpc_no_comb": self._metric(
                acc["ser_err_c_no"],
                acc["ser_total_c_no"]
            ),
            "ser_uncoded_comb": self._metric(
                acc["ser_err_u_comb"],
                acc["ser_total_u_comb"]
            ),
            "ser_uncoded_no_comb": self._metric(
                acc["ser_err_u_no"],
                acc["ser_total_u_no"]
            ),
        }