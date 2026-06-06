import tensorflow as tf

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper

try:
    from sionna.phy.utils import ebnodb2no
except Exception:
    from sionna.utils import ebnodb2no


class AlamoutiSystem:
    """
    2x2 Alamouti/STBC MIMO system with:
    - LDPC 5G
    - 64-QAM
    - Rayleigh fading + AWGN
    - ZF / MMSE / No Equalizer
    - BER/SER evaluation
    """

    def __init__(
        self,
        code_rate=0.5,
        decoder_iter=15,
        num_data_symbols=512,
    ):
        self.num_tx_ant = 2
        self.num_rx_ant = 2
        self.bits_per_symbol = 6

        self.num_data_symbols = int(num_data_symbols)

        if self.num_data_symbols % 2 != 0:
            self.num_data_symbols -= 1

        self.num_pairs = self.num_data_symbols // 2

        self.n = self.num_data_symbols * self.bits_per_symbol
        self.k = int(self.n * code_rate)

        self.code_rate = self.k / self.n

        self.encoder = LDPC5GEncoder(self.k, self.n)

        self.decoder = LDPC5GDecoder(
            self.encoder,
            num_iter=int(decoder_iter),
            hard_out=True
        )

        self.mapper = Mapper("qam", self.bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.bits_per_symbol)

        self.alpha = tf.constant(1.0 / (2.0 ** 0.5), dtype=tf.float32)

    def compute_noise_variance(self, ebno_db, coderate):
        ebno_db = tf.cast(ebno_db, tf.float32)

        no = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.bits_per_symbol,
            coderate=coderate
        )

        return tf.cast(no, tf.float32)

    def hard_bits_from_llr(self, llr):
        return tf.cast(tf.math.greater(llr, 0.0), tf.float32)

    def count_bit_errors(self, bits_tx, bits_rx):
        errors = tf.reduce_sum(
            tf.cast(tf.not_equal(bits_tx, bits_rx), tf.float32)
        )

        total = tf.cast(tf.size(bits_tx), tf.float32)

        return errors, total

    def count_symbol_errors_from_bits(self, bits_tx, bits_rx):
        bits_tx_sym = tf.reshape(bits_tx, [-1, self.bits_per_symbol])
        bits_rx_sym = tf.reshape(bits_rx, [-1, self.bits_per_symbol])

        sym_err_bool = tf.reduce_any(
            tf.not_equal(bits_tx_sym, bits_rx_sym),
            axis=1
        )

        errors = tf.reduce_sum(tf.cast(sym_err_bool, tf.float32))
        total = tf.cast(tf.shape(bits_tx_sym)[0], tf.float32)

        return errors, total

    def alamouti_channel(self, x_symbols, no):
        """
        Alamouti 2Tx-2Rx over Rayleigh + AWGN.

        Returns:
        - z_comb: combined numerator
        - denom: channel energy
        - x_no_eq: direct received signal baseline
        - no_no_eq: noise variance baseline
        """

        x = tf.reshape(x_symbols, [-1, self.num_data_symbols])
        batch_size = tf.shape(x)[0]

        x_pair = tf.reshape(x, [batch_size, self.num_pairs, 2])

        s1 = x_pair[:, :, 0]
        s2 = x_pair[:, :, 1]

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

        h1 = h[:, :, :, 0]
        h2 = h[:, :, :, 1]

        alpha_c = tf.cast(self.alpha, tf.complex64)

        a1 = alpha_c * h1
        a2 = alpha_c * h2

        s1_e = tf.expand_dims(s1, axis=-1)
        s2_e = tf.expand_dims(s2, axis=-1)

        y1_clean = a1 * s1_e + a2 * s2_e
        y2_clean = -a1 * tf.math.conj(s2_e) + a2 * tf.math.conj(s1_e)

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

        z1 = tf.reduce_sum(
            tf.math.conj(a1) * y1 + a2 * tf.math.conj(y2),
            axis=2
        )

        z2 = tf.reduce_sum(
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

        z_pair = tf.stack([z1, z2], axis=-1)

        z_comb = tf.reshape(
            z_pair,
            [batch_size, self.num_data_symbols]
        )

        denom_pair = tf.stack([denom, denom], axis=-1)

        denom_flat = tf.reshape(
            denom_pair,
            [batch_size, self.num_data_symbols]
        )

        z_comb = tf.reshape(
            z_comb,
            [batch_size, 1, 1, self.num_data_symbols]
        )

        denom_flat = tf.reshape(
            denom_flat,
            [batch_size, 1, 1, self.num_data_symbols]
        )

        y1_rx0 = y1[:, :, 0]
        y2_rx0 = y2[:, :, 0]

        x_no_pair = tf.stack(
            [y1_rx0, tf.math.conj(y2_rx0)],
            axis=-1
        )

        x_no_eq = tf.reshape(
            x_no_pair,
            [batch_size, self.num_data_symbols]
        )

        x_no_eq = tf.reshape(
            x_no_eq,
            [batch_size, 1, 1, self.num_data_symbols]
        )

        no_no_eq = tf.ones_like(
            tf.math.real(x_no_eq),
            dtype=tf.float32
        ) * no

        return z_comb, denom_flat, x_no_eq, no_no_eq

    def apply_equalizer(self, z_comb, denom, x_no_eq, no_no_eq, no, mode):
        mode = mode.lower()

        if mode == "zf":
            x_hat = z_comb / tf.cast(denom, tf.complex64)
            no_eff = no / denom
            return x_hat, no_eff

        if mode == "mmse":
            denom_mmse = denom + no
            x_hat = z_comb / tf.cast(denom_mmse, tf.complex64)

            no_eff = (no * denom) / (denom_mmse ** 2)
            no_eff = tf.maximum(no_eff, tf.constant(1e-12, dtype=tf.float32))

            return x_hat, no_eff

        if mode == "none":
            return x_no_eq, no_no_eq

        raise ValueError("Unknown equalizer mode. Use: zf, mmse, none")

    def evaluate_coded(self, bits_info, coded_bits, x_hat, no_eff):
        llr = self.demapper(x_hat, no_eff)

        bits_info_hat = self.decoder(llr)

        ber_err, ber_total = self.count_bit_errors(
            bits_info,
            bits_info_hat
        )

        coded_bits_hat = self.hard_bits_from_llr(llr)

        ser_err, ser_total = self.count_symbol_errors_from_bits(
            coded_bits,
            coded_bits_hat
        )

        return ber_err, ber_total, ser_err, ser_total

    def evaluate_uncoded(self, bits_u, x_hat, no_eff):
        llr = self.demapper(x_hat, no_eff)

        bits_u_hat = self.hard_bits_from_llr(llr)

        ber_err, ber_total = self.count_bit_errors(
            bits_u,
            bits_u_hat
        )

        ser_err, ser_total = self.count_symbol_errors_from_bits(
            bits_u,
            bits_u_hat
        )

        return ber_err, ber_total, ser_err, ser_total

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

        bits_info = tf.random.uniform(
            [batch_size, 1, 1, self.k],
            minval=0,
            maxval=2,
            dtype=tf.int32
        )

        bits_info = tf.cast(bits_info, tf.float32)

        coded_bits = self.encoder(bits_info)
        x_coded = self.mapper(coded_bits)

        z_c, denom_c, x_c_no, no_c_no = self.alamouti_channel(
            x_coded,
            no_coded
        )

        bits_u = tf.random.uniform(
            [batch_size, 1, 1, self.n],
            minval=0,
            maxval=2,
            dtype=tf.int32
        )

        bits_u = tf.cast(bits_u, tf.float32)

        x_u = self.mapper(bits_u)

        z_u, denom_u, x_u_no, no_u_no = self.alamouti_channel(
            x_u,
            no_uncoded
        )

        x_c_zf, no_c_zf = self.apply_equalizer(
            z_c,
            denom_c,
            x_c_no,
            no_c_no,
            no_coded,
            "zf"
        )

        x_u_zf, no_u_zf = self.apply_equalizer(
            z_u,
            denom_u,
            x_u_no,
            no_u_no,
            no_uncoded,
            "zf"
        )

        x_c_mmse, no_c_mmse = self.apply_equalizer(
            z_c,
            denom_c,
            x_c_no,
            no_c_no,
            no_coded,
            "mmse"
        )

        x_u_mmse, no_u_mmse = self.apply_equalizer(
            z_u,
            denom_u,
            x_u_no,
            no_u_no,
            no_uncoded,
            "mmse"
        )

        x_c_none, no_c_none = self.apply_equalizer(
            z_c,
            denom_c,
            x_c_no,
            no_c_no,
            no_coded,
            "none"
        )

        x_u_none, no_u_none = self.apply_equalizer(
            z_u,
            denom_u,
            x_u_no,
            no_u_no,
            no_uncoded,
            "none"
        )

        c_zf_ber_e, c_zf_ber_t, c_zf_ser_e, c_zf_ser_t = self.evaluate_coded(
            bits_info,
            coded_bits,
            x_c_zf,
            no_c_zf
        )

        u_zf_ber_e, u_zf_ber_t, u_zf_ser_e, u_zf_ser_t = self.evaluate_uncoded(
            bits_u,
            x_u_zf,
            no_u_zf
        )

        c_mmse_ber_e, c_mmse_ber_t, c_mmse_ser_e, c_mmse_ser_t = self.evaluate_coded(
            bits_info,
            coded_bits,
            x_c_mmse,
            no_c_mmse
        )

        u_mmse_ber_e, u_mmse_ber_t, u_mmse_ser_e, u_mmse_ser_t = self.evaluate_uncoded(
            bits_u,
            x_u_mmse,
            no_u_mmse
        )

        c_none_ber_e, c_none_ber_t, c_none_ser_e, c_none_ser_t = self.evaluate_coded(
            bits_info,
            coded_bits,
            x_c_none,
            no_c_none
        )

        u_none_ber_e, u_none_ber_t, u_none_ser_e, u_none_ser_t = self.evaluate_uncoded(
            bits_u,
            x_u_none,
            no_u_none
        )

        return (
            c_zf_ber_e, c_zf_ber_t, c_zf_ser_e, c_zf_ser_t,
            u_zf_ber_e, u_zf_ber_t, u_zf_ser_e, u_zf_ser_t,

            c_mmse_ber_e, c_mmse_ber_t, c_mmse_ser_e, c_mmse_ser_t,
            u_mmse_ber_e, u_mmse_ber_t, u_mmse_ser_e, u_mmse_ser_t,

            c_none_ber_e, c_none_ber_t, c_none_ser_e, c_none_ser_t,
            u_none_ber_e, u_none_ber_t, u_none_ser_e, u_none_ser_t,
        )

    def _metric(self, errors, total):
        errors = float(errors)
        total = float(total)

        if total <= 0:
            return {
                "value": float("nan"),
                "errors": errors,
                "total": total,
                "upper_bound": True,
            }

        if errors <= 0:
            return {
                "value": 1.0 / total,
                "errors": errors,
                "total": total,
                "upper_bound": True,
            }

        return {
            "value": errors / total,
            "errors": errors,
            "total": total,
            "upper_bound": False,
        }

    def run_monte_carlo(
        self,
        ebno_db,
        min_errors=500,
        max_bits=5e6,
        batch_size=64,
    ):
        keys = [
            "c_zf_ber_e", "c_zf_ber_t", "c_zf_ser_e", "c_zf_ser_t",
            "u_zf_ber_e", "u_zf_ber_t", "u_zf_ser_e", "u_zf_ser_t",

            "c_mmse_ber_e", "c_mmse_ber_t", "c_mmse_ser_e", "c_mmse_ser_t",
            "u_mmse_ber_e", "u_mmse_ber_t", "u_mmse_ser_e", "u_mmse_ser_t",

            "c_none_ber_e", "c_none_ber_t", "c_none_ser_e", "c_none_ser_t",
            "u_none_ber_e", "u_none_ber_t", "u_none_ser_e", "u_none_ser_t",
        ]

        acc = {k: 0.0 for k in keys}

        while (
            (
                acc["c_zf_ber_e"] < min_errors
                or acc["u_zf_ber_e"] < min_errors
                or acc["c_mmse_ber_e"] < min_errors
                or acc["u_mmse_ber_e"] < min_errors
            )
            and acc["c_zf_ber_t"] < max_bits
        ):
            result = self.process_batch(
                int(batch_size),
                tf.constant(float(ebno_db), dtype=tf.float32)
            )

            for key, value in zip(keys, result):
                acc[key] += float(value.numpy())

        return {
            "zf": {
                "coded_ber": self._metric(acc["c_zf_ber_e"], acc["c_zf_ber_t"]),
                "coded_ser": self._metric(acc["c_zf_ser_e"], acc["c_zf_ser_t"]),
                "uncoded_ber": self._metric(acc["u_zf_ber_e"], acc["u_zf_ber_t"]),
                "uncoded_ser": self._metric(acc["u_zf_ser_e"], acc["u_zf_ser_t"]),
            },
            "mmse": {
                "coded_ber": self._metric(acc["c_mmse_ber_e"], acc["c_mmse_ber_t"]),
                "coded_ser": self._metric(acc["c_mmse_ser_e"], acc["c_mmse_ser_t"]),
                "uncoded_ber": self._metric(acc["u_mmse_ber_e"], acc["u_mmse_ber_t"]),
                "uncoded_ser": self._metric(acc["u_mmse_ser_e"], acc["u_mmse_ser_t"]),
            },
            "none": {
                "coded_ber": self._metric(acc["c_none_ber_e"], acc["c_none_ber_t"]),
                "coded_ser": self._metric(acc["c_none_ser_e"], acc["c_none_ser_t"]),
                "uncoded_ber": self._metric(acc["u_none_ber_e"], acc["u_none_ber_t"]),
                "uncoded_ser": self._metric(acc["u_none_ser_e"], acc["u_none_ser_t"]),
            },
        }