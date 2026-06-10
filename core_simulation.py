import tensorflow as tf

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper

try:
    from sionna.phy.utils import ebnodb2no
except Exception:
    from sionna.utils import ebnodb2no


class AlamoutiSystem:
    """
    2x2 MIMO-OFDM Alamouti/STBC system with:
    - LDPC 5G
    - 64-QAM
    - OFDM IFFT / CP / FFT
    - Rayleigh fading + AWGN per OFDM subcarrier
    - ZF / MMSE / No Equalizer
    - BER/SER evaluation
    """

    def __init__(
        self,
        code_rate=0.5,
        decoder_iter=15,
        num_data_symbols=512,
        fft_size=64,
        cp_length=16,
    ):
        self.num_tx_ant = 2
        self.num_rx_ant = 2
        self.bits_per_symbol = 6  # 64-QAM = log2(64) = 6 bits/symbol

        self.fft_size = int(fft_size)
        self.cp_length = int(cp_length)

        requested_symbols = int(num_data_symbols)

        # Need full OFDM symbols and an even number of OFDM symbols
        self.num_ofdm_symbols = (requested_symbols + self.fft_size - 1) // self.fft_size
        if self.num_ofdm_symbols % 2 != 0:
            self.num_ofdm_symbols += 1

        self.num_ofdm_pairs = self.num_ofdm_symbols // 2
        self.num_data_symbols = self.num_ofdm_symbols * self.fft_size

        # LDPC codeword length
        self.n = self.num_data_symbols * self.bits_per_symbol

        # Information length
        self.k = int(self.n * code_rate)

        # Actual code rate after integer rounding
        self.code_rate = self.k / self.n

        self.encoder = LDPC5GEncoder(self.k, self.n)

        self.decoder = LDPC5GDecoder(
            self.encoder,
            num_iter=int(decoder_iter),
            hard_out=True,
        )

        self.mapper = Mapper("qam", self.bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.bits_per_symbol)

        # Total transmit power normalization for 2 Tx antennas
        self.alpha = tf.constant(1.0 / (2.0**0.5), dtype=tf.float32)

    def compute_noise_variance(self, ebno_db, coderate):
        ebno_db = tf.cast(ebno_db, tf.float32)

        no = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.bits_per_symbol,
            coderate=coderate,
        )

        return tf.cast(no, tf.float32)

    def hard_bits_from_llr(self, llr):
        return tf.cast(tf.math.greater(llr, 0.0), tf.float32)

    def count_bit_errors(self, bits_tx, bits_rx):
        errors = tf.reduce_sum(tf.cast(tf.not_equal(bits_tx, bits_rx), tf.float32))
        total = tf.cast(tf.size(bits_tx), tf.float32)
        return errors, total

    def count_symbol_errors_from_bits(self, bits_tx, bits_rx):
        total_bits = tf.shape(bits_tx)[-1]
        usable_bits = (total_bits // self.bits_per_symbol) * self.bits_per_symbol

        bits_tx_cut = bits_tx[..., :usable_bits]
        bits_rx_cut = bits_rx[..., :usable_bits]

        bits_tx_sym = tf.reshape(bits_tx_cut, [-1, self.bits_per_symbol])
        bits_rx_sym = tf.reshape(bits_rx_cut, [-1, self.bits_per_symbol])

        sym_err_bool = tf.reduce_any(tf.not_equal(bits_tx_sym, bits_rx_sym), axis=1)
        errors = tf.reduce_sum(tf.cast(sym_err_bool, tf.float32))
        total = tf.cast(tf.shape(bits_tx_sym)[0], tf.float32)

        return errors, total

    def ofdm_modulate(self, x_fd):
        """
        x_fd shape: [batch, ofdm_symbols, tx_ant, fft_size]
        return: [batch, ofdm_symbols, tx_ant, fft_size + cp_length]
        """
        sqrt_fft = tf.cast(tf.sqrt(tf.cast(self.fft_size, tf.float32)), tf.complex64)
        x_td = tf.signal.ifft(x_fd) * sqrt_fft
        cp = x_td[..., -self.cp_length :]
        return tf.concat([cp, x_td], axis=-1)

    def ofdm_demodulate(self, y_cp):
        """
        y_cp shape: [batch, ofdm_symbols, rx_or_tx_ant, fft_size + cp_length]
        return: [batch, ofdm_symbols, rx_or_tx_ant, fft_size]
        """
        sqrt_fft = tf.cast(tf.sqrt(tf.cast(self.fft_size, tf.float32)), tf.complex64)
        y_td = y_cp[..., self.cp_length :]
        return tf.signal.fft(y_td) / sqrt_fft

    def alamouti_ofdm_channel(self, x_symbols, no):
        """
        2x2 Alamouti/STBC over OFDM Rayleigh + AWGN.

        OFDM is explicitly included:
        1. Map QAM symbols to OFDM resource grid
        2. Apply Alamouti coding over pairs of OFDM symbols
        3. IFFT
        4. Add cyclic prefix
        5. Remove cyclic prefix
        6. FFT
        7. Apply equivalent per-subcarrier Rayleigh channel + AWGN
        8. Alamouti/STBC combining per subcarrier
        """

        x = tf.reshape(x_symbols, [-1, self.num_data_symbols])
        batch_size = tf.shape(x)[0]

        # Resource grid: [B, OFDM symbols, subcarriers]
        x_grid = tf.reshape(x, [batch_size, self.num_ofdm_symbols, self.fft_size])

        # Alamouti works over pairs of OFDM symbols on each subcarrier
        x_pair = tf.reshape(
            x_grid,
            [batch_size, self.num_ofdm_pairs, 2, self.fft_size],
        )

        s1 = x_pair[:, :, 0, :]
        s2 = x_pair[:, :, 1, :]

        alpha_c = tf.cast(self.alpha, tf.complex64)

        # Frequency-domain transmit grid after Alamouti/STBC coding
        # Shape: [B, pair, time_in_pair, Tx, F]
        x_t0_tx0 = alpha_c * s1
        x_t0_tx1 = alpha_c * s2
        x_t1_tx0 = -alpha_c * tf.math.conj(s2)
        x_t1_tx1 = alpha_c * tf.math.conj(s1)

        t0 = tf.stack([x_t0_tx0, x_t0_tx1], axis=2)  # [B, pair, Tx, F]
        t1 = tf.stack([x_t1_tx0, x_t1_tx1], axis=2)  # [B, pair, Tx, F]

        x_tx_pair = tf.stack([t0, t1], axis=2)  # [B, pair, 2, Tx, F]
        x_tx_fd = tf.reshape(
            x_tx_pair,
            [batch_size, self.num_ofdm_symbols, self.num_tx_ant, self.fft_size],
        )

        # OFDM transmitter: IFFT + CP
        x_tx_cp = self.ofdm_modulate(x_tx_fd)

        # OFDM receiver front-end for the transmitted grid: remove CP + FFT.
        # With CP, the multipath channel becomes a per-subcarrier multiplication.
        x_tx_after_fft = self.ofdm_demodulate(x_tx_cp)
        x_tx_after_fft = tf.reshape(
            x_tx_after_fft,
            [batch_size, self.num_ofdm_pairs, 2, self.num_tx_ant, self.fft_size],
        )

        # Rayleigh channel per OFDM subcarrier, constant over each Alamouti pair
        h_real = tf.random.normal(
            [
                batch_size,
                self.num_ofdm_pairs,
                self.num_rx_ant,
                self.num_tx_ant,
                self.fft_size,
            ],
            dtype=tf.float32,
        )

        h_imag = tf.random.normal(
            [
                batch_size,
                self.num_ofdm_pairs,
                self.num_rx_ant,
                self.num_tx_ant,
                self.fft_size,
            ],
            dtype=tf.float32,
        )

        h = tf.complex(h_real, h_imag) / tf.cast(
            tf.sqrt(tf.constant(2.0, dtype=tf.float32)),
            tf.complex64,
        )

        # Received OFDM grid after FFT, per pair and time slot
        # y[b,p,t,rx,f] = sum_tx h[b,p,rx,tx,f] * x[b,p,t,tx,f] + n
        y_clean = tf.reduce_sum(
            tf.expand_dims(h, axis=2) * tf.expand_dims(x_tx_after_fft, axis=3),
            axis=4,
        )

        noise_std = tf.sqrt(no / 2.0)
        noise_std_c = tf.cast(noise_std, tf.complex64)

        noise = noise_std_c * tf.complex(
            tf.random.normal(tf.shape(y_clean), dtype=tf.float32),
            tf.random.normal(tf.shape(y_clean), dtype=tf.float32),
        )

        y = y_clean + noise

        y1 = y[:, :, 0, :, :]
        y2 = y[:, :, 1, :, :]

        # Effective channels include the 1/sqrt(2) transmit-power normalization
        a1 = alpha_c * h[:, :, :, 0, :]
        a2 = alpha_c * h[:, :, :, 1, :]

        # Alamouti/STBC combiner per subcarrier
        z1 = tf.reduce_sum(tf.math.conj(a1) * y1 + a2 * tf.math.conj(y2), axis=2)
        z2 = tf.reduce_sum(tf.math.conj(a2) * y1 - a1 * tf.math.conj(y2), axis=2)

        denom = tf.reduce_sum(tf.abs(a1) ** 2 + tf.abs(a2) ** 2, axis=2)
        denom = tf.maximum(denom, tf.constant(1e-12, dtype=tf.float32))

        z_pair = tf.stack([z1, z2], axis=2)  # [B, pair, 2, F]
        z_grid = tf.reshape(z_pair, [batch_size, self.num_ofdm_symbols, self.fft_size])
        z_comb = tf.reshape(z_grid, [batch_size, self.num_data_symbols])
        z_comb = tf.reshape(z_comb, [batch_size, 1, 1, self.num_data_symbols])

        denom_pair = tf.stack([denom, denom], axis=2)
        denom_grid = tf.reshape(
            denom_pair,
            [batch_size, self.num_ofdm_symbols, self.fft_size],
        )
        denom_flat = tf.reshape(denom_grid, [batch_size, self.num_data_symbols])
        denom_flat = tf.reshape(denom_flat, [batch_size, 1, 1, self.num_data_symbols])

        # Optional no-equalizer baseline from Rx0 only
        y1_rx0 = y[:, :, 0, 0, :]
        y2_rx0 = y[:, :, 1, 0, :]
        x_no_pair = tf.stack([y1_rx0, tf.math.conj(y2_rx0)], axis=2)
        x_no_grid = tf.reshape(
            x_no_pair,
            [batch_size, self.num_ofdm_symbols, self.fft_size],
        )
        x_no_eq = tf.reshape(x_no_grid, [batch_size, self.num_data_symbols])
        x_no_eq = tf.reshape(x_no_eq, [batch_size, 1, 1, self.num_data_symbols])

        no_no_eq = tf.ones_like(tf.math.real(x_no_eq), dtype=tf.float32) * no

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
            no_eff = (no * denom) / (denom_mmse**2)
            no_eff = tf.maximum(no_eff, tf.constant(1e-12, dtype=tf.float32))
            return x_hat, no_eff

        if mode == "none":
            return x_no_eq, no_no_eq

        raise ValueError("Unknown equalizer mode. Use: zf, mmse, none")

    def evaluate_coded(self, bits_info, coded_bits, x_hat, no_eff):
        llr = self.demapper(x_hat, no_eff)
        bits_info_hat = self.decoder(llr)

        ber_err, ber_total = self.count_bit_errors(bits_info, bits_info_hat)

        # SER after LDPC decoder: group decoded information bits into 6-bit blocks
        ser_err, ser_total = self.count_symbol_errors_from_bits(bits_info, bits_info_hat)

        return ber_err, ber_total, ser_err, ser_total

    def evaluate_uncoded(self, bits_u, x_hat, no_eff):
        llr = self.demapper(x_hat, no_eff)
        bits_u_hat = self.hard_bits_from_llr(llr)

        ber_err, ber_total = self.count_bit_errors(bits_u, bits_u_hat)
        ser_err, ser_total = self.count_symbol_errors_from_bits(bits_u, bits_u_hat)

        return ber_err, ber_total, ser_err, ser_total

    @tf.function
    def process_batch(self, batch_size, ebno_db):
        no_coded = self.compute_noise_variance(ebno_db, coderate=self.code_rate)
        no_uncoded = self.compute_noise_variance(ebno_db, coderate=1.0)

        bits_info = tf.random.uniform(
            [batch_size, 1, 1, self.k],
            minval=0,
            maxval=2,
            dtype=tf.int32,
        )
        bits_info = tf.cast(bits_info, tf.float32)

        coded_bits = self.encoder(bits_info)
        x_coded = self.mapper(coded_bits)

        z_c, denom_c, x_c_no, no_c_no = self.alamouti_ofdm_channel(x_coded, no_coded)

        bits_u = tf.random.uniform(
            [batch_size, 1, 1, self.n],
            minval=0,
            maxval=2,
            dtype=tf.int32,
        )
        bits_u = tf.cast(bits_u, tf.float32)

        x_u = self.mapper(bits_u)

        z_u, denom_u, x_u_no, no_u_no = self.alamouti_ofdm_channel(x_u, no_uncoded)

        x_c_zf, no_c_zf = self.apply_equalizer(z_c, denom_c, x_c_no, no_c_no, no_coded, "zf")
        x_u_zf, no_u_zf = self.apply_equalizer(z_u, denom_u, x_u_no, no_u_no, no_uncoded, "zf")

        x_c_mmse, no_c_mmse = self.apply_equalizer(z_c, denom_c, x_c_no, no_c_no, no_coded, "mmse")
        x_u_mmse, no_u_mmse = self.apply_equalizer(z_u, denom_u, x_u_no, no_u_no, no_uncoded, "mmse")

        x_c_none, no_c_none = self.apply_equalizer(z_c, denom_c, x_c_no, no_c_no, no_coded, "none")
        x_u_none, no_u_none = self.apply_equalizer(z_u, denom_u, x_u_no, no_u_no, no_uncoded, "none")

        c_zf_ber_e, c_zf_ber_t, c_zf_ser_e, c_zf_ser_t = self.evaluate_coded(
            bits_info,
            coded_bits,
            x_c_zf,
            no_c_zf,
        )
        u_zf_ber_e, u_zf_ber_t, u_zf_ser_e, u_zf_ser_t = self.evaluate_uncoded(bits_u, x_u_zf, no_u_zf)

        c_mmse_ber_e, c_mmse_ber_t, c_mmse_ser_e, c_mmse_ser_t = self.evaluate_coded(
            bits_info,
            coded_bits,
            x_c_mmse,
            no_c_mmse,
        )
        u_mmse_ber_e, u_mmse_ber_t, u_mmse_ser_e, u_mmse_ser_t = self.evaluate_uncoded(
            bits_u,
            x_u_mmse,
            no_u_mmse,
        )

        c_none_ber_e, c_none_ber_t, c_none_ser_e, c_none_ser_t = self.evaluate_coded(
            bits_info,
            coded_bits,
            x_c_none,
            no_c_none,
        )
        u_none_ber_e, u_none_ber_t, u_none_ser_e, u_none_ser_t = self.evaluate_uncoded(
            bits_u,
            x_u_none,
            no_u_none,
        )

        return (
            c_zf_ber_e,
            c_zf_ber_t,
            c_zf_ser_e,
            c_zf_ser_t,
            u_zf_ber_e,
            u_zf_ber_t,
            u_zf_ser_e,
            u_zf_ser_t,
            c_mmse_ber_e,
            c_mmse_ber_t,
            c_mmse_ser_e,
            c_mmse_ser_t,
            u_mmse_ber_e,
            u_mmse_ber_t,
            u_mmse_ser_e,
            u_mmse_ser_t,
            c_none_ber_e,
            c_none_ber_t,
            c_none_ser_e,
            c_none_ser_t,
            u_none_ber_e,
            u_none_ber_t,
            u_none_ser_e,
            u_none_ser_t,
        )

    def _metric(self, errors, total):
        errors = float(errors)
        total = float(total)

        if total <= 0:
            return {"value": float("nan"), "errors": errors, "total": total, "upper_bound": True}

        if errors <= 0:
            return {"value": 1.0 / total, "errors": errors, "total": total, "upper_bound": True}

        return {"value": errors / total, "errors": errors, "total": total, "upper_bound": False}

    def run_monte_carlo(self, ebno_db, min_errors=500, max_bits=5e6, batch_size=64):
        keys = [
            "c_zf_ber_e",
            "c_zf_ber_t",
            "c_zf_ser_e",
            "c_zf_ser_t",
            "u_zf_ber_e",
            "u_zf_ber_t",
            "u_zf_ser_e",
            "u_zf_ser_t",
            "c_mmse_ber_e",
            "c_mmse_ber_t",
            "c_mmse_ser_e",
            "c_mmse_ser_t",
            "u_mmse_ber_e",
            "u_mmse_ber_t",
            "u_mmse_ser_e",
            "u_mmse_ser_t",
            "c_none_ber_e",
            "c_none_ber_t",
            "c_none_ser_e",
            "c_none_ser_t",
            "u_none_ber_e",
            "u_none_ber_t",
            "u_none_ser_e",
            "u_none_ser_t",
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
                tf.constant(float(ebno_db), dtype=tf.float32),
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
