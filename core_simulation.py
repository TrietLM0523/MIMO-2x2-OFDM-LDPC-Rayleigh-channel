import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.channel import RayleighBlockFading, OFDMChannel, AWGN
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LMMSEEqualizer


class MIMOSystem:
    def __init__(self, code_rate=0.5):
        # =========================
        # 1. Tham số hệ thống MIMO 2x2
        # =========================
        self.num_tx_ant = 2
        self.num_rx_ant = 2 
        # 2 antenna phát và 2 antenna thu, 
        # nhưng chỉ dùng 1 stream để đơn giản hóa mô phỏng. 
        # Nếu muốn spatial multiplexing thực sự, cần thay đổi stream management và decoder để hỗ trợ multi-stream.

        # 64-QAM: 6 bit / symbol
        self.bits_per_symbol = 6

        # =========================
        # 2. Resource Grid OFDM
        # =========================
        self.rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=15e3,
            num_tx=1,
            num_streams_per_tx=self.num_tx_ant,
            cyclic_prefix_length=16
        )

        self.num_data_symbols = int(self.rg.num_data_symbols)

        # n: số bit sau mã hóa LDPC
        self.n = self.num_data_symbols * self.bits_per_symbol

        # k: số bit thông tin trước mã hóa LDPC
        self.k = int(self.n * code_rate)

        self.code_rate = self.k / self.n

        print("===== System Parameters =====")
        print(f"Tx antennas              : {self.num_tx_ant}")
        print(f"Rx antennas              : {self.num_rx_ant}")
        print(f"Modulation               : 64-QAM")
        print(f"Bits per symbol          : {self.bits_per_symbol}")
        print(f"Number of data symbols   : {self.num_data_symbols}")
        print(f"LDPC k                   : {self.k}")
        print(f"LDPC n                   : {self.n}")
        print(f"Code rate                : {self.code_rate:.4f}")
        print("=============================")

        # =========================
        # 3. LDPC Encoder / Decoder
        # =========================
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(
            self.encoder,
            num_iter=50,
            hard_out=True
        )

        # =========================
        # 4. Mapper / Demapper QAM
        # =========================
        self.mapper = Mapper("qam", self.bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.bits_per_symbol)

        # =========================
        # 5. OFDM Mapper
        # =========================
        self.rg_mapper = ResourceGridMapper(self.rg)

        # =========================
        # 6. Rayleigh + OFDM Channel
        # =========================
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

        # =========================
        # 7. Stream Management + LMMSE Equalizer
        # =========================
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
        Tính SER bằng cách gom từng nhóm bits_per_symbol bit.

        Với 64-QAM:
            1 symbol = 6 bit

        Một symbol được xem là sai nếu trong 6 bit đó
        có ít nhất 1 bit sai.

        bits_tx: bit gốc trước mapper
        llr_rx : LLR sau demapper
        """

        # Hard decision từ LLR
        bits_rx = tf.cast(tf.math.greater(llr_rx, 0.0), tf.float32)

        # Reshape thành từng nhóm 6 bit/symbol
        bits_tx_sym = tf.reshape(bits_tx, [-1, self.bits_per_symbol])
        bits_rx_sym = tf.reshape(bits_rx, [-1, self.bits_per_symbol])

        # Một symbol sai nếu có ít nhất một bit sai
        sym_err_bool = tf.reduce_any(
            tf.not_equal(bits_tx_sym, bits_rx_sym),
            axis=1
        )

        num_sym_err = tf.reduce_sum(tf.cast(sym_err_bool, tf.float32))
        num_symbols = tf.cast(tf.shape(bits_tx_sym)[0], tf.float32)

        return num_sym_err, num_symbols

    @tf.function
    def process_batch(self, batch_size, ebno_linear):
        # ==========================================================
        # LUỒNG 1: CODED SYSTEM
        # Bit -> LDPC -> 64-QAM -> OFDM -> Rayleigh + AWGN
        # -> LMMSE -> Demapper -> LDPC Decoder -> BER
        # ==========================================================

        # Tạo bit thông tin
        bits = tf.random.uniform(
            [batch_size, 1, self.num_tx_ant, self.k],
            0,
            2,
            dtype=tf.int32
        )

        bits_float = tf.cast(bits, tf.float32)

        # Noise variance cho coded system
        no_coded = 1.0 / (
            self.bits_per_symbol * self.code_rate * ebno_linear
        )
        no_coded_tensor = tf.cast(no_coded, tf.float32)

        # LDPC encode
        coded_bits = self.encoder(bits_float)

        # 64-QAM mapping
        x_coded = self.mapper(coded_bits)

        # OFDM resource grid mapping
        x_rg_coded = self.rg_mapper(x_coded)

        # Rayleigh channel
        y_rg_clean_coded, h_freq_coded = self.channel(x_rg_coded)

        # AWGN
        y_rg_coded = self.awgn(y_rg_clean_coded, no_coded_tensor)

        # Perfect CSI error variance
        err_var_coded = tf.zeros_like(h_freq_coded, dtype=tf.float32)

        # LMMSE equalization
        x_hat_c, no_eff_c = self.equalizer(
            y_rg_coded,
            h_freq_coded,
            err_var_coded,
            no_coded_tensor
        )

        # Soft demapping
        llr_c = self.demapper(x_hat_c, no_eff_c)

        # LDPC decoding
        bits_est_c = self.decoder(llr_c)

        # BER coded
        err_c = tf.reduce_sum(
            tf.cast(tf.not_equal(bits_float, bits_est_c), tf.float32)
        )

        num_bits_c = tf.cast(tf.size(bits_float), tf.float32)

        # SER coded, tính trước LDPC decoder
        sym_err_c, num_sym_c = self.compute_ser_from_bits(
            coded_bits,
            llr_c
        )

        # ==========================================================
        # LUỒNG 2: UNCODED SYSTEM
        # Bit -> 64-QAM -> OFDM -> Rayleigh + AWGN
        # -> LMMSE -> Demapper -> BER/SER
        # ==========================================================

        # Với uncoded, số bit phải bằng n để mapper map đủ resource grid
        bits_u = tf.random.uniform(
            [batch_size, 1, self.num_tx_ant, self.n],
            0,
            2,
            dtype=tf.int32
        )

        bits_u_float = tf.cast(bits_u, tf.float32)

        # Noise variance cho uncoded system
        no_uncoded = 1.0 / (
            self.bits_per_symbol * ebno_linear
        )
        no_uncoded_tensor = tf.cast(no_uncoded, tf.float32)

        # 64-QAM mapping
        x_u = self.mapper(bits_u_float)

        # OFDM resource grid mapping
        x_rg_u = self.rg_mapper(x_u)

        # Rayleigh channel
        y_rg_clean_u, h_freq_u = self.channel(x_rg_u)

        # AWGN
        y_rg_u = self.awgn(y_rg_clean_u, no_uncoded_tensor)

        # Perfect CSI error variance
        err_var_u = tf.zeros_like(h_freq_u, dtype=tf.float32)

        # LMMSE equalization
        x_hat_u, no_eff_u = self.equalizer(
            y_rg_u,
            h_freq_u,
            err_var_u,
            no_uncoded_tensor
        )

        # Soft demapping
        llr_u = self.demapper(x_hat_u, no_eff_u)

        # Hard decision
        bits_est_u = tf.cast(tf.math.greater(llr_u, 0.0), tf.float32)

        # BER uncoded
        err_u = tf.reduce_sum(
            tf.cast(tf.not_equal(bits_u_float, bits_est_u), tf.float32)
        )

        num_bits_u = tf.cast(tf.size(bits_u_float), tf.float32)

        # SER uncoded
        sym_err_u, num_sym_u = self.compute_ser_from_bits(
            bits_u_float,
            llr_u
        )

        return (
            err_c,
            num_bits_c,
            sym_err_c,
            num_sym_c,
            err_u,
            num_bits_u,
            sym_err_u,
            num_sym_u
        )

    def run_monte_carlo(
        self,
        ebno_db,
        min_errors=500,
        max_bits=5e6,
        batch_size=64
    ):
        ebno_linear = 10.0 ** (ebno_db / 10.0)

        total_err_c = 0.0
        total_bits_c = 0.0
        total_sym_err_c = 0.0
        total_sym_c = 0.0

        total_err_u = 0.0
        total_bits_u = 0.0
        total_sym_err_u = 0.0
        total_sym_u = 0.0

        while (
            (total_err_c < min_errors or total_err_u < min_errors)
            and total_bits_c < max_bits
        ):
            (
                e_c,
                b_c,
                se_c,
                ns_c,
                e_u,
                b_u,
                se_u,
                ns_u
            ) = self.process_batch(batch_size, ebno_linear)

            total_err_c += e_c.numpy()
            total_bits_c += b_c.numpy()
            total_sym_err_c += se_c.numpy()
            total_sym_c += ns_c.numpy()

            total_err_u += e_u.numpy()
            total_bits_u += b_u.numpy()
            total_sym_err_u += se_u.numpy()
            total_sym_u += ns_u.numpy()

            # Tránh chạy quá lâu ở SNR cao
            if total_err_c >= min_errors and ebno_db > 16:
                break

        ber_c = max(total_err_c, 1.0) / total_bits_c
        ser_c = max(total_sym_err_c, 1.0) / total_sym_c

        ber_u = max(total_err_u, 1.0) / total_bits_u
        ser_u = max(total_sym_err_u, 1.0) / total_sym_u

        return ber_c, ser_c, ber_u, ser_u


def main():
    # =========================
    # Cấu hình mô phỏng
    # =========================
    ebno_dbs = np.arange(0, 21, 2)

    system = MIMOSystem(code_rate=0.5)

    ber_coded = []
    ser_coded = []
    ber_uncoded = []
    ser_uncoded = []

    start_time = time.time()

    print("\n===== Start Monte Carlo Simulation =====")

    for ebno_db in ebno_dbs:
        t0 = time.time()

        ber_c, ser_c, ber_u, ser_u = system.run_monte_carlo(
            ebno_db=ebno_db,
            min_errors=500,
            max_bits=5e6,
            batch_size=64
        )

        ber_coded.append(ber_c)
        ser_coded.append(ser_c)
        ber_uncoded.append(ber_u)
        ser_uncoded.append(ser_u)

        elapsed = time.time() - t0

        print(
            f"Eb/N0 = {ebno_db:>2} dB | "
            f"BER coded = {ber_c:.4e} | "
            f"SER coded = {ser_c:.4e} | "
            f"BER uncoded = {ber_u:.4e} | "
            f"SER uncoded = {ser_u:.4e} | "
            f"time = {elapsed:.2f}s"
        )

    total_time = time.time() - start_time

    print("===== Simulation Finished =====")
    print(f"Total time: {total_time:.2f}s")

    # =========================
    # Vẽ BER
    # =========================
    plt.figure()
    plt.semilogy(ebno_dbs, ber_coded, "o-", label="BER coded LDPC")
    plt.semilogy(ebno_dbs, ber_uncoded, "s-", label="BER uncoded")
    plt.grid(True, which="both")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.title("BER of 2x2 MIMO-OFDM over Rayleigh + AWGN")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =========================
    # Vẽ SER
    # =========================
    plt.figure()
    plt.semilogy(ebno_dbs, ser_coded, "o-", label="SER coded")
    plt.semilogy(ebno_dbs, ser_uncoded, "s-", label="SER uncoded")
    plt.grid(True, which="both")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("SER")
    plt.title("SER of 2x2 MIMO-OFDM over Rayleigh + AWGN")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()