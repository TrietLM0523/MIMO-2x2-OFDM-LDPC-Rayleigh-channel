import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core_simulation import MIMOSystem


st.set_page_config(
    page_title="Đề 5 - 2x2 MIMO-OFDM LDPC",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #f7f7f9;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Đề 5: Mô phỏng hệ thống 2x2 MIMO-OFDM sử dụng mã hóa LDPC")

st.markdown(
    """
    **Kênh truyền:** Rayleigh fading + nhiễu trắng AWGN  
    **Điều chế:** 64-QAM  
    **Đánh giá chất lượng:** BER/SER theo Eb/N0
    """
)

st.markdown("---")

with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")

    ebno_min = st.number_input(
        "Eb/N0 Min (dB)",
        value=0,
        step=1,
        key="ebno_min_input"
    )

    ebno_max = st.number_input(
        "Eb/N0 Max (dB)",
        value=18,
        step=1,
        key="ebno_max_input"
    )

    ebno_step = st.number_input(
        "Eb/N0 Bước (dB)",
        value=2,
        min_value=1,
        step=1,
        key="ebno_step_input"
    )

    code_rate_str = st.selectbox(
        "LDPC Code Rate",
        ["1/2", "2/3", "3/4"],
        key="code_rate_select"
    )

    rate_map = {
        "1/2": 1 / 2,
        "2/3": 2 / 3,
        "3/4": 3 / 4
    }

    code_rate = rate_map[code_rate_str]

    st.markdown("---")
    st.subheader("🎲 Monte Carlo")

    min_errs = st.slider(
        "Số lỗi tối thiểu",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        key="min_errors_slider"
    )

    max_bits = st.select_slider(
        "Giới hạn bit tối đa",
        options=[1e5, 5e5, 1e6, 5e6, 1e7, 2e7, 5e7],
        value=1e7,
        key="max_bits_slider"
    )

    batch_size = st.select_slider(
        "Batch size",
        options=[16, 32, 64, 128, 256],
        value=128,
        key="batch_size_slider"
    )

    run_btn = st.button(
        "🚀 CHẠY MÔ PHỎNG",
        use_container_width=True,
        key="run_simulation_button"
    )


if run_btn:
    if ebno_max < ebno_min:
        st.error("Eb/N0 Max phải lớn hơn hoặc bằng Eb/N0 Min.")
        st.stop()

    if ebno_step <= 0:
        st.error("Bước Eb/N0 phải lớn hơn 0.")
        st.stop()

    ebno_range = np.arange(
        ebno_min,
        ebno_max + 1,
        ebno_step
    )

    st.subheader("📌 Thông tin mô phỏng")

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)

    info_col1.metric("MIMO", "2x2")
    info_col2.metric("Điều chế", "64-QAM")
    info_col3.metric("LDPC Rate", code_rate_str)
    info_col4.metric("Equalizer chính", "LMMSE")

    with st.expander("🧠 Ghi chú mô hình và đường tham chiếu"):
        st.markdown(
            """
            - Hệ thống sử dụng **2 anten phát** và **2 anten thu**.
            - Mô hình MIMO hiện tại là **spatial multiplexing**, không phải Alamouti/STBC.
            - Bộ thu chính sử dụng **LMMSE Equalizer**.
            - Bộ thu giả sử biết hoàn hảo đáp ứng kênh, tức **perfect CSI**.
            - Noise variance được tính bằng hàm **ebnodb2no** của Sionna để phù hợp với ResourceGrid/OFDM.
            - Đường tham chiếu chính:
                - **LDPC coded + LMMSE**
                - **Uncoded + LMMSE reference**
            - Đường phụ:
                - **Uncoded without MIMO equalization**
            - Đường without equalization được dùng để minh họa vai trò của bộ cân bằng LMMSE trong hệ MIMO spatial multiplexing.
            - **SER before LDPC decoder** chỉ là SER ở tầng demapper trước sửa lỗi, nên không dùng làm đường chính để kết luận lợi ích LDPC.
            """
        )

    progress_bar = st.progress(0)
    status_text = st.empty()

    ber_coded = []
    ser_pre_ldpc_coded = []
    ber_uncoded = []
    ser_uncoded = []
    ber_no_eq = []
    ser_no_eq = []

    start_time = time.time()

    try:
        system = MIMOSystem(code_rate=code_rate)

        for i, ebno in enumerate(ebno_range):
            status_text.info(f"Đang mô phỏng Eb/N0 = {ebno} dB...")

            t0 = time.time()

            (
                ber_c,
                ser_pre_c,
                ber_u,
                ser_u,
                ber_ne,
                ser_ne
            ) = system.run_monte_carlo(
                ebno_db=float(ebno),
                min_errors=int(min_errs),
                max_bits=float(max_bits),
                batch_size=int(batch_size)
            )

            elapsed = time.time() - t0

            ber_coded.append(ber_c)
            ser_pre_ldpc_coded.append(ser_pre_c)
            ber_uncoded.append(ber_u)
            ser_uncoded.append(ser_u)
            ber_no_eq.append(ber_ne)
            ser_no_eq.append(ser_ne)

            progress_bar.progress((i + 1) / len(ebno_range))

            st.write(
                f"Eb/N0 = **{ebno} dB** | "
                f"BER coded + LMMSE = `{ber_c:.3e}` | "
                f"BER uncoded + LMMSE = `{ber_u:.3e}` | "
                f"BER uncoded no EQ = `{ber_ne:.3e}` | "
                f"SER uncoded + LMMSE = `{ser_u:.3e}` | "
                f"SER uncoded no EQ = `{ser_ne:.3e}` | "
                f"time = `{elapsed:.2f}s`"
            )

        total_time = time.time() - start_time
        status_text.success(f"✅ Mô phỏng hoàn tất trong {total_time:.2f} giây.")

    except Exception as e:
        st.error("Có lỗi khi chạy mô phỏng.")
        st.exception(e)
        st.stop()

    st.markdown("---")
    st.subheader("📈 Kết quả tổng quan")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("BER LDPC + LMMSE thấp nhất", f"{min(ber_coded):.2e}")
    m2.metric("BER Uncoded + LMMSE thấp nhất", f"{min(ber_uncoded):.2e}")
    m3.metric("BER No EQ thấp nhất", f"{min(ber_no_eq):.2e}")
    m4.metric("SER Uncoded + LMMSE thấp nhất", f"{min(ser_uncoded):.2e}")

    # ============================================================
    # BER figure
    # ============================================================

    fig_ber = go.Figure()

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ber_coded,
            mode="lines+markers",
            name=f"BER LDPC coded + LMMSE, R={code_rate_str}",
            line=dict(width=4),
            marker=dict(size=9, symbol="diamond")
        )
    )

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ber_uncoded,
            mode="lines+markers",
            name="BER uncoded + LMMSE reference",
            line=dict(width=3, dash="dash"),
            marker=dict(size=9, symbol="x")
        )
    )

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ber_no_eq,
            mode="lines+markers",
            name="BER uncoded without MIMO equalization",
            line=dict(width=3, dash="dot"),
            marker=dict(size=9, symbol="circle")
        )
    )

    fig_ber.update_layout(
        title="BER vs Eb/N0 - 2x2 MIMO-OFDM Rayleigh + AWGN, 64-QAM",
        xaxis_title="Eb/N0 (dB)",
        yaxis_title="Bit Error Rate (BER)",
        yaxis_type="log",
        template="plotly_white",
        hovermode="x unified",
        height=600
    )

    # ============================================================
    # SER figure
    # ============================================================

        # ============================================================
    # SER figure
    # SER có 3 đường:
    # 1. SER before LDPC decoder
    # 2. SER uncoded + LMMSE
    # 3. SER uncoded without equalization
    # ============================================================

    fig_ser = go.Figure()

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_pre_ldpc_coded,
            mode="lines+markers",
            name=f"SER before LDPC decoder, R={code_rate_str}",
            line=dict(width=4),
            marker=dict(size=9, symbol="diamond")
        )
    )

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_uncoded,
            mode="lines+markers",
            name="SER uncoded + LMMSE reference",
            line=dict(width=3, dash="dash"),
            marker=dict(size=9, symbol="x")
        )
    )

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_no_eq,
            mode="lines+markers",
            name="SER uncoded without MIMO equalization",
            line=dict(width=3, dash="dot"),
            marker=dict(size=9, symbol="circle")
        )
    )

    fig_ser.update_layout(
        title="SER vs Eb/N0 - Effect of LMMSE Equalization, 64-QAM",
        xaxis_title="Eb/N0 (dB)",
        yaxis_title="Symbol Error Rate (SER)",
        yaxis_type="log",
        template="plotly_white",
        hovermode="x unified",
        height=600
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📉 BER", "📊 SER Reference", "📋 Bảng số liệu", "📝 Mô tả báo cáo"]
    )

    with tab1:
        st.plotly_chart(fig_ber, use_container_width=True)

    with tab2:
        st.plotly_chart(fig_ser, use_container_width=True)

        st.info(
            "SER graph hiển thị 3 đường: SER before LDPC decoder, "
            "SER uncoded + LMMSE và SER uncoded without MIMO equalization. "
            "Đường without equalization có thể gần như nằm ngang ở mức lỗi cao, "
            "vì trong MIMO spatial multiplexing nếu không có equalizer thì các luồng tín hiệu bị trộn qua ma trận kênh Rayleigh."
        )

    with tab3:
        result_df = pd.DataFrame(
            {
                "Eb/N0 (dB)": ebno_range,
                "BER LDPC Coded + LMMSE": ber_coded,
                "BER Uncoded + LMMSE": ber_uncoded,
                "BER Uncoded without Equalization": ber_no_eq,
                "SER Uncoded + LMMSE": ser_uncoded,
                "SER Uncoded without Equalization": ser_no_eq,
                "SER before LDPC Decoder": ser_pre_ldpc_coded,
            }
        )

        st.dataframe(
            result_df.style.format(
                {
                    "BER LDPC Coded + LMMSE": "{:.3e}",
                    "BER Uncoded + LMMSE": "{:.3e}",
                    "BER Uncoded without Equalization": "{:.3e}",
                    "SER Uncoded + LMMSE": "{:.3e}",
                    "SER Uncoded without Equalization": "{:.3e}",
                    "SER before LDPC Decoder": "{:.3e}",
                }
            ),
            use_container_width=True
        )

    with tab4:
        st.markdown(
            f"""
            **Đề 5: Mô phỏng hệ thống 2x2 MIMO-OFDM sử dụng mã hóa LDPC trên kênh truyền Rayleigh nhiễu trắng, điều chế 64-QAM. Đánh giá chất lượng BER/SER.**

            Chương trình mô phỏng hệ thống **2x2 MIMO-OFDM** trên kênh truyền **Rayleigh block fading** có cộng **nhiễu trắng AWGN**.
            Dữ liệu nhị phân được mã hóa bằng **LDPC 5G** với code rate **{code_rate_str}**, sau đó được điều chế bằng **64-QAM**.
            Các symbol điều chế được ánh xạ lên **Resource Grid OFDM** và truyền qua kênh Rayleigh MIMO 2x2.

            Tại phía thu, hệ thống chính sử dụng bộ cân bằng **LMMSE Equalizer** với giả thiết biết hoàn hảo đáp ứng kênh, tức **perfect CSI**.
            Nhiễu AWGN được thêm vào sau kênh fading, và noise variance được tính bằng hàm **ebnodb2no** của Sionna để phù hợp với Eb/N0, số bit trên symbol, code rate và cấu hình OFDM Resource Grid.

            Hiệu năng hệ thống được đánh giá chủ yếu thông qua:
            - **BER LDPC coded + LMMSE**: BER của nhánh có mã hóa LDPC, đo sau LDPC decoder.
            - **BER uncoded + LMMSE reference**: BER của nhánh MIMO-OFDM không mã hóa LDPC nhưng vẫn có cân bằng LMMSE.
            - **BER uncoded without MIMO equalization**: BER của nhánh không mã hóa và không dùng bộ cân bằng MIMO.
            - **SER uncoded + LMMSE** và **SER uncoded without equalization**: dùng để minh họa vai trò của bộ cân bằng LMMSE.

            Trong hệ thống **spatial multiplexing MIMO**, tín hiệu từ nhiều anten phát bị trộn qua ma trận kênh Rayleigh.
            Nếu không có bộ cân bằng, bộ thu không thể tách chính xác các luồng dữ liệu, dẫn đến BER/SER cao.
            Bộ cân bằng **LMMSE** sử dụng thông tin kênh và phương sai nhiễu để ước lượng tín hiệu phát, từ đó cải thiện hiệu năng thu.

            Ngoài ra, chương trình vẫn tính **SER before LDPC Decoder** cho nhánh LDPC.
            Chỉ số này được đo tại tầng demapper trước khi giải mã LDPC, nên không dùng làm chỉ số chính để kết luận khả năng sửa lỗi của mã LDPC.
            Lợi ích chính của LDPC được thể hiện qua đường **BER LDPC coded + LMMSE** so với **BER uncoded + LMMSE reference**.

            Mô hình MIMO trong chương trình là **spatial multiplexing**, không phải Alamouti/STBC.
            """
        )

else:
    st.info("Nhấn **CHẠY MÔ PHỎNG** ở sidebar để bắt đầu Monte Carlo.")