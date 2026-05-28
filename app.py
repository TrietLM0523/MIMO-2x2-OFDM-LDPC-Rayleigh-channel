import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core_simulation import MIMOSystem


def ber_theory_rayleigh_qam(ebno_db, M=64):
    """
    BER tham chiếu gần đúng cho M-QAM trên Rayleigh SISO.
    Chỉ dùng để tham khảo, không phải lý thuyết chính xác cho MIMO 2x2 LMMSE.
    """
    k = np.log2(M)
    ebno_linear = 10 ** (ebno_db / 10)
    gamma_s = ebno_linear * k

    term = np.sqrt((1.5 * gamma_s) / (M - 1 + 1.5 * gamma_s))
    ber = (2 * (1 - 1 / np.sqrt(M)) / k) * 0.5 * (1 - term)

    return ber


st.set_page_config(
    page_title="2x2 MIMO-OFDM LDPC Simulator",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔬 Mô phỏng 2x2 MIMO-OFDM LDPC trên kênh Rayleigh + AWGN")

st.markdown(
    """
    Hệ thống mô phỏng:

    - **MIMO 2x2 spatial multiplexing**
    - **OFDM**
    - **Kênh Rayleigh block fading**
    - **Nhiễu trắng AWGN**
    - **LDPC 5G**
    - **Điều chế 64-QAM**
    - **Bộ cân bằng LMMSE**
    - Đánh giá **BER** và **SER**
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
        value=20,
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
        max_value=2000,
        value=500,
        step=100,
        key="min_errors_slider"
    )

    max_bits = st.select_slider(
        "Giới hạn bit tối đa",
        options=[1e5, 5e5, 1e6, 5e6, 1e7],
        value=5e6,
        key="max_bits_slider"
    )

    batch_size = st.select_slider(
        "Batch size",
        options=[16, 32, 64, 128, 256],
        value=64,
        key="batch_size_slider"
    )

    show_theory = st.checkbox(
        "Hiển thị đường tham chiếu Rayleigh SISO",
        value=True,
        key="show_theory_checkbox"
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
    info_col4.metric("Equalizer", "LMMSE")

    with st.expander("🧠 Ghi chú mô hình"):
        st.markdown(
            """
            - Hệ thống dùng **2 anten phát** và **2 anten thu**.
            - Mô hình MIMO hiện tại là **spatial multiplexing**, không phải Alamouti/STBC.
            - Bộ thu dùng **LMMSE equalizer**.
            - Bộ thu giả sử biết hoàn hảo đáp ứng kênh, tức **perfect CSI**.
            - Đường lý thuyết Rayleigh nếu bật chỉ là đường **tham chiếu SISO uncoded 64-QAM**.
            """
        )

    progress_bar = st.progress(0)
    status_text = st.empty()

    ber_coded = []
    ser_coded = []
    ber_uncoded = []
    ser_uncoded = []
    theory_bers = []

    start_time = time.time()

    try:
        system = MIMOSystem(code_rate=code_rate)

        for i, ebno in enumerate(ebno_range):
            status_text.info(f"Đang mô phỏng Eb/N0 = {ebno} dB...")

            t0 = time.time()

            ber_c, ser_c, ber_u, ser_u = system.run_monte_carlo(
                ebno_db=float(ebno),
                min_errors=int(min_errs),
                max_bits=float(max_bits),
                batch_size=int(batch_size)
            )

            elapsed = time.time() - t0

            ber_coded.append(ber_c)
            ser_coded.append(ser_c)
            ber_uncoded.append(ber_u)
            ser_uncoded.append(ser_u)

            theory_bers.append(
                ber_theory_rayleigh_qam(float(ebno), M=64)
            )

            progress_bar.progress((i + 1) / len(ebno_range))

            st.write(
                f"Eb/N0 = **{ebno} dB** | "
                f"BER coded = `{ber_c:.3e}` | "
                f"SER coded = `{ser_c:.3e}` | "
                f"BER uncoded = `{ber_u:.3e}` | "
                f"SER uncoded = `{ser_u:.3e}` | "
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

    m1.metric("BER coded thấp nhất", f"{min(ber_coded):.2e}")
    m2.metric("SER coded thấp nhất", f"{min(ser_coded):.2e}")
    m3.metric("BER uncoded thấp nhất", f"{min(ber_uncoded):.2e}")
    m4.metric("SER uncoded thấp nhất", f"{min(ser_uncoded):.2e}")

    fig_ber = go.Figure()

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ber_coded,
            mode="lines+markers",
            name=f"BER LDPC coded, R={code_rate_str}",
            line=dict(color="#00ffcc", width=4),
            marker=dict(size=9, symbol="diamond")
        )
    )

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ber_uncoded,
            mode="lines+markers",
            name="BER uncoded simulation",
            line=dict(color="#ffcc00", width=3),
            marker=dict(size=9, symbol="x")
        )
    )

    if show_theory:
        fig_ber.add_trace(
            go.Scatter(
                x=ebno_range,
                y=theory_bers,
                mode="lines",
                name="Tham chiếu Rayleigh SISO 64-QAM",
                line=dict(color="#ff4b4b", width=2, dash="dash")
            )
        )

    fig_ber.update_layout(
        title="BER vs Eb/N0",
        xaxis_title="Eb/N0 (dB)",
        yaxis_title="Bit Error Rate (BER)",
        yaxis_type="log",
        template="plotly_dark",
        yaxis=dict(
            exponentformat="e",
            gridcolor="#333"
        ),
        xaxis=dict(
            gridcolor="#333"
        ),
        hovermode="x unified",
        height=600
    )

    fig_ser = go.Figure()

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_coded,
            mode="lines+markers",
            name=f"SER coded, R={code_rate_str}",
            line=dict(color="#00ffcc", width=4),
            marker=dict(size=9, symbol="diamond")
        )
    )

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_uncoded,
            mode="lines+markers",
            name="SER uncoded simulation",
            line=dict(color="#ffcc00", width=3),
            marker=dict(size=9, symbol="x")
        )
    )

    fig_ser.update_layout(
        title="SER vs Eb/N0",
        xaxis_title="Eb/N0 (dB)",
        yaxis_title="Symbol Error Rate (SER)",
        yaxis_type="log",
        template="plotly_dark",
        yaxis=dict(
            exponentformat="e",
            gridcolor="#333"
        ),
        xaxis=dict(
            gridcolor="#333"
        ),
        hovermode="x unified",
        height=600
    )

    tab1, tab2, tab3 = st.tabs(
        ["📉 BER", "📊 SER", "📋 Bảng số liệu"]
    )

    with tab1:
        st.plotly_chart(fig_ber, use_container_width=True)

    with tab2:
        st.plotly_chart(fig_ser, use_container_width=True)

    with tab3:
        result_df = pd.DataFrame(
            {
                "Eb/N0 (dB)": ebno_range,
                "BER LDPC Coded": ber_coded,
                "SER LDPC Coded": ser_coded,
                "BER Uncoded": ber_uncoded,
                "SER Uncoded": ser_uncoded,
                "BER Theory Rayleigh SISO": theory_bers
            }
        )

        st.dataframe(
            result_df.style.format(
                {
                    "BER LDPC Coded": "{:.3e}",
                    "SER LDPC Coded": "{:.3e}",
                    "BER Uncoded": "{:.3e}",
                    "SER Uncoded": "{:.3e}",
                    "BER Theory Rayleigh SISO": "{:.3e}"
                }
            ),
            use_container_width=True
        )


else:
    st.info("Nhấn **CHẠY MÔ PHỎNG** ở sidebar để bắt đầu Monte Carlo.")