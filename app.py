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

    with st.expander("🧠 Ghi chú mô hình và đường tham chiếu"):
        st.markdown(
            """
            - Hệ thống sử dụng **2 anten phát** và **2 anten thu**.
            - Mô hình MIMO hiện tại là **spatial multiplexing**, không phải Alamouti/STBC.
            - Bộ thu sử dụng **LMMSE Equalizer**.
            - Bộ thu giả sử biết hoàn hảo đáp ứng kênh, tức **perfect CSI**.
            - Đường tham chiếu chính trong đồ thị là **MIMO-OFDM không mã hóa LDPC**.
            - Vì vậy, so sánh chính là:
                - **LDPC coded MIMO-OFDM**
                - **Uncoded MIMO-OFDM reference**
            """
        )

    progress_bar = st.progress(0)
    status_text = st.empty()

    ber_coded = []
    ser_coded = []
    ber_uncoded = []
    ser_uncoded = []

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

            progress_bar.progress((i + 1) / len(ebno_range))

            st.write(
                f"Eb/N0 = **{ebno} dB** | "
                f"BER coded = `{ber_c:.3e}` | "
                f"SER coded = `{ser_c:.3e}` | "
                f"BER uncoded reference = `{ber_u:.3e}` | "
                f"SER uncoded reference = `{ser_u:.3e}` | "
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
    m1.metric("BER LDPC thấp nhất", f"{min(ber_coded):.2e}")
    m2.metric("SER LDPC thấp nhất", f"{min(ser_coded):.2e}")
    m3.metric("BER reference thấp nhất", f"{min(ber_uncoded):.2e}")
    m4.metric("SER reference thấp nhất", f"{min(ser_uncoded):.2e}")

    fig_ber = go.Figure()

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ber_coded,
            mode="lines+markers",
            name=f"BER LDPC coded, R={code_rate_str}",
            line=dict(width=4),
            marker=dict(size=9, symbol="diamond")
        )
    )

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ber_uncoded,
            mode="lines+markers",
            name="BER MIMO-OFDM uncoded reference",
            line=dict(width=3, dash="dash"),
            marker=dict(size=9, symbol="x")
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

    fig_ser = go.Figure()

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_coded,
            mode="lines+markers",
            name=f"SER LDPC coded, R={code_rate_str}",
            line=dict(width=4),
            marker=dict(size=9, symbol="diamond")
        )
    )

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_uncoded,
            mode="lines+markers",
            name="SER MIMO-OFDM uncoded reference",
            line=dict(width=3, dash="dash"),
            marker=dict(size=9, symbol="x")
        )
    )

    fig_ser.update_layout(
        title="SER vs Eb/N0 - 2x2 MIMO-OFDM Rayleigh + AWGN, 64-QAM",
        xaxis_title="Eb/N0 (dB)",
        yaxis_title="Symbol Error Rate (SER)",
        yaxis_type="log",
        template="plotly_white",
        hovermode="x unified",
        height=600
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📉 BER", "📊 SER", "📋 Bảng số liệu", "📝 Mô tả báo cáo"]
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
                "BER MIMO-OFDM Uncoded Reference": ber_uncoded,
                "SER MIMO-OFDM Uncoded Reference": ser_uncoded,
            }
        )

        st.dataframe(
            result_df.style.format(
                {
                    "BER LDPC Coded": "{:.3e}",
                    "SER LDPC Coded": "{:.3e}",
                    "BER MIMO-OFDM Uncoded Reference": "{:.3e}",
                    "SER MIMO-OFDM Uncoded Reference": "{:.3e}",
                }
            ),
            use_container_width=True
        )

    with tab4:
        st.markdown(
            f"""
            **Đề 5: Mô phỏng hệ thống 2x2 MIMO-OFDM sử dụng mã hóa LDPC trên kênh truyền Rayleigh nhiễu trắng, điều chế 64-QAM. Đánh giá chất lượng BER/SER.**

            Chương trình mô phỏng hệ thống **2x2 MIMO-OFDM** trên kênh truyền **Rayleigh fading** có cộng **nhiễu trắng AWGN**.
            Dữ liệu nhị phân được mã hóa bằng **LDPC 5G** với code rate **{code_rate_str}**, sau đó được điều chế bằng **64-QAM**.
            Các symbol điều chế được ánh xạ lên resource grid OFDM và truyền qua kênh Rayleigh 2x2.
            Tại phía thu, hệ thống sử dụng bộ cân bằng **LMMSE** với giả thiết biết hoàn hảo đáp ứng kênh.
            Hiệu năng hệ thống được đánh giá thông qua **BER** và **SER** theo các mức **Eb/N0** khác nhau.

            Đường tham chiếu trong mô phỏng là nhánh **MIMO-OFDM không mã hóa LDPC**, sử dụng cùng cấu hình anten, kênh truyền, điều chế và bộ cân bằng.
            """
        )

else:
    st.info("Nhấn **CHẠY MÔ PHỎNG** ở sidebar để bắt đầu Monte Carlo.")
