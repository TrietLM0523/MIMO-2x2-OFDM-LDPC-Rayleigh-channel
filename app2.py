import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core_simulation import MIMOSystem
from core_simulation2 import AlamoutiSystem


st.set_page_config(
    page_title="So sánh Spatial Multiplexing và Alamouti",
    layout="wide"
)

st.title("So sánh 2x2 MIMO-OFDM: Spatial Multiplexing vs Alamouti/STBC")

st.markdown(
    """
    **Mục tiêu:** so sánh hiệu năng BER/SER giữa hai hướng triển khai MIMO 2x2:

    - **Spatial Multiplexing + LMMSE:** truyền 2 stream song song, cần equalizer để tách stream.
    - **Alamouti/STBC:** truyền theo phân tập không gian, ưu tiên độ tin cậy hơn throughput.

    Cả hai đều dùng **LDPC**, **64-QAM**, **Rayleigh fading + AWGN**.
    """
)

st.markdown("---")

with st.sidebar:
    st.header("⚙️ Cấu hình")

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
        options=[1e5, 5e5, 1e6, 5e6, 1e7, 2e7],
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
        "🚀 CHẠY SO SÁNH",
        use_container_width=True,
        key="run_compare_button"
    )


if run_btn:
    if ebno_max < ebno_min:
        st.error("Eb/N0 Max phải lớn hơn hoặc bằng Eb/N0 Min.")
        st.stop()

    ebno_range = np.arange(
        ebno_min,
        ebno_max + 1,
        ebno_step
    )

    st.subheader("📌 Ghi chú quan trọng")

    st.info(
        "Spatial Multiplexing và Alamouti/STBC không hoàn toàn cùng mục tiêu. "
        "Spatial Multiplexing truyền nhiều stream để tăng throughput, còn Alamouti/STBC dùng phân tập không gian để tăng độ tin cậy. "
        "Vì vậy so sánh này dùng để quan sát xu hướng BER/SER, không nên kết luận đơn giản rằng kỹ thuật nào luôn tốt hơn trong mọi tiêu chí."
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    ber_spatial_ldpc = []
    ser_spatial_pre_ldpc = []
    ber_spatial_uncoded = []
    ser_spatial_uncoded = []

    ber_alamouti_ldpc = []
    ser_alamouti_pre_ldpc = []
    block6_alamouti_after_decoder = []
    ber_alamouti_uncoded = []
    ser_alamouti_uncoded = []

    start_time = time.time()

    try:
        spatial_system = MIMOSystem(code_rate=code_rate)
        alamouti_system = AlamoutiSystem(code_rate=code_rate)

        for i, ebno in enumerate(ebno_range):
            status_text.info(f"Đang mô phỏng Eb/N0 = {ebno} dB...")

            t0 = time.time()

            spatial_result = spatial_system.run_monte_carlo(
                ebno_db=float(ebno),
                min_errors=int(min_errs),
                max_bits=float(max_bits),
                batch_size=int(batch_size)
            )

            # Compatible with latest core_simulation.py:
            # return: ber_c, ser_pre_ldpc_c, ber_u, ser_u, ber_no_eq, ser_no_eq
            ber_s_c = spatial_result[0]
            ser_s_pre = spatial_result[1]
            ber_s_u = spatial_result[2]
            ser_s_u = spatial_result[3]

            (
                ber_a_c,
                ser_a_pre,
                block6_a_after,
                ber_a_u,
                ser_a_u
            ) = alamouti_system.run_monte_carlo(
                ebno_db=float(ebno),
                min_errors=int(min_errs),
                max_bits=float(max_bits),
                batch_size=int(batch_size)
            )

            elapsed = time.time() - t0

            ber_spatial_ldpc.append(ber_s_c)
            ser_spatial_pre_ldpc.append(ser_s_pre)
            ber_spatial_uncoded.append(ber_s_u)
            ser_spatial_uncoded.append(ser_s_u)

            ber_alamouti_ldpc.append(ber_a_c)
            ser_alamouti_pre_ldpc.append(ser_a_pre)
            block6_alamouti_after_decoder.append(block6_a_after)
            ber_alamouti_uncoded.append(ber_a_u)
            ser_alamouti_uncoded.append(ser_a_u)

            progress_bar.progress((i + 1) / len(ebno_range))

            st.write(
                f"Eb/N0 = **{ebno} dB** | "
                f"BER Spatial LDPC = `{ber_s_c:.3e}` | "
                f"BER Alamouti LDPC = `{ber_a_c:.3e}` | "
                f"BER Spatial uncoded = `{ber_s_u:.3e}` | "
                f"BER Alamouti uncoded = `{ber_a_u:.3e}` | "
                f"time = `{elapsed:.2f}s`"
            )

        total_time = time.time() - start_time
        status_text.success(f"✅ Hoàn tất trong {total_time:.2f} giây.")

    except Exception as e:
        st.error("Có lỗi khi chạy mô phỏng.")
        st.exception(e)
        st.stop()

    st.markdown("---")
    st.subheader("📈 Kết quả")

    # ============================================================
    # BER comparison
    # ============================================================

    fig_ber = go.Figure()

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ber_spatial_ldpc,
            mode="lines+markers",
            name=f"BER Spatial Multiplexing + LDPC + LMMSE, R={code_rate_str}",
            line=dict(width=4),
            marker=dict(size=9, symbol="diamond")
        )
    )

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ber_alamouti_ldpc,
            mode="lines+markers",
            name=f"BER Alamouti/STBC + LDPC, R={code_rate_str}",
            line=dict(width=4, dash="dash"),
            marker=dict(size=9, symbol="circle")
        )
    )

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ber_spatial_uncoded,
            mode="lines+markers",
            name="BER Spatial uncoded + LMMSE",
            line=dict(width=3, dash="dot"),
            marker=dict(size=8, symbol="x")
        )
    )

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ber_alamouti_uncoded,
            mode="lines+markers",
            name="BER Alamouti uncoded",
            line=dict(width=3, dash="longdash"),
            marker=dict(size=8, symbol="square")
        )
    )

    fig_ber.update_layout(
        title="BER vs Eb/N0 - Spatial Multiplexing vs Alamouti/STBC",
        xaxis_title="Eb/N0 (dB)",
        yaxis_title="Bit Error Rate (BER)",
        yaxis_type="log",
        template="plotly_white",
        hovermode="x unified",
        height=650
    )

    # ============================================================
    # SER comparison
    # ============================================================

    fig_ser = go.Figure()

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_spatial_pre_ldpc,
            mode="lines+markers",
            name="SER Spatial before LDPC decoder",
            line=dict(width=3),
            marker=dict(size=8, symbol="diamond")
        )
    )

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_alamouti_pre_ldpc,
            mode="lines+markers",
            name="SER Alamouti before LDPC decoder",
            line=dict(width=3, dash="dash"),
            marker=dict(size=8, symbol="circle")
        )
    )

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_spatial_uncoded,
            mode="lines+markers",
            name="SER Spatial uncoded + LMMSE",
            line=dict(width=3, dash="dot"),
            marker=dict(size=8, symbol="x")
        )
    )

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_alamouti_uncoded,
            mode="lines+markers",
            name="SER Alamouti uncoded",
            line=dict(width=3, dash="longdash"),
            marker=dict(size=8, symbol="square")
        )
    )

    fig_ser.update_layout(
        title="SER vs Eb/N0 - Spatial Multiplexing vs Alamouti/STBC",
        xaxis_title="Eb/N0 (dB)",
        yaxis_title="Symbol Error Rate (SER)",
        yaxis_type="log",
        template="plotly_white",
        hovermode="x unified",
        height=650
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📉 BER", "📊 SER", "📋 Bảng số liệu", "📝 Ghi chú báo cáo"]
    )

    with tab1:
        st.plotly_chart(fig_ber, use_container_width=True)

    with tab2:
        st.plotly_chart(fig_ser, use_container_width=True)

    with tab3:
        result_df = pd.DataFrame(
            {
                "Eb/N0 (dB)": ebno_range,
                "BER Spatial LDPC + LMMSE": ber_spatial_ldpc,
                "BER Alamouti LDPC": ber_alamouti_ldpc,
                "BER Spatial Uncoded + LMMSE": ber_spatial_uncoded,
                "BER Alamouti Uncoded": ber_alamouti_uncoded,
                "SER Spatial before LDPC": ser_spatial_pre_ldpc,
                "SER Alamouti before LDPC": ser_alamouti_pre_ldpc,
                "SER Spatial Uncoded + LMMSE": ser_spatial_uncoded,
                "SER Alamouti Uncoded": ser_alamouti_uncoded,
                "6-bit Block Error Alamouti after Decoder": block6_alamouti_after_decoder,
            }
        )

        st.dataframe(
            result_df.style.format(
                {
                    "BER Spatial LDPC + LMMSE": "{:.3e}",
                    "BER Alamouti LDPC": "{:.3e}",
                    "BER Spatial Uncoded + LMMSE": "{:.3e}",
                    "BER Alamouti Uncoded": "{:.3e}",
                    "SER Spatial before LDPC": "{:.3e}",
                    "SER Alamouti before LDPC": "{:.3e}",
                    "SER Spatial Uncoded + LMMSE": "{:.3e}",
                    "SER Alamouti Uncoded": "{:.3e}",
                    "6-bit Block Error Alamouti after Decoder": "{:.3e}",
                }
            ),
            use_container_width=True
        )

    with tab4:
        st.markdown(
            f"""
            ## Nhận xét mô hình

            Trong mô phỏng này, hai kỹ thuật MIMO được so sánh:

            **1. Spatial Multiplexing + LMMSE**

            Kỹ thuật này truyền nhiều luồng dữ liệu song song qua các anten phát.
            Với cấu hình 2x2 MIMO, hệ thống có thể truyền hai spatial streams cùng lúc.
            Do các stream bị trộn qua kênh Rayleigh, bộ thu sử dụng **LMMSE Equalizer**
            để tách và ước lượng tín hiệu phát.

            **2. Alamouti/STBC**

            Kỹ thuật này sử dụng mã khối không gian-thời gian để tạo phân tập không gian.
            Thay vì truyền hai stream độc lập, Alamouti truyền một luồng dữ liệu được mã hóa
            qua hai anten phát trong hai khe thời gian. Mục tiêu chính là tăng độ tin cậy
            và giảm ảnh hưởng của fading.

            ## Lưu ý khi so sánh

            Spatial Multiplexing và Alamouti/STBC không hoàn toàn cùng mục tiêu:

            - **Spatial Multiplexing** ưu tiên throughput vì truyền nhiều stream song song.
            - **Alamouti/STBC** ưu tiên độ tin cậy nhờ diversity gain.

            Vì vậy, nếu đường BER của Alamouti thấp hơn hoặc mượt hơn, điều đó không có nghĩa là
            Spatial Multiplexing sai. Nó chỉ phản ánh rằng Alamouti có lợi thế phân tập trong kênh
            Rayleigh fading, còn Spatial Multiplexing cần equalizer tốt để tách các stream.

            ## Chỉ số đánh giá

            - **BER LDPC** được đo sau LDPC decoder.
            - **SER before LDPC decoder** được đo ở tầng demapper trước khi mã LDPC sửa lỗi.
            - **6-bit Block Error after Decoder** không phải SER điều chế, mà chỉ là tỷ lệ lỗi nhóm 6 bit
              thông tin sau giải mã LDPC.
            """
        )

else:
    st.info("Nhấn **CHẠY SO SÁNH** để bắt đầu mô phỏng.")