import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core_simulation2 import AlamoutiSystem


st.set_page_config(
    page_title="Alamouti/STBC 2x2 MIMO-OFDM LDPC",
    layout="wide"
)

st.title("Alamouti/STBC 2x2 MIMO-OFDM với LDPC, 64-QAM, Rayleigh + AWGN")

st.markdown(
    """
    **Mô hình:** 2x2 Alamouti/STBC  
    **Điều chế:** 64-QAM  
    **Mã hóa:** LDPC 5G  
    **Kênh:** Rayleigh fading + AWGN  
    **Đánh giá:** BER/SER theo Eb/N0
    """
)

st.markdown("---")


def metric_value(m):
    return m["value"]


def metric_text(m):
    if m["upper_bound"]:
        return f"≤ {m['value']:.3e}  (0 lỗi / {m['total']:.0f})"
    return f"{m['value']:.3e}  ({m['errors']:.0f} lỗi / {m['total']:.0f})"


def metric_status(m):
    return "Upper bound" if m["upper_bound"] else "Measured"


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

    decoder_iter = st.select_slider(
        "LDPC Decoder Iterations",
        options=[10, 15, 20, 25, 30, 40, 50],
        value=25,
        key="decoder_iter_slider"
    )

    st.markdown("---")
    st.subheader("🎲 Monte Carlo")

    min_errs = st.slider(
        "Số lỗi tối thiểu",
        min_value=100,
        max_value=5000,
        value=500,
        step=100,
        key="min_errors_slider"
    )

    max_bits = st.select_slider(
        "Giới hạn bit tối đa",
        options=[1e5, 5e5, 1e6, 5e6, 1e7, 2e7],
        value=5e6,
        key="max_bits_slider"
    )

    batch_size = st.select_slider(
        "Batch size",
        options=[16, 32, 64, 128, 256],
        value=128,
        key="batch_size_slider"
    )

    run_btn = st.button(
        "🚀 CHẠY ALAMOUTI",
        use_container_width=True,
        key="run_alamouti_button"
    )


if run_btn:
    if ebno_max < ebno_min:
        st.error("Eb/N0 Max phải lớn hơn hoặc bằng Eb/N0 Min.")
        st.stop()

    if ebno_step <= 0:
        st.error("Bước Eb/N0 phải lớn hơn 0.")
        st.stop()

    ebno_range = np.arange(ebno_min, ebno_max + 1, ebno_step)

    st.subheader("📌 Ghi chú mô hình")

    st.warning(
        "Bản này là bản debug Alamouti. Nếu BER LDPC sau decoder không có lỗi, "
        "app sẽ ghi rõ là upper bound, không coi đó là BER đo chính xác."
    )

    st.info(
        "Alamouti/STBC dùng STBC combiner, không dùng LMMSEEqualizer kiểu Spatial Multiplexing. "
        "Đường without STBC combining là baseline yếu để thấy nếu không xử lý kênh thì lỗi rất cao."
    )

    system = AlamoutiSystem(
        code_rate=code_rate,
        decoder_iter=int(decoder_iter)
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    rows = []

    ber_ldpc_comb = []
    ber_uncoded_comb = []
    ber_ldpc_no_comb = []
    ber_uncoded_no_comb = []

    pre_ber_ldpc_comb = []
    pre_ber_ldpc_no_comb = []

    ser_ldpc_comb = []
    ser_uncoded_comb = []
    ser_ldpc_no_comb = []
    ser_uncoded_no_comb = []

    start_time = time.time()

    try:
        for i, ebno in enumerate(ebno_range):
            status_text.info(f"Đang mô phỏng Eb/N0 = {ebno} dB...")

            t0 = time.time()

            result = system.run_monte_carlo(
                ebno_db=float(ebno),
                min_errors=int(min_errs),
                max_bits=float(max_bits),
                batch_size=int(batch_size)
            )

            elapsed = time.time() - t0

            ber_ldpc_comb.append(metric_value(result["ber_ldpc_comb"]))
            ber_uncoded_comb.append(metric_value(result["ber_uncoded_comb"]))
            ber_ldpc_no_comb.append(metric_value(result["ber_ldpc_no_comb"]))
            ber_uncoded_no_comb.append(metric_value(result["ber_uncoded_no_comb"]))

            pre_ber_ldpc_comb.append(metric_value(result["pre_ber_ldpc_comb"]))
            pre_ber_ldpc_no_comb.append(metric_value(result["pre_ber_ldpc_no_comb"]))

            ser_ldpc_comb.append(metric_value(result["ser_ldpc_comb"]))
            ser_uncoded_comb.append(metric_value(result["ser_uncoded_comb"]))
            ser_ldpc_no_comb.append(metric_value(result["ser_ldpc_no_comb"]))
            ser_uncoded_no_comb.append(metric_value(result["ser_uncoded_no_comb"]))

            rows.append(
                {
                    "Eb/N0 (dB)": ebno,

                    "BER LDPC + STBC Combiner": result["ber_ldpc_comb"]["value"],
                    "BER LDPC + STBC Combiner Status": metric_status(result["ber_ldpc_comb"]),
                    "LDPC Combiner Bit Errors": result["ber_ldpc_comb"]["errors"],
                    "LDPC Combiner Total Bits": result["ber_ldpc_comb"]["total"],

                    "Pre-decoder BER LDPC + Combiner": result["pre_ber_ldpc_comb"]["value"],
                    "Pre-decoder LDPC Combiner Bit Errors": result["pre_ber_ldpc_comb"]["errors"],

                    "BER Uncoded + STBC Combiner": result["ber_uncoded_comb"]["value"],
                    "BER Uncoded + STBC Combiner Status": metric_status(result["ber_uncoded_comb"]),

                    "BER LDPC without STBC Combining": result["ber_ldpc_no_comb"]["value"],
                    "BER LDPC without STBC Combining Status": metric_status(result["ber_ldpc_no_comb"]),

                    "BER Uncoded without STBC Combining": result["ber_uncoded_no_comb"]["value"],
                    "BER Uncoded without STBC Combining Status": metric_status(result["ber_uncoded_no_comb"]),

                    "SER LDPC before Decoder + STBC Combiner": result["ser_ldpc_comb"]["value"],
                    "SER Uncoded + STBC Combiner": result["ser_uncoded_comb"]["value"],
                    "SER LDPC before Decoder without STBC Combining": result["ser_ldpc_no_comb"]["value"],
                    "SER Uncoded without STBC Combining": result["ser_uncoded_no_comb"]["value"],
                }
            )

            progress_bar.progress((i + 1) / len(ebno_range))

            st.write(
                f"Eb/N0 = **{ebno} dB** | "
                f"BER LDPC + Combiner = `{metric_text(result['ber_ldpc_comb'])}` | "
                f"Pre-BER LDPC + Combiner = `{metric_text(result['pre_ber_ldpc_comb'])}` | "
                f"BER Uncoded + Combiner = `{metric_text(result['ber_uncoded_comb'])}` | "
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
    # BER figure
    # ============================================================

    fig_ber = go.Figure()

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ber_ldpc_comb,
            mode="lines+markers",
            name=f"BER LDPC + STBC combiner, R={code_rate_str}",
            line=dict(width=4),
            marker=dict(size=9, symbol="diamond")
        )
    )

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=pre_ber_ldpc_comb,
            mode="lines+markers",
            name=f"Pre-decoder BER LDPC + combiner, R={code_rate_str}",
            line=dict(width=3, dash="dot"),
            marker=dict(size=8, symbol="circle")
        )
    )

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ber_uncoded_comb,
            mode="lines+markers",
            name="BER uncoded + STBC combiner",
            line=dict(width=3, dash="dash"),
            marker=dict(size=9, symbol="x")
        )
    )

    fig_ber.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ber_uncoded_no_comb,
            mode="lines+markers",
            name="BER uncoded without STBC combining",
            line=dict(width=3, dash="longdash"),
            marker=dict(size=9, symbol="square")
        )
    )

    fig_ber.update_layout(
        title="BER vs Eb/N0 - 2x2 Alamouti/STBC, Rayleigh + AWGN, 64-QAM",
        xaxis_title="Eb/N0 (dB)",
        yaxis_title="Bit Error Rate (BER)",
        yaxis_type="log",
        template="plotly_white",
        hovermode="x unified",
        height=650
    )

    # ============================================================
    # SER figure
    # ============================================================

    fig_ser = go.Figure()

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_ldpc_comb,
            mode="lines+markers",
            name=f"SER LDPC before decoder + STBC combiner, R={code_rate_str}",
            line=dict(width=4),
            marker=dict(size=9, symbol="diamond")
        )
    )

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_uncoded_comb,
            mode="lines+markers",
            name="SER uncoded + STBC combiner",
            line=dict(width=3, dash="dash"),
            marker=dict(size=9, symbol="x")
        )
    )

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_ldpc_no_comb,
            mode="lines+markers",
            name=f"SER LDPC before decoder without STBC combining, R={code_rate_str}",
            line=dict(width=3, dash="dot"),
            marker=dict(size=8, symbol="circle")
        )
    )

    fig_ser.add_trace(
        go.Scatter(
            x=ebno_range,
            y=ser_uncoded_no_comb,
            mode="lines+markers",
            name="SER uncoded without STBC combining",
            line=dict(width=3, dash="longdash"),
            marker=dict(size=9, symbol="square")
        )
    )

    fig_ser.update_layout(
        title="SER vs Eb/N0 - 2x2 Alamouti/STBC, Rayleigh + AWGN, 64-QAM",
        xaxis_title="Eb/N0 (dB)",
        yaxis_title="Symbol Error Rate (SER)",
        yaxis_type="log",
        template="plotly_white",
        hovermode="x unified",
        height=650
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📉 BER", "📊 SER", "📋 Bảng số liệu", "📝 Mô tả báo cáo"]
    )

    with tab1:
        st.plotly_chart(fig_ber, use_container_width=True)

        st.info(
            "Đường BER LDPC sau decoder nếu có trạng thái Upper bound nghĩa là mô phỏng chưa quan sát được lỗi nào ở mốc đó. "
            "Hãy xem thêm đường Pre-decoder BER để biết tín hiệu trước LDPC decoder còn lỗi nhiều hay ít."
        )

    with tab2:
        st.plotly_chart(fig_ser, use_container_width=True)

        st.info(
            "SER LDPC được đo trước LDPC decoder tại tầng demapper. "
            "Sau LDPC decoder, chỉ số chính nên dùng là BER."
        )

    with tab3:
        result_df = pd.DataFrame(rows)

        st.dataframe(
            result_df.style.format(
                {
                    "BER LDPC + STBC Combiner": "{:.3e}",
                    "Pre-decoder BER LDPC + Combiner": "{:.3e}",
                    "BER Uncoded + STBC Combiner": "{:.3e}",
                    "BER LDPC without STBC Combining": "{:.3e}",
                    "BER Uncoded without STBC Combining": "{:.3e}",
                    "SER LDPC before Decoder + STBC Combiner": "{:.3e}",
                    "SER Uncoded + STBC Combiner": "{:.3e}",
                    "SER LDPC before Decoder without STBC Combining": "{:.3e}",
                    "SER Uncoded without STBC Combining": "{:.3e}",
                    "LDPC Combiner Bit Errors": "{:.0f}",
                    "LDPC Combiner Total Bits": "{:.0f}",
                    "Pre-decoder LDPC Combiner Bit Errors": "{:.0f}",
                }
            ),
            use_container_width=True
        )

    with tab4:
        st.markdown(
            f"""
            **Mô phỏng hệ thống 2x2 MIMO-OFDM sử dụng Alamouti/STBC, LDPC, 64-QAM trên kênh Rayleigh + AWGN.**

            Hệ thống sử dụng **2 anten phát** và **2 anten thu** theo kỹ thuật **Alamouti/STBC**.
            Khác với Spatial Multiplexing, Alamouti không truyền hai stream độc lập cùng lúc mà mã hóa một luồng dữ liệu qua hai anten phát trong hai khe thời gian.
            Mục tiêu chính của Alamouti/STBC là khai thác **phân tập không gian** để giảm ảnh hưởng của fading.

            Tại phía thu, bản đúng sử dụng **Alamouti/STBC combiner** để kết hợp tín hiệu thu từ hai anten và hai khe thời gian.
            App cũng mô phỏng thêm trường hợp **without STBC combining** để minh họa rằng nếu không xử lý kênh đúng, BER/SER sẽ cao.

            Trong bản này, tín hiệu phát trên hai anten được chuẩn hóa công suất bởi hệ số **1/sqrt(2)** để tránh lợi thế giả do tổng công suất phát tăng gấp đôi.

            Các chỉ số chính:
            - **BER LDPC + STBC combiner**: BER sau LDPC decoder.
            - **Pre-decoder BER LDPC + combiner**: BER của coded bits trước LDPC decoder, dùng để debug chất lượng demapper/LLR.
            - **BER uncoded + STBC combiner**: BER của nhánh không mã hóa.
            - **SER LDPC before decoder**: SER đo ở tầng demapper trước LDPC decoder.
            - **Upper bound**: nếu không quan sát được lỗi nào, app hiển thị giá trị dạng 1/N như một giới hạn trên, không phải BER đo chính xác.
            """
        )

else:
    st.info("Nhấn **CHẠY ALAMOUTI** để bắt đầu mô phỏng.")