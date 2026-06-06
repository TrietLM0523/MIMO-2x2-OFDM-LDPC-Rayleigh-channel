import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core_simulation import AlamoutiSystem


st.set_page_config(
    page_title="2x2 MIMO-OFDM LDPC",
    layout="wide"
)

st.title("Đề 5: Mô phỏng hệ thống 2x2 MIMO-OFDM sử dụng mã hóa LDPC")

st.markdown(
    """
    **Kênh truyền:** Rayleigh fading + AWGN  
    **Điều chế:** 64-QAM  
    **Kỹ thuật MIMO:** Alamouti/STBC  
    **Đánh giá:** BER/SER theo Eb/N0
    """
)

st.markdown("---")


def get_value(result, mode, metric):
    return result[mode][metric]["value"]


def get_status(result, mode, metric):
    return "Upper bound" if result[mode][metric]["upper_bound"] else "Measured"


def get_errors(result, mode, metric):
    return result[mode][metric]["errors"]


def get_total(result, mode, metric):
    return result[mode][metric]["total"]


def format_metric(result, mode, metric):
    item = result[mode][metric]
    value = item["value"]
    errors = item["errors"]
    total = item["total"]

    if item["upper_bound"]:
        return f"≤ {value:.3e} | 0/{total:.0f}"

    return f"{value:.3e} | {errors:.0f}/{total:.0f}"


with st.sidebar:
    st.header("Cấu hình mô phỏng")

    ebno_min = st.number_input(
        "Eb/N0 Min (dB)",
        value=0.0,
        step=0.5,
        format="%.1f",
        key="ebno_min_input"
    )

    ebno_max = st.number_input(
        "Eb/N0 Max (dB)",
        value=10.0,
        step=0.5,
        format="%.1f",
        key="ebno_max_input"
    )

    ebno_step = st.number_input(
        "Eb/N0 Bước (dB)",
        value=0.5,
        min_value=0.1,
        step=0.1,
        format="%.1f",
        key="ebno_step_input"
    )

    code_rate_str = st.selectbox(
        "LDPC Code Rate",
        ["1/2", "2/3", "3/4"],
        index=1,
        key="code_rate_select"
    )

    rate_map = {
        "1/2": 1 / 2,
        "2/3": 2 / 3,
        "3/4": 3 / 4,
    }

    code_rate = rate_map[code_rate_str]

    decoder_iter = st.select_slider(
        "LDPC Decoder Iterations",
        options=[5, 10, 15, 20, 25, 30, 40, 50],
        value=15,
        key="decoder_iter_slider"
    )

    num_data_symbols = st.select_slider(
        "Packet/Data Symbols",
        options=[128, 256, 512, 896],
        value=512,
        key="num_data_symbols_slider"
    )

    equalizer_choices = st.multiselect(
        "Equalizer",
        ["ZF", "MMSE"],
        default=["ZF", "MMSE"],
        key="equalizer_multiselect"
    )

    show_no_equalizer = st.checkbox(
        "No Equalizer",
        value=False,
        key="show_no_equalizer_checkbox"
    )

    st.markdown("---")

    min_errors = st.slider(
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
        "Chạy mô phỏng",
        use_container_width=True,
        key="run_simulation_button"
    )


if run_btn:
    if ebno_max < ebno_min:
        st.error("Eb/N0 Max phải lớn hơn hoặc bằng Eb/N0 Min.")
        st.stop()

    if len(equalizer_choices) == 0 and not show_no_equalizer:
        st.error("Cần chọn ít nhất một Equalizer.")
        st.stop()

    modes = []

    if "ZF" in equalizer_choices:
        modes.append("zf")

    if "MMSE" in equalizer_choices:
        modes.append("mmse")

    if show_no_equalizer:
        modes.append("none")

    mode_label = {
        "zf": "ZF",
        "mmse": "MMSE",
        "none": "No Equalizer",
    }

    ebno_range = np.arange(
        float(ebno_min),
        float(ebno_max) + float(ebno_step) / 2,
        float(ebno_step)
    )

    st.subheader("Thông số hệ thống")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MIMO", "2x2 Alamouti")
    c2.metric("Modulation", "64-QAM")
    c3.metric("LDPC Rate", code_rate_str)
    c4.metric("Packet Symbols", str(num_data_symbols))

    c5, c6, c7, c8 = st.columns(4)

    c5.metric("Decoder Iterations", str(decoder_iter))
    c6.metric("Eb/N0 Points", str(len(ebno_range)))
    c7.metric("Batch Size", str(batch_size))
    c8.metric("Max Bits", f"{float(max_bits):.0e}")

    progress_bar = st.progress(0)
    status_text = st.empty()

    system = AlamoutiSystem(
        code_rate=code_rate,
        decoder_iter=int(decoder_iter),
        num_data_symbols=int(num_data_symbols)
    )

    data = {
        mode: {
            "coded_ber": [],
            "uncoded_ber": [],
            "coded_ser": [],
            "uncoded_ser": [],
        }
        for mode in modes
    }

    rows = []

    start_time = time.time()

    try:
        for i, ebno in enumerate(ebno_range):
            status_text.text(f"Đang tính Eb/N0 = {ebno:.1f} dB")

            t0 = time.time()

            result = system.run_monte_carlo(
                ebno_db=float(ebno),
                min_errors=int(min_errors),
                max_bits=float(max_bits),
                batch_size=int(batch_size)
            )

            elapsed = time.time() - t0

            row = {
                "Eb/N0 (dB)": ebno,
                "Time (s)": elapsed,
            }

            for mode in modes:
                data[mode]["coded_ber"].append(
                    get_value(result, mode, "coded_ber")
                )
                data[mode]["uncoded_ber"].append(
                    get_value(result, mode, "uncoded_ber")
                )
                data[mode]["coded_ser"].append(
                    get_value(result, mode, "coded_ser")
                )
                data[mode]["uncoded_ser"].append(
                    get_value(result, mode, "uncoded_ser")
                )

                prefix = mode_label[mode]

                row[f"BER LDPC + {prefix}"] = get_value(
                    result,
                    mode,
                    "coded_ber"
                )
                row[f"BER LDPC + {prefix} Status"] = get_status(
                    result,
                    mode,
                    "coded_ber"
                )
                row[f"BER LDPC + {prefix} Errors"] = get_errors(
                    result,
                    mode,
                    "coded_ber"
                )
                row[f"BER LDPC + {prefix} Total"] = get_total(
                    result,
                    mode,
                    "coded_ber"
                )

                row[f"BER Uncoded + {prefix}"] = get_value(
                    result,
                    mode,
                    "uncoded_ber"
                )
                row[f"BER Uncoded + {prefix} Status"] = get_status(
                    result,
                    mode,
                    "uncoded_ber"
                )
                row[f"BER Uncoded + {prefix} Errors"] = get_errors(
                    result,
                    mode,
                    "uncoded_ber"
                )
                row[f"BER Uncoded + {prefix} Total"] = get_total(
                    result,
                    mode,
                    "uncoded_ber"
                )

                row[f"SER LDPC + {prefix}"] = get_value(
                    result,
                    mode,
                    "coded_ser"
                )
                row[f"SER LDPC + {prefix} Status"] = get_status(
                    result,
                    mode,
                    "coded_ser"
                )
                row[f"SER Uncoded + {prefix}"] = get_value(
                    result,
                    mode,
                    "uncoded_ser"
                )
                row[f"SER Uncoded + {prefix} Status"] = get_status(
                    result,
                    mode,
                    "uncoded_ser"
                )

            rows.append(row)

            progress_bar.progress((i + 1) / len(ebno_range))

            line_parts = [
                f"Eb/N0 = {ebno:.1f} dB",
                f"time = {elapsed:.2f}s",
            ]

            for mode in modes:
                prefix = mode_label[mode]
                line_parts.append(
                    f"BER LDPC {prefix}: {format_metric(result, mode, 'coded_ber')}"
                )
                line_parts.append(
                    f"BER Uncoded {prefix}: {format_metric(result, mode, 'uncoded_ber')}"
                )

            st.write(" | ".join(line_parts))

        total_time = time.time() - start_time
        status_text.text(f"Hoàn tất trong {total_time:.2f} giây")

    except Exception as e:
        st.error("Có lỗi khi chạy mô phỏng.")
        st.exception(e)
        st.stop()

    st.markdown("---")

    fig_ber = go.Figure()

    for mode in modes:
        prefix = mode_label[mode]

        fig_ber.add_trace(
            go.Scatter(
                x=ebno_range,
                y=data[mode]["coded_ber"],
                mode="lines+markers",
                name=f"BER LDPC + {prefix}",
                line=dict(width=4),
                marker=dict(size=8)
            )
        )

        fig_ber.add_trace(
            go.Scatter(
                x=ebno_range,
                y=data[mode]["uncoded_ber"],
                mode="lines+markers",
                name=f"BER Uncoded + {prefix}",
                line=dict(width=3, dash="dash"),
                marker=dict(size=8)
            )
        )

    fig_ber.update_layout(
        title="BER vs Eb/N0",
        xaxis_title="Eb/N0 (dB)",
        yaxis_title="Bit Error Rate (BER)",
        yaxis_type="log",
        template="plotly_white",
        hovermode="x unified",
        height=650
    )

    fig_ser = go.Figure()

    for mode in modes:
        prefix = mode_label[mode]

        fig_ser.add_trace(
            go.Scatter(
                x=ebno_range,
                y=data[mode]["coded_ser"],
                mode="lines+markers",
                name=f"SER LDPC + {prefix}",
                line=dict(width=4),
                marker=dict(size=8)
            )
        )

        fig_ser.add_trace(
            go.Scatter(
                x=ebno_range,
                y=data[mode]["uncoded_ser"],
                mode="lines+markers",
                name=f"SER Uncoded + {prefix}",
                line=dict(width=3, dash="dash"),
                marker=dict(size=8)
            )
        )

    fig_ser.update_layout(
        title="SER vs Eb/N0",
        xaxis_title="Eb/N0 (dB)",
        yaxis_title="Symbol Error Rate (SER)",
        yaxis_type="log",
        template="plotly_white",
        hovermode="x unified",
        height=650
    )

    result_df = pd.DataFrame(rows)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["BER", "SER", "Bảng số liệu", "Tính toán hệ thống"]
    )

    with tab1:
        st.plotly_chart(fig_ber, use_container_width=True)

    with tab2:
        st.plotly_chart(fig_ser, use_container_width=True)

    with tab3:
        result_df = pd.DataFrame(rows)

        # Tách cột số và cột chữ để tránh lỗi:
        # ValueError: Unknown format code 'e' for object of type 'str'
        format_dict = {}

        for col in result_df.columns:
            # Cột text/status thì tuyệt đối không format số
            if (
                col.endswith("Status")
                or col in ["Equalizer", "Mode", "Metric"]
                or result_df[col].dtype == "object"
            ):
                continue

            # Cột Eb/N0
            if col == "Eb/N0 (dB)":
                format_dict[col] = "{:.1f}"

            # Cột thời gian
            elif col == "Time (s)":
                format_dict[col] = "{:.2f}"

            # Cột đếm lỗi / tổng bit
            elif col.endswith("Errors") or col.endswith("Total"):
                format_dict[col] = "{:.0f}"

            # Cột BER/SER dạng số thực
            elif col.startswith("BER") or col.startswith("SER"):
                format_dict[col] = "{:.3e}"

        st.dataframe(
            result_df.style.format(format_dict),
            use_container_width=True
        )

    with tab4:
        system_df = pd.DataFrame(
            {
                "Tham số": [
                    "MIMO Scheme",
                    "Transmit Antennas",
                    "Receive Antennas",
                    "Modulation",
                    "Bits per Symbol",
                    "LDPC Code Rate",
                    "LDPC k",
                    "LDPC n",
                    "Packet/Data Symbols",
                    "Decoder Iterations",
                    "Channel",
                    "Equalizer",
                    "Eb/N0 Range",
                    "Monte Carlo Min Errors",
                    "Monte Carlo Max Bits",
                    "Batch Size",
                ],
                "Giá trị": [
                    "Alamouti/STBC",
                    "2",
                    "2",
                    "64-QAM",
                    str(system.bits_per_symbol),
                    code_rate_str,
                    str(system.k),
                    str(system.n),
                    str(system.num_data_symbols),
                    str(decoder_iter),
                    "Rayleigh + AWGN",
                    ", ".join([mode_label[m] for m in modes]),
                    f"{ebno_min:.1f} → {ebno_max:.1f} dB, step {ebno_step:.1f} dB",
                    str(min_errors),
                    f"{float(max_bits):.0e}",
                    str(batch_size),
                ],
            }
        )

        st.dataframe(system_df, use_container_width=True)

else:
    st.info("Nhấn nút Chạy mô phỏng ở sidebar để bắt đầu.")