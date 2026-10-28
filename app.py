"""
app.py  --  MachinePulse, the Streamlit interface.
Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_utils import (FEATURES, INPUT_COLUMNS, LABELS, add_features,
                        load_dataset, missing_columns, standardize_columns)
from pipeline import (feature_influence, load_bundle, predict_proba_df,
                      risk_level, train_model)

st.set_page_config(page_title="MachinePulse", page_icon="⚙️", layout="wide")

LEVEL_COLORS = {"Healthy": "#2E8B57", "Warning": "#E0A100", "Critical": "#C0392B"}
ADVICE = {
    "Healthy": "The machine is running inside its safe range. Keep to the routine inspection schedule.",
    "Warning": "Risk is rising. Plan an inspection in the next maintenance window and watch the readings.",
    "Critical": "Failure is likely. Reduce the load or stop the machine and send it for maintenance now.",
}
PRESETS = {
    "Healthy machine": dict(in_type="M", in_air=300.0, in_proc=310.0, in_rpm=1500, in_torque=40.0, in_wear=60),
    "Overheating": dict(in_type="L", in_air=303.0, in_proc=310.5, in_rpm=1300, in_torque=45.0, in_wear=90),
    "Heavy overload": dict(in_type="L", in_air=300.0, in_proc=310.0, in_rpm=1400, in_torque=68.0, in_wear=200),
    "Worn tool under load": dict(in_type="L", in_air=300.0, in_proc=310.0, in_rpm=1500, in_torque=55.0, in_wear=230),
    "Near the limit": dict(in_type="H", in_air=300.0, in_proc=310.0, in_rpm=1500, in_torque=56.0, in_wear=225),
}
PAGES = ["Overview", "Train the ANN", "Live prediction", "Batch prediction"]

# ----------------------------------------------------------------------------
# Look and feel: one deliberate element, the pulse trace in the header
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@600;700&display=swap');
.mp-banner{display:flex;align-items:center;justify-content:space-between;gap:1.5rem;background:#1B3445;border-radius:6px;padding:1.2rem 1.6rem;margin-bottom:1.2rem;}
.mp-title{font-family:'Barlow Semi Condensed',sans-serif;font-weight:700;font-size:2.5rem;line-height:1.05;color:#ffffff;}
.mp-sub{color:#B9CAD6;font-size:1rem;margin-top:0.3rem;}
.mp-trace{width:42%;height:56px;flex-shrink:0;}
.mp-trace polyline{fill:none;stroke:#4FD1E0;stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round;}
.mp-badge{display:inline-block;padding:0.3rem 1rem;border-radius:4px;color:#ffffff;font-weight:600;font-size:1.1rem;}
@media (max-width:700px){.mp-trace{display:none;}}
</style>
<div class="mp-banner">
<div><div class="mp-title">MachinePulse</div><div class="mp-sub">Predictive maintenance with an artificial neural network</div></div>
<svg class="mp-trace" viewBox="0 0 320 60" preserveAspectRatio="none"><polyline points="0,32 70,32 84,32 96,8 108,54 120,22 132,32 200,32 214,32 226,6 238,54 250,24 262,32 320,32"/></svg>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data and model state
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data():
    return load_dataset()


try:
    df, is_demo = get_data()
except ValueError as err:
    st.error(str(err))
    st.stop()

if "bundle" not in st.session_state:
    st.session_state["bundle"] = load_bundle()          # None if nothing saved yet


def require_model():
    bundle = st.session_state.get("bundle")
    if bundle is None:
        st.info("There is no trained network yet. Open “Train the ANN” in the sidebar and press Train network.")
    return bundle


def parse_layers(text):
    try:
        sizes = [int(p) for p in text.replace(" ", "").split(",") if p]
    except ValueError:
        return None
    if not 1 <= len(sizes) <= 4 or any(s < 2 or s > 256 for s in sizes):
        return None
    return sizes


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    page = st.radio("Go to", PAGES, label_visibility="collapsed")
    st.divider()
    if is_demo:
        st.warning("Generated demo data is in use. Put ai4i2020.csv inside the data folder, restart the app and train again to use the real dataset.")
    else:
        st.success("Real dataset loaded: ai4i2020.csv")
    model = st.session_state.get("bundle")
    if model is None:
        st.info("No trained network yet.")
    elif model["trained_on_demo"] and not is_demo:
        st.warning("The saved network was trained on demo data. Train it again on the real dataset.")
    else:
        st.success("A trained network is ready.")


# ----------------------------------------------------------------------------
# Page 1: overview
# ----------------------------------------------------------------------------
def page_overview():
    total, fails = len(df), int(df["failure"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Machine records", f"{total:,}")
    c2.metric("Recorded failures", f"{fails:,}")
    c3.metric("Failure rate", f"{fails / total * 100:.2f}%")
    c4.metric("ANN input features", len(FEATURES))

    tab_data, tab_sensor, tab_cause = st.tabs(["Data preview", "Sensor behaviour", "Failure causes"])
    with tab_data:
        st.dataframe(df.head(200))
    with tab_sensor:
        sensor = st.selectbox("Sensor", ["air_temp", "process_temp", "rpm", "torque", "tool_wear"],
                              format_func=lambda k: LABELS[k])
        plot_df = df.copy()
        plot_df["Status"] = np.where(plot_df["failure"] == 1, "Failure", "Normal")
        fig = px.histogram(plot_df, x=sensor, color="Status", barmode="overlay", opacity=0.7,
                           nbins=40, labels=LABELS,
                           color_discrete_map={"Normal": "#5B8FA8", "Failure": "#C0392B"})
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Machines")
        st.plotly_chart(fig)
        st.caption("Red bars show where the failed machines sit compared with healthy ones.")
    with tab_cause:
        modes = {"TWF": "Tool wear", "HDF": "Heat dissipation", "PWF": "Power failure",
                 "OSF": "Overstrain", "RNF": "Random"}
        present = [m for m in modes if m in df.columns]
        if present:
            causes = pd.DataFrame({"Cause": [modes[m] for m in present],
                                   "Failures": [int(df[m].sum()) for m in present]})
            fig = px.bar(causes, x="Cause", y="Failures", color_discrete_sequence=["#0B7A8C"])
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig)
        else:
            st.info("This dataset has no failure-cause columns.")

    with st.expander("How MachinePulse works"):
        st.markdown(
            "1. **Sensors** report air temperature, process temperature, rotational speed, torque and tool wear.\n"
            "2. **Feature engineering** adds three physical quantities: temperature difference, mechanical power and strain.\n"
            "3. **Scaling** puts every input on the same scale so that no single sensor dominates.\n"
            "4. **The neural network** passes the inputs through hidden ReLU layers and a sigmoid output, "
            "and learns by back-propagation with the Adam optimiser.\n"
            "5. **The output** is a failure probability, shown as Healthy, Warning or Critical."
        )


# ----------------------------------------------------------------------------
# Page 2: training
# ----------------------------------------------------------------------------
def show_results(bundle):
    st.divider()
    st.subheader("Results on unseen test data")
    for col, (name, value) in zip(st.columns(5), bundle["metrics"].items()):
        col.metric(name, f"{value:.3f}" if name == "ROC-AUC" else f"{value * 100:.1f}%")
    st.caption("Recall is the share of real failures the network caught. Precision is the share of its alarms that were real.")

    left, right = st.columns(2)
    with left:
        hist = bundle["history"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=hist["loss"], mode="lines", name="Training loss",
                                 line=dict(color="#0B7A8C", width=3)))
        fig.add_trace(go.Scatter(y=hist["val_loss"], mode="lines", name="Validation loss",
                                 line=dict(color="#E0A100", width=3)))
        fig.update_layout(title="Learning curve", xaxis_title="Epoch", yaxis_title="Weighted loss",
                          height=380, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig)
    with right:
        fig = px.imshow(bundle["confusion"], text_auto=True, color_continuous_scale="Teal",
                        x=["Predicted normal", "Predicted failure"],
                        y=["Actually normal", "Actually failure"])
        fig.update_layout(title="Confusion matrix", height=380, coloraxis_showscale=False,
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig)

    layers = " and ".join(str(n) for n in bundle["hidden_layers"])
    st.caption(f"Network layout: {len(FEATURES)} inputs, hidden layers of {layers} neurons, 1 output. "
               f"It has {bundle['parameters']:,} trainable parameters and was tested on {bundle['test_size']:,} unseen machines.")


def page_train():
    st.subheader("Train the neural network")
    left, right = st.columns(2)
    with left:
        layers_text = st.text_input("Hidden layers (neurons per layer, separated by commas)", "32,16")
        learning_rate = st.select_slider("Learning rate", options=[0.0005, 0.001, 0.003, 0.005, 0.01], value=0.003)
        batch_size = st.select_slider("Batch size", options=[32, 64, 128, 256], value=64)
    with right:
        epochs = st.slider("Maximum epochs", 10, 300, 100, step=10)
        pos_weight = st.slider("Failure class weight", 1.0, 20.0, 3.0, step=0.5,
                               help="Failures are rare. A higher weight makes the network pay more attention to them: more failures caught, more false alarms.")
        st.caption("Training stops early when the validation loss stops improving.")

    if st.button("Train network", type="primary"):
        hidden = parse_layers(layers_text)
        if hidden is None:
            st.error("Enter 1 to 4 layer sizes between 2 and 256, separated by commas. Example: 32,16")
        else:
            bar = st.progress(0.0, text="Starting training")

            def on_epoch(epoch, total, loss, val_loss):
                bar.progress(epoch / total, text=f"Epoch {epoch} of {total}: loss {loss:.4f}, validation loss {val_loss:.4f}")

            st.session_state["bundle"] = train_model(
                df, hidden, learning_rate, epochs, batch_size, pos_weight,
                is_demo=is_demo, progress_callback=on_epoch)
            bar.progress(1.0, text="Training finished and the network was saved")

    bundle = st.session_state.get("bundle")
    if bundle is not None:
        show_results(bundle)


# ----------------------------------------------------------------------------
# Page 3: live prediction
# ----------------------------------------------------------------------------
def apply_preset(name):
    for key, value in PRESETS[name].items():
        st.session_state[key] = value


def make_gauge(prob, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Probability of failure"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": 0.3},
               "steps": [{"range": [0, 25], "color": "rgba(46,139,87,0.25)"},
                         {"range": [25, 50], "color": "rgba(224,161,0,0.25)"},
                         {"range": [50, 100], "color": "rgba(192,57,43,0.25)"}]}))
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def page_live():
    bundle = require_model()
    if bundle is None:
        return
    for key, value in PRESETS["Healthy machine"].items():
        st.session_state.setdefault(key, value)

    st.subheader("Check one machine")
    st.caption("Move the sliders or load a ready-made scenario.")
    for col, name in zip(st.columns(len(PRESETS)), PRESETS):
        col.button(name, on_click=apply_preset, args=(name,))

    inputs, result = st.columns([1, 1.1], gap="large")
    with inputs:
        st.selectbox("Product quality type", ["L", "M", "H"], key="in_type")
        st.slider("Air temperature [K]", 290.0, 310.0, step=0.1, key="in_air")
        st.slider("Process temperature [K]", 300.0, 320.0, step=0.1, key="in_proc")
        st.slider("Rotational speed [rpm]", 1000, 3000, step=10, key="in_rpm")
        st.slider("Torque [Nm]", 0.0, 80.0, step=0.5, key="in_torque")
        st.slider("Tool wear [min]", 0, 260, step=1, key="in_wear")

    row = pd.DataFrame([{
        "type": st.session_state["in_type"], "air_temp": st.session_state["in_air"],
        "process_temp": st.session_state["in_proc"], "rpm": st.session_state["in_rpm"],
        "torque": st.session_state["in_torque"], "tool_wear": st.session_state["in_wear"],
    }])
    prob = float(predict_proba_df(bundle, row)[0])
    level = risk_level(prob)

    with result:
        st.plotly_chart(make_gauge(prob, LEVEL_COLORS[level]))
        st.markdown(f'<span class="mp-badge" style="background:{LEVEL_COLORS[level]}">{level}</span>',
                    unsafe_allow_html=True)
        st.write(ADVICE[level])

    derived = add_features(row).iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Temperature difference", f"{derived['temp_diff']:.1f} K")
    m2.metric("Mechanical power", f"{derived['power_w']:,.0f} W")
    m3.metric("Strain (wear x torque)", f"{derived['strain']:,.0f}")

    st.subheader("What is driving this result")
    _, influence = feature_influence(bundle, row)
    effect = pd.DataFrame({"Reading": [LABELS[k] for k in influence],
                           "Change in failure probability": list(influence.values())})
    effect["Direction"] = np.where(effect["Change in failure probability"] >= 0, "Raises risk", "Lowers risk")
    effect = effect.sort_values("Change in failure probability")
    fig = px.bar(effect, x="Change in failure probability", y="Reading", orientation="h",
                 color="Direction", color_discrete_map={"Raises risk": "#C0392B", "Lowers risk": "#2E8B57"})
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig)
    st.caption("Each bar shows how much this reading changes the failure probability compared with an average value for that reading.")


# ----------------------------------------------------------------------------
# Page 4: batch prediction
# ----------------------------------------------------------------------------
def page_batch():
    bundle = require_model()
    if bundle is None:
        return
    st.subheader("Check many machines at once")

    sample = df.sample(min(25, len(df)), random_state=3)[INPUT_COLUMNS].copy()
    sample.columns = ["Type", "Air temperature [K]", "Process temperature [K]",
                      "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    st.download_button("Download a sample file", sample.to_csv(index=False).encode("utf-8"),
                       file_name="machinepulse_sample.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload a CSV file with sensor readings", type=["csv"])
    if uploaded is None:
        st.info("The file needs these columns: Type, Air temperature, Process temperature, "
                "Rotational speed, Torque and Tool wear. Units in brackets are optional.")
        return
    try:
        raw = pd.read_csv(uploaded)
    except Exception:
        st.error("That file could not be read as a CSV.")
        return

    data = standardize_columns(raw)
    missing = missing_columns(data, INPUT_COLUMNS)
    if missing:
        st.error(f"These columns are missing: {', '.join(missing)}")
        return

    prob = predict_proba_df(bundle, data)
    result = raw.copy()
    result["Failure probability (%)"] = (prob * 100).round(1)
    result["Risk level"] = [risk_level(p) for p in prob]

    counts = result["Risk level"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Healthy", int(counts.get("Healthy", 0)))
    c2.metric("Warning", int(counts.get("Warning", 0)))
    c3.metric("Critical", int(counts.get("Critical", 0)))
    st.dataframe(result)
    st.download_button("Download results", result.to_csv(index=False).encode("utf-8"),
                       file_name="machinepulse_predictions.csv", mime="text/csv")


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
{"Overview": page_overview, "Train the ANN": page_train,
 "Live prediction": page_live, "Batch prediction": page_batch}[page]()