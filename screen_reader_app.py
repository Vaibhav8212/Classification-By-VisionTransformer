import streamlit as st
import pandas as pd
from model_utils import capture_frame, preprocess_frames, predict_frames
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title='Screen Monitor', layout='wide')
st.title('🖥️ Real-Time Screen Monitoring')
threshold = st.sidebar.slider('Prediction Threshold', 0.0, 1.0, 0.8, 0.05)
refresh_interval = st.sidebar.slider('Refresh Interval (seconds)', 0.5, 5.0, 1.0, 0.5)


if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'obscene_count' not in st.session_state:
    st.session_state.obscene_count = 0
if 'violent_count' not in st.session_state:
    st.session_state.violent_count = 0
if 'pred_log' not in st.session_state:
    st.session_state.pred_log = []

start = st.sidebar.button("▶️ Start Monitoring")
stop = st.sidebar.button("⏹️ Stop Monitoring")

if start:
    st.session_state.monitoring = True

if stop:
    st.session_state.monitoring = False
    st.session_state.obscene_count = 0
    st.session_state.violent_count = 0
    st.session_state.pred_log = []
    st.success("✅ Monitoring stopped and reset.")

if st.session_state.monitoring:
    st_autorefresh(interval=int(refresh_interval * 1000), key="refresh_key")
    st.success("✅ Monitoring running...")
    frame = capture_frame()
    preprocessed = preprocess_frames([frame])
    results = predict_frames(preprocessed, threshold)

    st.session_state.obscene_count += results['threshold_counts'].get('obscene', 0)
    st.session_state.violent_count += results['threshold_counts'].get('violent', 0)
    st.session_state.pred_log.append(results['average_predictions'])

    st.subheader('📊 Live Prediction Scores')
    for label, score in results['average_predictions'].items():
        st.write(f"**{label.capitalize()}**: {score:.2f}")

    st.subheader('⚠️ Warnings')
    if results['threshold_counts'].get('obscene', 0) > 0:
        st.warning("🚨 Obscene content detected")
    if results['threshold_counts'].get('violent', 0) > 0:
        st.warning("🚨 Violent content detected")

    st.subheader('📌 Cumulative Frame Warnings')
    st.write(f"**Obscene Frames**: {st.session_state.obscene_count}")
    st.write(f"**Violent Frames**: {st.session_state.violent_count}")

    if len(st.session_state.pred_log) > 1:
        df = pd.DataFrame(st.session_state.pred_log[-50:])
        st.line_chart(df)
else:
    st.info("Click ▶️ Start Monitoring to begin.")
