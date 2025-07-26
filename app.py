import streamlit as st
import tempfile
import os
import time

from keyframe import extract_keyframes
from model_utils import predict

st.set_page_config(page_title='Video Classification', layout='centered')
st.title('Video Classification App')

uploaded_file = st.file_uploader('Upload a video file', type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    st.video(uploaded_file)

    st.markdown('---')
    st.markdown('🔍 **Classifying the video...**')
    start_time = time.time()
    with st.spinner('Running model on extracted keyframes...'):
        results = predict(extract_keyframes, video_path)
    end_time = time.time()

    st.markdown('---')
    if 'error' in results:
        st.error(f'❌ {results["error"]}')
    else:
        st.success('✅ Classification Complete')
        st.write(f'⏱️ Time taken: **{end_time - start_time:.2f} seconds**')

        st.subheader('📊 Average Predictions:')
        for label, score in results['average_predictions'].items():
            st.write(f"**{label.capitalize()}**: {score:.2f}")

        st.subheader('📈 Frame Counts Above Threshold:')
        for label, count in results['threshold_counts'].items():
            st.write(f"**{label.capitalize()}**: {count} frames")

        st.subheader('🧠 Final Predicted Class:')
        st.write(f"**{results['prediction'].capitalize()}**")

    os.remove(video_path)
