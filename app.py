import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import time

from src.media_loader import load_media
from src.face_validator import validate_media_for_faces
from src.face_extractor import extract_faces
from src.detectors import load_detectors, run_detectors
from src.ensemble import run_ensemble

# Page configuration — must be first streamlit command
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="centered"
)

# Load detectors once when app starts
# st.cache_resource keeps them in memory across reruns


@st.cache_resource
def get_detectors():
    return load_detectors()


# Header
st.title("🔍 Deepfake Detector")
st.markdown("Upload an image or video to analyze it for manipulation.")
st.divider()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a file",
    type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
    help="Supported formats: JPG, PNG, MP4, AVI, MOV"
)

if uploaded_file is not None:
    # Save uploaded file temporarily to disk
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Show uploaded media preview
    if suffix.lower() in ['.jpg', '.jpeg', '.png']:
        st.image(uploaded_file, caption="Uploaded Image",
                 use_container_width=True)
    else:
        st.video(uploaded_file)

    st.divider()
# Analyze button
    if st.button("🔍 Analyze Media", type="primary", use_container_width=True):

        with st.spinner("Loading detectors..."):
            detector_a, detector_b, detector_c = get_detectors()

        with st.spinner("Loading media..."):
            result = load_media(tmp_path)

        if result['error']:
            st.error(result['error'])

        else:
            with st.spinner("Validating faces..."):
                validation = validate_media_for_faces(result['frames'])

            if not validation['has_face']:
                st.warning(validation['message'])

            else:
                with st.spinner("Extracting faces..."):
                    faces = extract_faces(
                        result['frames'],
                        validation['face_details']
                    )

                with st.spinner("Running analysis..."):
                    all_face_scores = []
                    for face in faces:
                        scores = run_detectors(
                            face['face_tensor'],
                            face['face_crop'],
                            detector_a, detector_b, detector_c
                        )
                        all_face_scores.append(scores)

                    final = run_ensemble(all_face_scores)

                # Clean up temp file
                os.unlink(tmp_path)

                # Store result in session state
                st.session_state['result'] = final
                st.session_state['faces'] = faces
                st.session_state['media_type'] = result['media_type']
# Display results if available
if 'result' in st.session_state:
    final = st.session_state['result']
    faces = st.session_state['faces']
    media_type = st.session_state['media_type']

    st.subheader("Analysis Results")

    # Prediction banner
    if final['prediction'] == 'FAKE':
        st.error(f"⚠️ FAKE MEDIA DETECTED")
    else:
        st.success(f"✅ REAL MEDIA")

    # Confidence and stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", final['prediction'])
    with col2:
        st.metric("Confidence", f"{final['confidence']}%")
    with col3:
        st.metric("Faces Analyzed", final['total_faces_analyzed'])

    # Confidence bar
    st.markdown("**Manipulation Confidence:**")
    st.progress(int(final['final_score'] * 100))

    # Reasons
    if final['reasons']:
        st.divider()
        st.markdown("**Reasons:**")
        for reason in final['reasons']:
            st.markdown(f"• {reason}")

    # Individual detector scores
    st.divider()
    st.markdown("**Detector Breakdown:**")
    scores = final['individual_scores']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Face Swap",
            f"{round(scores['face_swap_score'] * 100, 1)}%"
        )
    with col2:
        st.metric(
            "AI Generated",
            f"{round(scores['ai_generated_score'] * 100, 1)}%"
        )
    with col3:
        st.metric(
            "Manual Edit",
            f"{round(scores['manual_edit_score'] * 100, 1)}%"
        )

    # Show extracted face crops
    st.divider()
    st.markdown("**Detected Faces:**")
    cols = st.columns(min(len(faces), 4))
    for i, face in enumerate(faces):
        with cols[i % 4]:
            face_rgb = cv2.cvtColor(face['face_crop'], cv2.COLOR_BGR2RGB)
            st.image(face_rgb, caption=f"Face {i+1}", use_container_width=True)
