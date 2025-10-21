import streamlit as st
import cv2 
import numpy as np
from ball_tracker import track_ball
from data.rules import show_rules_page
from data.techniques import show_techniques_page
from data.drills import show_drills_page
from data.mental_game import show_mental_game_page
import tempfile
from utils import run_roboflow_inference
from utils import overlay_video
import os
import re

# Set page config
st.set_page_config(
    page_title="Table Tennis Trainer",
    page_icon="🏓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("Table Tennis Trainer")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Ball Detection", "Rules", "Techniques", "Training Drills", "Mental Game", "Analytics"]
)

# Home page
if page == "Home":
    st.title("Table Tennis Skills Development Platform")
    st.write("""
    Welcome to your personal table tennis training assistant. This platform combines
    real-time ball tracking technology with comprehensive educational content to help
    you improve your game at any level.
    """)
    
    st.image("assets/table_tennis_hero.jpg", use_container_width=True)
    
    st.subheader("Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Ball Tracking")
        st.write("Analyze your shots with real-time ball tracking")
    
    with col2:
        st.markdown("### 📚 Learning Resources")
        st.write("Comprehensive guides on techniques, rules, and strategy")
    
    with col3:
        st.markdown("### 🏆 Progress Tracking")
        st.write("Monitor your improvement over time")

elif page == "Ball Detection":
    st.title("Ball Detection from Uploaded Video")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "webm"])
    if uploaded_file is not None:
        save_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        input_video_path = save_path
        st.video(input_video_path)
        # Use regex to replace extension for JSON and overlay video
        base, ext = os.path.splitext(input_video_path)
        output_json_path = base + "-output.json"
        output_video_path = base + "-overlay.mp4"
        print("Output JSON path:", output_json_path)
        print("Output video path:", output_video_path)

        with st.spinner("Running inference..."):
            run_roboflow_inference(input_video_path, output_json_path)
        with st.spinner("Generating overlay video..."):
            overlay_video(input_video_path, output_json_path, output_video_path)

        st.success("Processing complete!")
        st.video(output_video_path)

# Rules page
elif page == "Rules":
    show_rules_page()

# Techniques page
elif page == "Techniques":
    show_techniques_page()

# Training Drills page
elif page == "Training Drills":
    show_drills_page()

# Mental Game page
elif page == "Mental Game":
    show_mental_game_page()

# Analytics page
elif page == "Analytics":
    st.title("Performance Analytics")
    st.write("Analytics content will go here.")
