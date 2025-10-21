Link to the Web APP :- https://tabletennisanalyser.streamlit.app/

AI-Powered Table Tennis: Trajectory Prediction & Optimal Shot Strategy
Overview

This project integrates AI and computer vision to develop a table tennis training system that predicts ball trajectories and recommends optimal shots. It uses YOLOv11 for real-time object detection, multi-angle video for 3D trajectory analysis, and Streamlit for a user-friendly interface.

Team:

Siddharth Goyal (2022MEB1346)

Spandan Seth (2022MEB1348)

Sunny Bharti (2022MEB1353)

Supervisor:

Dr. Navin Kumar (IIT Ropar)

Features

AI-based Trajectory Prediction: Uses YOLOv11 for detecting the ball and paddle in real-time.

3D Trajectory Reconstruction: Multi-angle video from two iPhone cameras for accurate ball trajectory analysis.

Real-time Feedback: Streamlit app provides performance metrics, trajectory visualizations, and shot recommendations.

Methodology

Multi-Angle Capture: Two synchronized iPhone cameras (60 FPS) for precise 3D reconstruction.

Object Detection: YOLOv11 model detects ball and racket.

Trajectory Analysis: 3D position estimation and shot recommendation system.

Streamlit Interface: Displays real-time feedback and performance metrics.

Ball Thrower: Arduino-controlled spring-loaded mechanism for consistent practice shots.

Installation

Clone the repo:

git clone https://github.com/your-repo.git


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py

Challenges

Low FPS (60 FPS) affected fast ball tracking.

Lighting and synchronization issues in video capture.

Hardware limitations during ball thrower integration.

Acknowledgments

Thanks to Dr. Navin Kumar for his guidance throughout this project.

Future Work

Higher FPS cameras for better tracking.

Enhanced AI model training for improved accuracy.

Integration with professional coaching systems.
