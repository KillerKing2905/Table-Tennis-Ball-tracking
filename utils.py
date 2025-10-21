import cv2
import numpy as np
import streamlit as st
from roboflow import Roboflow
import json

def run_roboflow_inference(input_video_path, output_json_path):
    rf = Roboflow(api_key="EJuRj2c5k4WSeHZHmqG7")
    project = rf.workspace().project("ball-ugsvu-pejz9-l1ln1")
    model = project.version("1").model

    job_id, signed_url, expire_time = model.predict_video(
        input_video_path,
        fps=24,
        prediction_type="batch-video",
    )

    results = model.poll_until_video_results(job_id)

    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file, indent=2)
    return output_json_path

def overlay_video(input_video_path, json_path, output_video_path):
    with open(json_path, 'r') as json_file:
        inference_data = json.load(json_file)

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if inference_data.get("status_info") == "decode error":
        st.error("Inference failed: decode error. Please check your video format.")
        return None
    
    frame_idx = 0
    for frame_data in inference_data['ball-ugsvu-pejz9-l1ln1']:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        for prediction in frame_data['predictions']:
            x = int(prediction['x'])
            y = int(prediction['y'])
            w = int(prediction['width'])
            h = int(prediction['height'])
            confidence = prediction['confidence']
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            coords_text = f"x: {x}, y: {y}"
            text_size = cv2.getTextSize(coords_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = width - text_size[0] - 10
            text_y = 30
            cv2.putText(frame, coords_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    return output_video_path