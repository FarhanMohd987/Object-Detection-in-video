import cv2
import torch
import numpy as np
from collections import deque
import json
import csv
import os
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import math

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

motion_csv_file = "performance_log.csv"
velocity_csv_file = "velocity_log.csv"

if not os.path.exists(motion_csv_file):
    with open(motion_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Object Class", "Motion Events", "Trajectory Points"])

if not os.path.exists(velocity_csv_file):
    with open(velocity_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Timestamp", "Object Class", "Velocity"])

object_tracks = {}
max_len = 50
motion_data = {}
fps_list = []
frame_count = 0
processed_frames = 0
start_time = cv2.getTickCount()

cap = cv2.VideoCapture(r'C:\Users\mohdf\OneDrive\Desktop\hehe\2165-155327596_small.mp4')

root = tk.Tk()
root.title("Real-Time Object Detection and Tracking")
root.geometry("800x500")
video_label = Label(root)
video_label.pack()

def update_frame():
    global frame_count, processed_frames, start_time

    ret, frame = cap.read()
    if not ret:
        cap.release()
        root.quit()
        return

    if frame_count % 5 != 0:
        frame_count += 1
        root.after(10, update_frame)
        return

    frame = cv2.resize(frame, (640, 360))
    frame_count += 1
    processed_frames += 1

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    detected_ids = []
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        label = f"{model.names[cls]} {conf:.2f}"

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        center = ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2)
        detected_ids.append((cls, center))

        if cls not in object_tracks:
            object_tracks[cls] = deque(maxlen=max_len)
            motion_data[cls] = {'motion_events': 0}

        if len(object_tracks[cls]) > 0:
            last_position = object_tracks[cls][-1]
            if center != last_position:
                motion_data[cls]['motion_events'] += 1

            if model.names[cls] == "car":
                dx = center[0] - last_position[0]
                dy = center[1] - last_position[1]
                distance = math.sqrt(dx**2 + dy**2)
                velocity = distance

                with open(velocity_csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([frame_count, processed_frames, "car", velocity])

        object_tracks[cls].append(center)

    for cls, trajectory in object_tracks.items():
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)

    end_time = cv2.getTickCount()
    time_spent = (end_time - start_time) / cv2.getTickFrequency()
    fps = int(1 / time_spent) if time_spent > 0 else 0
    fps_list.append(fps)
    start_time = end_time

    with open(motion_csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for cls in object_tracks.keys():
            writer.writerow([
                frame_count,
                model.names[cls],
                motion_data[cls]['motion_events'],
                len(object_tracks[cls])
            ])

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

root.after(0, update_frame)
root.mainloop()

average_fps = np.mean(fps_list)

performance_report = {
    "average_fps": average_fps,
    "motion_analysis": {
        cls: {
            "motion_events_count": motion_data[cls]['motion_events'],
            "trajectory_points_count": len(object_tracks[cls])
        }
        for cls in object_tracks.keys()
    },
    "detected_object_classes": list(model.names[cls] for cls in object_tracks.keys()),
    "total_frames_processed": processed_frames,
    "strengths": [
        "Fast detection with YOLOv5, capable of real-time processing.",
        "Consistent tracking of objects across frames, with trajectory visualization.",
        "Simple motion event detection for tracking movement."
    ],
    "limitations": [
        "May struggle with object occlusion in crowded scenes.",
        "Performance can degrade in low lighting conditions.",
        "Assumes uniform frame intervals for motion calculation."
    ]
}

with open("performance_report.json", "w") as f:
    json.dump(performance_report, f, indent=4)

print("Report saved as 'performance_report.json'")
