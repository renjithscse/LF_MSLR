"""
Sign Language Recognition using Multimodal Fusion
=================================================
This project implements a late fusion-based multimodal sign language recognition model 
using RGB video frames, Optical Flow, and Skeleton Data.

Author: S Renjith
GitHub: https://github.com/renjithscse/LF_MSLR
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LSTM, Dense, Flatten, TimeDistributed, Bidirectional
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# --- 1. Keyframe Extraction using Optical Flow and skeleton features---
def compute_optical_flow(prev_frame, next_frame):
    """Computes optical flow magnitude between consecutive frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 
                                        pyr_scale=0.5, levels=3, winsize=15, 
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)

def compute_skeleton_change(skeleton_prev, skeleton_next):
    """Computes Euclidean distance between consecutive skeleton frames."""
    return np.linalg.norm(np.array(skeleton_next) - np.array(skeleton_prev))

def keyframe_extraction(video_path, skeleton_data, alpha=1.5, beta=1.5):
    """
    Extracts keyframes based on motion intensity (optical flow) and skeleton changes.
    
    Parameters:
    - video_path: Path to the input video.
    - skeleton_data: List of skeleton keypoints for each frame.
    - alpha, beta: Hyperparameters for threshold tuning.
    
    Returns:
    - keyframes: List of extracted keyframe indices.
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    motion_values = []
    skeleton_values = []

    frame_idx = 0
    skeleton_prev = skeleton_data[frame_idx]

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break
        
        # Compute motion intensity (optical flow)
        motion_values.append(compute_optical_flow(prev_frame, next_frame))

        # Compute skeleton-based change
        skeleton_next = skeleton_data[frame_idx + 1]
        skeleton_values.append(compute_skeleton_change(skeleton_prev, skeleton_next))

        prev_frame = next_frame
        skeleton_prev = skeleton_next
        frame_idx += 1

    cap.release()

    # Compute thresholds
    T_motion = np.mean(motion_values) + alpha * np.std(motion_values)
    T_skeleton = np.mean(skeleton_values) + beta * np.std(skeleton_values)

    keyframes = []
    for i in range(len(motion_values)):
        if motion_values[i] > T_motion or skeleton_values[i] > T_skeleton:
            keyframes.append(i)

    return keyframes

# --- 2. Skeleton Extraction using Mediapipe ---
mp_pose = mp.solutions.pose
def extract_skeleton(frame):
    """
    Extracts skeleton keypoints from a frame using Mediapipe.

    Parameters:
        frame (numpy array): Input image frame.

    Returns:
        numpy array: Skeleton keypoints [x, y, z] coordinates.
    """
    pose = mp_pose.Pose()
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    skeleton = []
    
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            skeleton.append([lm.x, lm.y, lm.z])
    
    return np.array(skeleton)

# --- 3. CNN Feature Extraction ---
def build_cnn_model(input_shape):
    """
    Builds a CNN-BiLSTM model for feature extraction and classification.

    Parameters:
        input_shape (tuple): Shape of the input data.

    Returns:
        Sequential: Compiled Keras model.
    """
    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  # Adjust based on class count
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 4. GCN for Skeleton Feature Extraction ---
class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) model for skeleton-based feature extraction.
    """
    def __init__(self, in_features, hidden_dim, out_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_features)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# --- 5. Fusion & Training ---
def train_fusion_model(rgb_features, flow_features, skeleton_features, labels):
    """
    Trains a fusion model combining RGB, Optical Flow, and Skeleton features.

    Parameters:
        rgb_features (numpy array): RGB-based features.
        flow_features (numpy array): Optical Flow-based features.
        skeleton_features (numpy array): Skeleton-based features.
        labels (numpy array): Ground truth labels.

    Returns:
        None
    """
    fused_features = np.concatenate([rgb_features, flow_features, skeleton_features], axis=1)

    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(fused_features.shape[1], fused_features.shape[2])),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # Adjust based on dataset
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(fused_features, labels, epochs=50, batch_size=32, validation_split=0.2)

# --- 6. Full Pipeline Execution ---
if __name__ == "__main__":
    video_path = "sign_language_video.mp4"
    keyframes = extract_keyframes(video_path)

    rgb_features, flow_features, skeleton_features = [], [], []
    for frame in keyframes:
        rgb_features.append(cv2.resize(frame, (256, 256)))
        flow_features.append(cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150))  # Edge detection
        skeleton_features.append(extract_skeleton(frame))

    rgb_features = np.array(rgb_features)
    flow_features = np.array(flow_features)
    skeleton_features = np.array(skeleton_features)

    train_fusion_model(rgb_features, flow_features, skeleton_features, labels=np.random.randint(0, 10, size=(len(rgb_features), 10)))  # Dummy labels
