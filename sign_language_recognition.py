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
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import cv2
import numpy as np
import mediapipe as mp
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a video at a specified frame rate.

    Parameters:
    - video_path: Path to the input video.
    - output_dir: Directory to save extracted frames.
    - frame_rate: Number of frames to skip between extractions.
    """
    cap = cv2.VideoCapture(video_path)
    count = 0
    success, frame = cap.read()
    while success:
        if count % frame_rate == 0:
            cv2.imwrite(os.path.join(output_dir, f"frame_{count}.jpg"), frame)
        success, frame = cap.read()
        count += 1
    cap.release()

class SignLanguageDataset(Dataset):
    def __init__(self, frames_dir, skeletons_file, labels_file, transform=None):
        """
        Initialize the dataset.

        Parameters:
        - frames_dir: Directory containing extracted frames.
        - skeletons_file: Path to the file containing skeleton data.
        - labels_file: Path to the file containing labels.
        - transform: Transformations to apply to the frames.
        """
        self.frames_dir = frames_dir
        self.skeletons = np.load(skeletons_file, allow_pickle=True).item()
        self.labels = np.load(labels_file, allow_pickle=True).item()
        self.transform = transform
        self.frame_files = sorted(os.listdir(frames_dir))

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        frame_path = os.path.join(self.frames_dir, self.frame_files[idx])
        frame = Image.open(frame_path).convert('RGB')
        if self.transform:
            frame = self.transform(frame)
        skeleton = torch.tensor(self.skeletons[self.frame_files[idx]], dtype=torch.float32)
        label = torch.tensor(self.labels[self.frame_files[idx]], dtype=torch.long)
        return frame, skeleton, label

# Example usage:
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = SignLanguageDataset(frames_dir='data/frames',
                              skeletons_file='data/skeletons.npy',
                              labels_file='data/labels.npy',
                              transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


# --- 1. Optical Flow Estimation (Lucas-Kanade) ---
def compute_optical_flow_lk(frame1, frame2, window_size=5, regularization=1e-4):
    """
    Compute Optical Flow using the Lucas-Kanade Method.

    Parameters:
        frame1 (numpy.ndarray): First frame (grayscale).
        frame2 (numpy.ndarray): Second frame (grayscale).
        window_size (int): Size of the local window for computing gradients.
        regularization (float): Small constant to avoid singular matrices.

    Returns:
        u (numpy.ndarray): Optical flow in x-direction.
        v (numpy.ndarray): Optical flow in y-direction.
    """
    # Convert to grayscale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute Image Gradients
    Ix = cv2.Sobel(frame1_gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(frame1_gray, cv2.CV_64F, 0, 1, ksize=3)
    It = frame2_gray.astype(np.float64) - frame1_gray.astype(np.float64)

    half_w = window_size // 2
    u = np.zeros_like(frame1_gray, dtype=np.float64)
    v = np.zeros_like(frame1_gray, dtype=np.float64)

    for y in range(half_w, frame1_gray.shape[0] - half_w):
        for x in range(half_w, frame1_gray.shape[1] - half_w):
            Ix_win = Ix[y - half_w: y + half_w + 1, x - half_w: x + half_w + 1].flatten()
            Iy_win = Iy[y - half_w: y + half_w + 1, x - half_w: x + half_w + 1].flatten()
            It_win = It[y - half_w: y + half_w + 1, x - half_w: x + half_w + 1].flatten()

            A = np.vstack((Ix_win, Iy_win)).T
            b = -It_win.reshape(-1, 1)

            if A.shape[0] >= 2:
                AtA = A.T @ A
                if np.linalg.det(AtA) > 1e-5:
                    v_flow = np.linalg.inv(AtA) @ (A.T @ b)
                else:
                    v_flow = np.linalg.inv(AtA + regularization * np.eye(2)) @ (A.T @ b)

                u[y, x] = v_flow[0, 0]
                v[y, x] = v_flow[1, 0]

    return u, v

# --- 2. Keyframe Extraction using Optical Flow and Skeleton Features ---
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

        # Compute motion intensity (optical flow) using LK method
        u, v = compute_optical_flow_lk(prev_frame, next_frame)
        motion_values.append(np.mean(np.sqrt(u**2 + v**2)))  # Compute motion magnitude

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

# --- 3. Skeleton Extraction using Mediapipe ---
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

# --- 4. Full Pipeline Execution ---
if __name__ == "__main__":
    video_path = "sign_language_video.mp4"
    
    # Assume skeleton_data is already extracted for the video
    skeleton_data = [extract_skeleton(cv2.imread(f"frame_{i}.jpg")) for i in range(100)]  

    keyframes = keyframe_extraction(video_path, skeleton_data)

    rgb_features, flow_features, skeleton_features = [], [], []
    for idx in keyframes:
        frame = cv2.imread(f"frame_{idx}.jpg")
        rgb_features.append(cv2.resize(frame, (256, 256)))

        # Compute Optical Flow
        if idx > 0:
            prev_frame = cv2.imread(f"frame_{idx - 1}.jpg")
            u, v = compute_optical_flow_lk(prev_frame, frame)
            flow_features.append(np.sqrt(u**2 + v**2).mean())  # Motion magnitude

        skeleton_features.append(extract_skeleton(frame))

    rgb_features = np.array(rgb_features)
    flow_features = np.array(flow_features).reshape(-1, 1)
    skeleton_features = np.array(skeleton_features)

    #train_fusion_model(rgb_features, flow_features, skeleton_features, labels=np.random.randint(0, 10, size=(len(rgb_features), 10)))  
from dataloader import SignLanguageDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize dataset and dataloader
dataset = SignLanguageDataset(frames_dir='data/frames',
                              skeletons_file='data/skeletons.npy',
                              labels_file='data/labels.npy',
                              transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Training loop
for epoch in range(num_epochs):
    for frames, skeletons, labels in dataloader:
        # Forward pass, loss computation, backward pass, and optimization
        pass

