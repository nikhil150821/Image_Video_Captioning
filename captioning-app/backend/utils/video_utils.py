import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def sample_video_frames(video_path, num_frames=100):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transforms.ToTensor()(Image.fromarray(frame))
            frames.append(frame_tensor)
    cap.release()
    return torch.stack(frames)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

def preprocess_video_frames(frames_tensor):
    return torch.stack([transform(frame) for frame in frames_tensor])
