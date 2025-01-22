import easyocr
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from model import CNN
from PIL import Image
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans JP'

def sliding_window(roi, window_size, step_size):
    (w, h) = window_size
    windows = []  # List to hold the windows
    for y in range(0, roi.shape[0] - h + 1, step_size):
        for x in range(0, roi.shape[1] - w + 1, step_size):
            window = roi[y:y + h, x:x + w]
            windows.append((x, y, window))  # Store the position and the window
    return windows

# Initialize EasyOCR Reader
reader = easyocr.Reader(['ja'])  # 'ja' for Japanese, add other languages as needed

# Load your image
image_path = 'data/vn/1.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform OCR
results = reader.readtext(image)

# Load the trained model
checkpoint = torch.load("ETL.pth")
model = CNN()
model.load_state_dict(checkpoint["model"])
model.eval()  # Set the model to evaluation mode

# Process each detected text region
for (bbox, text, prob) in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))

    # Extract ROI
    roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Generate sliding windows
    windows = sliding_window(roi, (32, 32), 20)

    # Process each sliding window (e.g., make predictions)
    for i, (x, y, window) in enumerate(windows):
        # Convert window to a format suitable for your CNN model
        window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        window_tensor = window_tensor / 255.0  # Normalize if needed

        # Make predictions with your CNN model
        with torch.no_grad():
            output = model(window_tensor)
            _, predicted = torch.max(output, 1)  # Get the predicted class

        print(f'Prediction for window {i} at ({x}, {y}): Class {predicted.item()}')

    # Display the ROI with the detected text (optional)
    plt.imshow(roi, cmap="gray")
    plt.axis('off')  # Hide axis
    plt.title(f'Detected Text: {text}')
    plt.show()
