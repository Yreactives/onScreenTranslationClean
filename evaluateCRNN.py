import torch
import torch.nn as nn
import os
import pickle
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.backends.cudnn import deterministic

from model import CRNN
from functions import decode_ctc_output, set_deterministic_mode, NPZDataset
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import LambdaLR

def evaluate_model(model, image_path, vocab):

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Variables

    #model = CRNN(input_height=32, num_classes=len(vocab)).to(device)
    #checkpoint = torch.load(model_path)
    #model.load_state_dict(checkpoint["model"])


    # Set model to eval mode
    model.eval()

    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((32,), antialias=True),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, ], [0.5, ])
    ])

    # Load and process image
    with Image.open(image_path) as img:
        img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference with deterministic settings
    with torch.inference_mode():
        output = model(img_tensor)

        #output = output.permute(1, 0, 2)
        output = torch.nn.functional.log_softmax(output, 2)
        predicted = decode_ctc_output(output, vocab)

    return predicted
def warmup_scheduler_lambda(current_step: int, warmup_steps: int):
    if current_step < warmup_steps:
        return current_step / warmup_steps
    return 1.0

def train_data(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0



    for images, labels in dataloader:
        optimizer.zero_grad()
        images = images.to(device)

        # Get batch size
        batch_size = images.size(0)

        # Handle padding for the batch
        max_label_length = max(len(label) for label in labels)
        padded_labels = torch.zeros(batch_size, max_label_length, dtype=torch.long)
        label_lengths = torch.zeros(batch_size, dtype=torch.long)

        # Fill in the padded labels and lengths
        for idx, label in enumerate(labels):
            label_length = len(label)
            padded_labels[idx, :label_length] = label
            label_lengths[idx] = label_length

        # Move tensors to device
        padded_labels = padded_labels.to(device)
        label_lengths = label_lengths.to(device)

        # Forward pass
        outputs = model(images)  # [batch_size, seq_length, num_classes]

        # Prepare inputs for CTC loss
        log_probs = nn.functional.log_softmax(outputs, dim=2)

        # Get input lengths (sequence length for each sample in batch)
        input_lengths = torch.full(size=(batch_size,),
                                   fill_value=log_probs.size(1),
                                   dtype=torch.long).to(device)

        # Compute loss
        loss = criterion(log_probs.transpose(0, 1),  # [seq_length, batch_size, num_classes]
                         padded_labels,
                         input_lengths,
                         label_lengths)
        loss.backward()
        total_loss += loss.item()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()



    return total_loss / len(dataloader)

if __name__ == "__main__":
    # Load vocabulary
    with open("data/generated/train_text_lines/uniquecharactersize.pkl", "rb") as f:
        vocab = pickle.load(f)
    vocab = list(vocab)

    # Run evaluation
    model_path = "rnn_epoch_2.pth"
    checkpoint = torch.load(model_path)
    vocab = checkpoint["vocab"]
    image_path = "data/generated/validation_text_lines/line_1.png"
    #image_path = "data/vn/Screenshot 2024-08-11 143230.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(32, len(vocab)).to(device)


    model.load_state_dict(checkpoint["model"])


    batch_size = 4
    trainData = NPZDataset("train_dataset.npz", vocab, is_train=True)
    trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=False, drop_last=True)
    #losses = checkpoint["losses"]

    predicted_text = evaluate_model(model, image_path, vocab)
    print(f"Predicted Text: {predicted_text}")
