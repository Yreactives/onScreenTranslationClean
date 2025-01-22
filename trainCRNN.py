import numpy.random
import torch
import numpy as np
import random
import os
from functions import OCRDataset, decode_ctc_output, NPZDataset
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from model import CRNN
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
def print_model_structure(model):
    for name, module in model.named_modules():
        print(f"{name}: {module}")

def check_model_params(model):
    for name, param in model.named_parameters():
        print(f"{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Data type: {param.dtype}")
        print(f"  Mean: {param.mean().item()}")
        print(f"  Std: {param.std().item()}")

def check_data_sample(dataloader):
    batch = next(iter(dataloader))
    inputs, targets = batch
    print(f"Input shape: {inputs.shape}")
    print(f"Input dtype: {inputs.dtype}")
    print(f"Input mean: {inputs.mean().item()}")
    print(f"Input std: {inputs.std().item()}")
    print(f"Target shape: {targets.shape}")
    print(f"Target dtype: {targets.dtype}")

def test_forward_pass(model, dataloader):
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        inputs, _ = batch
        outputs = model(inputs.to(torch.device("cuda")))
        print(f"Output shape: {outputs.shape}")
        print(f"Output mean: {outputs.mean().item()}")
        print(f"Output std: {outputs.std().item()}")

def check_loss_computation(model, dataloader, criterion):
    model.eval()
    with torch.no_grad():
        device = torch.device("cuda")
        batch = next(iter(dataloader))
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        outputs = outputs.permute(1, 0, 2)  # Permute to (sequence_length, batch_size, num_classes)

        # Prepare lengths and targets
        input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)
        target_lengths = torch.tensor([len(label) for label in targets], dtype=torch.long).to(device)

        # Padding targets for batch consistency
        max_target_length = max(len(label) for label in targets)
        targets_padded = torch.stack([
            torch.cat([label, torch.zeros(max_target_length - len(label), dtype=torch.long).to(device)])
            for label in targets
        ])
        loss = criterion(outputs, targets_padded, input_lengths, target_lengths)
        print(f"Computed loss: {loss.item()}")

def set_all_seeds(seed_value=42):
    # PyTorch
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    # Python random
    random.seed(seed_value)

    # NumPy random
    np.random.seed(seed_value)

    # Ensure CUBLAS is deterministic
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)


import torch.nn.functional as F


def train_data(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    every_loss = []

    for images, labels in dataloader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = [label.to(device) for label in labels]

        output = model(images)  # Forward pass

        output = output.permute(1, 0, 2)  # Permute to [seq_len, batch_size, num_classes]
        log_probs = F.log_softmax(output, 2)# Apply log softmax

        # Calculate lengths
        input_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=torch.long).to(device)
        target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long).to(device)

        # Flatten targets for CTC loss
        targets_padded = torch.cat([label for label in labels]).to(device)

        # Calculate loss
        loss = criterion(log_probs, targets_padded, input_lengths, target_lengths)
        every_loss.append(loss.item())

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader), every_loss


import torch.nn.functional as F

def evaluate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0

    with torch.inference_mode():  # Disable gradient calculation
        for images, labels in dataloader:
            images = images.to(device)
            labels = [label.to(device) for label in labels]

            # Forward pass
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)  # Permute to (sequence_length, batch_size, num_classes)
            log_probs = F.log_softmax(outputs, 2)

            # Prepare lengths and targets
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)
            target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long).to(device)

            # Flatten targets for CTC loss
            targets_padded = torch.cat([label for label in labels]).to(device)

            # Compute loss
            loss = criterion(log_probs, targets_padded, input_lengths, target_lengths)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")


    img = Image.open("data/generated/validation_text_lines/line_1.png")
    transform = transforms.Compose([
        transforms.Resize((32,)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, ], [0.5, ])
    ])
    with torch.inference_mode():
        img = transform(img).to(device)
        img = img.unsqueeze(0)
        output = model(img)
        output = output.permute(1, 0, 2)
        predicted = decode_ctc_output(output, vocab)
        print(predicted)

if __name__ == "__main__":

    # Initialization
    characterSizePath = "data/generated/train_text_lines/uniquecharactersize.pkl"
    with open(characterSizePath, "rb") as f:
        vocab = pickle.load(f)
    vocab = list(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 4
    lr = 1e-4
    epochs = 0
    num_epochs = 50

    print("Loading Datasets...")
    # Loading Datasets
    #trainData = OCRDataset("data/generated/train_text_lines/labels.csv", vocab)
    trainData = NPZDataset("train_dataset.npz", vocab, is_train=True)
    trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True, drop_last=True)
    valData = OCRDataset("data/generated/validation_text_lines/labels.csv", vocab, is_train=False)
    valLoader = DataLoader(valData, batch_size=batch_size, drop_last=True)
    print("Datasets Loaded Successfully")

    # Model Initialization
    model = CRNN(input_height=32, num_classes=len(vocab)).to(device)
    criterion = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", 0.5, 5)
    trained = True

    # Trying To Load from Checkpoint if Trained
    if trained:
        try:
            checkpoint = torch.load("rnn_epoch_4.pth")
            epochs = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model"])  # Correctly loading model state
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            #torch.set_rng_state(checkpoint["rng_state"])
            #torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
            print("Checkpoint keys:", checkpoint.keys())
            print("Model state dict keys:", checkpoint['model_state_dict'].keys())
            model_state = model.state_dict()
            for key in model_state.keys():
                if key not in checkpoint['model_state_dict']:
                    print(f"Missing key in checkpoint: {key}")
            print(f"Last Loss: {checkpoint['loss']}")
        except Exception as e:
            print(e)


    # Training, Validation, and Save
    print("Training Starts...")
    losses = []
    for epoch in range(num_epochs):
        #evaluate(model, valLoader, criterion, device)
        loss, every_losses = train_data(model, trainLoader, optimizer, criterion, device)
        losses.append(every_losses)
        scheduler.step(loss)
        print(f"Epoch [{epoch + 1 + epochs}/{num_epochs + epochs}], Train Loss: {loss:.4f}")

        evaluate(model, valLoader, criterion, device)
        data = {
            "model": model.state_dict(),  # Save the model state dict
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1 + epochs,
            "loss": loss,
            "losses": losses
        }


        #torch.save(data, f"rnn_epoch_{epoch + 1 + epochs}.pth")
