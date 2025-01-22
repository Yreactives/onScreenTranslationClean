#trainCRNN.py
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
torch.backends.cudnn.enabled = False
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = '0'
from functions import OCRDataset, decode_ctc_output, set_deterministic_mode
set_deterministic_mode()

import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader

from model import CRNN
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

def train_data(model, dataloader, optimizer, criterion, device):

    model.train()
    total_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = [label.to(device) for label in labels]
        output = model(images, deterministic=True)
        output = output.permute(1, 0, 2)
        input_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=torch.int64).to(device)
        target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.int64).to(device)
        max_target_length = max(len(label) for label in labels)
        targets_padded = torch.stack([torch.cat([label, torch.zeros(max_target_length - len(label), dtype=torch.int64).to(device) ]) for label in labels])
        loss = criterion(output.cpu(), targets_padded.cpu(), input_lengths.cpu(), target_lengths.cpu())
        loss.backward()
        total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 7)
        optimizer.step()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    set_deterministic_mode()
    model.eval()  # Set the model to evaluation mode

    total_loss = 0

    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = [label.to(device) for label in labels]

            # Forward pass
            outputs = model(images, deterministic=True)
            outputs = outputs.permute(1, 0, 2)  # Permute to (sequence_length, batch_size, num_classes)

            # Prepare lengths and targets
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)
            target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long).to(device)

            # Padding targets for batch consistency
            max_target_length = max(len(label) for label in labels)
            targets_padded = torch.stack([
                torch.cat([label, torch.zeros(max_target_length - len(label), dtype=torch.long).to(device)])
                for label in labels
            ])

            # Compute loss
            loss = criterion(outputs, targets_padded, input_lengths, target_lengths)
            total_loss += loss.item()


    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    img = Image.open("data/generated/validation_text_lines/line_1.png")
    transform = transforms.Compose([
        transforms.Resize((32, )),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,], [0.5,])
    ])
    with torch.no_grad():
        img = transform(img).to(device)
        img = img.unsqueeze(0)
        output = model(img)
        output = output.permute(1, 0, 2)
        predicted = decode_ctc_output(output, vocab)
        print(predicted)


if __name__ == "__main__":
    set_deterministic_mode()
    #Initialization
    characterSizePath = "data/generated/train_text_lines/uniquecharactersize.pkl"
    with open(characterSizePath, "rb") as f:
        vocab = pickle.load(f)
    vocab = list(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Hyperparameters
    batch_size = 8
    lr = 1e-4
    epochs = 0
    num_epochs = 50

    print("Loading Datasets...")
    #Loading Datasets
    torch.manual_seed(0)
    trainData = OCRDataset("data/generated/train_text_lines/labels.csv", vocab)
    trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=0)
    valData = OCRDataset("data/generated/validation_text_lines/labels.csv", vocab, is_train=False)
    valLoader = DataLoader(valData, batch_size=batch_size)
    print("Datasets Loaded Successfully")

    #Model Initialization
    model = CRNN(input_height=32, num_classes=len(vocab)).to(device)
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", 0.5, 5)
    trained = False
    #Trying To Load from Checkpoint if Trained
    if trained:
        try:
            checkpoint = torch.load("rnn_epoch_7.pth")

            epochs = checkpoint["epoch"]

            model.load_state_dict(checkpoint["model"])

            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            torch.set_rng_state(checkpoint["rng_state"])
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

            print(f"last Loss: {checkpoint['loss']}")
        except Exception as e:
            print(e)
    #Training, Validation and Save
    print("Training Starts...")
    for epoch in range(num_epochs):
        loss = train_data(model, trainLoader, optimizer, criterion, device)
        scheduler.step(loss)
        print(f"Epoch [{epoch + 1 + epochs}/{num_epochs + epochs}], Avg Loss: {loss:.4f}")

        evaluate(model, valLoader, criterion, device)
        data ={
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1 + epochs,
            "loss": loss,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
        torch.save(data, f"rnn_epoch_{epoch+1+epochs}.pth")



#evaluateCRNN.py
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
torch.backends.cudnn.enabled = False
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = '0'
import pickle
from functions import decode_ctc_output, set_deterministic_mode
import numpy as np
import random
from model import CRNN
from PIL import Image
import torchvision.transforms as transforms


if __name__ == "__main__":
    set_deterministic_mode()
    #Initialization
    characterSizePath = "data/generated/train_text_lines/uniquecharactersize.pkl"
    with open(characterSizePath, "rb") as f:
        vocab = pickle.load(f)
    vocab = list(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Model Initialization
    model = CRNN(input_height=32, num_classes=len(vocab)).to(device)

    trained = True
    #Trying To Load from Checkpoint if Trained
    if trained:
        try:
            checkpoint = torch.load("rnn_epoch_7.pth")

            epochs = checkpoint["epoch"]

            model.load_state_dict(checkpoint["model"])
            torch.set_rng_state(checkpoint['rng_state'])

            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

            print(f"last Loss: {checkpoint['loss']}")
        except Exception as e:
            print(e)
    #Training, Validation and Save
    print("Evaluation Starts...")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model.eval()


    img = Image.open("data/generated/validation_text_lines/line_1.png")
    transform = transforms.Compose([
        transforms.Resize((32,)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, ], [0.5, ])
    ])
    with torch.no_grad():
        img = transform(img).to(device)
        img = img.unsqueeze(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        output = model(img, deterministic=True)
        output = output.permute(1, 0, 2)
        predicted = decode_ctc_output(output, vocab)
        print(predicted)




#model.py
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.enabled= False
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        #self.fc = nn.Linear(128*8*8, 2168)

    def forward(self, x):
        x = self.layers(x)
        #x = self.fc(x.view(x.size(0), -1))

        return x

class CRNN(nn.Module):
    def __init__(self, input_height=28, num_classes=10):
        super(CRNN, self).__init__()
        self.cnn = CNN()

        # Load pretrained weights while excluding the fully connected layer
        pretrained_weight = torch.load("ETL2.pth", weights_only=False)["model"]

        # loading only the weights for convolutional layers
        model_dict = self.cnn.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_weight.items() if k in model_dict and 'fc' not in k}
        model_dict.update(pretrained_dict)


        # Load CNN weights with strict=False to avoid missing keys
        self.cnn.load_state_dict(model_dict, strict=False)

        for param in self.cnn.parameters():
            param.requires_grad = True



        # Calculate input size for LSTM based on CNN output
        lstm_input_size = 128 * (input_height // 4)  # Assuming pooling reduces height by a factor of 4
        self.hidden_size = 256
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size * 2, num_heads=8)

        # Define LSTM layers
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=self.hidden_size,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.5

        )

        self.lstm2 = nn.LSTM(input_size=self.hidden_size * 2,
                             hidden_size=self.hidden_size,
                             num_layers=2,
                             bidirectional=True,
                             batch_first=True,
                             dropout=0.5

        )
        self.layer_norm1 = nn.LayerNorm(self.hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size * 2)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(self.hidden_size * 2, num_classes + 1)  # Output classes + 1 for blank
        self.log_softmax = nn.LogSoftmax(2)


    def _init_hidden(self, batch_size, device):
        if not self.training:
            torch.manual_seed(0)
        h0 = torch.zeros(4, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(4, batch_size, self.hidden_size).to(device)
        return (h0, c0)

    def _init_weights(self):
        """Initialize weights to prevent loss spikes"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'fc' in name:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def reset_states(self):
        """Reset any stateful parts of the model"""
        for layer in self.modules():
            if isinstance(layer, nn.LSTM):
                layer.flatten_parameters()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.reset_running_stats()


    def forward(self, x, deterministic=False):
        if deterministic:
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(0)
        device = x.device

        output = self.cnn(x)  # Pass input through CNN
        bs, c, h, w = output.size()  # Get batch size, channels, height, width

        output = output.permute(0, 2, 3, 1)  # Change dimensions for LSTM input
        output = output.contiguous().view(bs, w, -1)  # Reshape for LSTM

        # Get deterministic hidden states
        hidden1 = self._init_hidden(bs, device)
        hidden2 = self._init_hidden(bs, device)

        # First LSTM
        lstm1_out, _ = self.lstm(output, hidden1)
        lstm1_out = self.layer_norm1(lstm1_out)

        # Attention with fixed seed for evaluation
        if deterministic:
            torch.manual_seed(0)

        #self attention mechanism
        lstm1_out_transpose = lstm1_out.transpose(0, 1)
        attended_out, _ = self.attention(lstm1_out_transpose, lstm1_out_transpose, lstm1_out_transpose, need_weights=False)
        attended_out = attended_out.transpose(0, 1)


        # Second LSTM
        lstm2_out, _ = self.lstm2(attended_out, hidden2)
        lstm2_out = self.layer_norm2(lstm2_out)

        output = lstm1_out + lstm2_out

        # Final Classification
        if self.training:
            output = self.dropout(output)

        output = self.fc(output)
        output = self.log_softmax(output)  # Apply log softmax

        return output  # Return the LSTM output





