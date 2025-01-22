import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import CRNN
import pickle
from functions import NPZDataset, OCRDataset, decode_ctc_output
from PIL import Image
import torchvision.transforms as transforms

def train_batch(images, labels):
    optimizer.zero_grad()

    # Move images to device
    images = images.to(device)
    labels = labels.to(device)
    # Get batch size
    batch_size = images.size(0)

    # Forward pass
    outputs = model(images)  # [batch_size, seq_length, num_classes]

    # Prepare inputs for CTC loss
    log_probs = nn.functional.log_softmax(outputs, dim=2)

    # Get input lengths (sequence length for each sample in batch)
    input_lengths = torch.full(size=(batch_size,),
                               fill_value=log_probs.size(1),
                               dtype=torch.long).to(device)
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long).to(device)
    # Compute loss
    loss = criterion(log_probs.transpose(0, 1),  # [seq_length, batch_size, num_classes]
                     labels,
                     input_lengths,
                     label_lengths)
    return loss

# Initialization

data_path = "data/generated"

characterSizePath = f"{data_path}/train_text_lines/vocab.pkl"
with open(characterSizePath, "rb") as f:
    vocab = pickle.load(f)
vocab = list(vocab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 4
lr = 1e-4
num_epochs = 50

print("Loading Datasets...")
#trainData = OCRDataset(f"{data_path}/train_text_lines/labels.csv", vocab, is_train=True)
trainData = NPZDataset("train_dataset.npz", vocab, is_train=True)
trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True, drop_last=True)
valData = OCRDataset(f"{data_path}/validation_text_lines/labels.csv", vocab, is_train=False)
valLoader = DataLoader(valData, batch_size=batch_size, drop_last=True)
print("Datasets Loaded Successfully")

# Model setup
model = CRNN(32, len(vocab))
model.to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
#checkpoint = torch.load("rnn_epoch_12.pth", weights_only=True)
epochs = 0
#model.load_state_dict(checkpoint["model"])
#optimizer.load_state_dict(checkpoint["optimizer"])
#scheduler.load_state_dict(checkpoint["scheduler"])
#print(checkpoint["lr"])
#for g in optimizer.param_groups:
 #   g["lr"] = checkpoint["lr"][-1]
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(trainLoader):

        loss = train_batch(images, labels)
        # Backward pass and optimize
        loss.backward()

        """
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2)
                print(f"Gradient norm for {param.shape}: {grad_norm.item()}")
        """

        optimizer.step()

        running_loss += loss.item()
        """
        if (batch_idx + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Batch [{batch_idx + 1}/{len(trainLoader)}], '
                  f'Loss: {loss.item():.4f}')
        """
    # Print epoch statistics
    epoch_loss = running_loss / len(trainLoader)
    print(f'Epoch [{epoch + 1+epochs}/{num_epochs+epochs}], Average Loss: {epoch_loss:.4f}', end=" ")

    model.eval()
    total_loss = 0

    with torch.inference_mode():  # Disable gradient calculation
        for images, labels in valLoader:
            images = images.to(device)
            labels = labels.to(device)
            # Get batch size
            batch_size = images.size(0)

            # Forward pass
            outputs = model(images)  # [batch_size, seq_length, num_classes]

            # Prepare inputs for CTC loss
            log_probs = nn.functional.log_softmax(outputs, dim=2)

            # Get input lengths (sequence length for each sample in batch)
            input_lengths = torch.full(size=(batch_size,),
                                       fill_value=log_probs.size(1),
                                       dtype=torch.long).to(device)
            label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long).to(device)
            # Compute loss
            loss = criterion(log_probs.transpose(0, 1),  # [seq_length, batch_size, num_classes]
                             labels,
                             input_lengths,
                             label_lengths)
            total_loss += loss.item()

    avg_loss = total_loss / len(valLoader)
    print(f"Validation Loss: {avg_loss:.4f}")
    scheduler.step(avg_loss)
    img = Image.open(f"{data_path}/validation_text_lines/line_1.png")
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
        output = torch.nn.functional.log_softmax(output, 2)
        output = output.permute(1, 0, 2)
        predicted = decode_ctc_output(output, vocab)
        print(predicted)
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "vocab": vocab,
        "epoch": epoch+epochs+1,
        "scheduler": scheduler.state_dict(),
        "lr": scheduler.get_last_lr(),

    }
    torch.save(data, f"rnn_epoch_{epoch+1+epochs}.pth")