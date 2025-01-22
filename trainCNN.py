import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CNN
import splitfolders
image_size = 32

transformTrain = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Resize((image_size, image_size), antialias=True),
    transforms.Normalize([0.5, ], [0.5, ]),

])

transformVal=transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Resize((image_size, image_size), antialias=True),
    transforms.Normalize([0.5, ], [0.5, ])
])
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2168
    splitfolders.ratio("data/ETL2", "data/etlSplit", ratio=(.8, .2))
    train = datasets.ImageFolder("data/etlSplit/train", transform=transformTrain)
    validate = datasets.ImageFolder("data/etlSplit/val", transform=transformVal)
    train_loader = DataLoader(dataset=train, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=validate, batch_size=16)

    model = CNN(image_size, 2168).to(device)
    lr = 1e-4
    optimizer =torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    epochs = 0

    trainingLosses = []
    validationLosses = []

    for epoch in range(num_epochs):
        train_avg_loss = 0
        model.train()
        for image, label in train_loader:

            img = image.to(device)
            lbl = label.to(device)
            optimizer.zero_grad()
            output = model(img)

            train_loss = criterion(output, lbl)
            train_loss.backward()
            optimizer.step()
            train_avg_loss += train_loss.item()
        train_avg_loss /= len(train_loader)
        with torch.no_grad():
            model.eval()
            eval_avg_loss = 0
            total = 0
            correct = 0
            for image, label in val_loader:
                img = image.to(device)
                lbl = label.to(device)
                output = model(img)

                predicted = torch.softmax(output.data, 1)
                _, predicted = torch.max(predicted, 1)
                #print(predicted)
                total += lbl.size(0)
                correct += (predicted == lbl).sum().item()

                eval_loss = criterion(output, lbl)
                eval_avg_loss += eval_loss.item()
            eval_avg_loss /= len(val_loader)

        print(f"epoch: {epoch + 1 + epochs} / {num_epochs+epochs}, train loss: {train_avg_loss:.4f}, evaluation loss: {eval_avg_loss:.4f}, accuracy: {100 * correct / total:.2f}%")
        trainingLosses.append(train_avg_loss)
        validationLosses.append(eval_avg_loss)
        dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + epochs + 1
        }
        torch.save(dict, "ETL2.pth")
