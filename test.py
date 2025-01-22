import torch
checkpoint = torch.load("model/CSAR-MBiLNet.pth")
print(checkpoint["lr"])