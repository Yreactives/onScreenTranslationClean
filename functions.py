import torch
from PIL import Image
import torchvision.transforms as transforms

from torch.utils.data import Dataset
import csv
import numpy as np
import random


class OCRDataset(Dataset):
    def __init__(self, csv_file, vocab, is_train=True):
        self.data = []
        self.vocab = vocab
        self.is_train = is_train
        # Add blank token at index 0 for CTC
        self.char_to_index = {char: idx + 1 for idx, char in enumerate(vocab)}
        # Add blank token mapping
        self.char_to_index['<blank>'] = 0

        self.train_transform = transforms.Compose([
            transforms.Resize((32, ), antialias=False),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, ], [0.5, ]),


        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((32, ), antialias=True),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, ], [0.5, ])
        ])

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            self.data = [(row[0], row[1]) for row in reader]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('L')
        transform = self.train_transform if self.is_train else self.eval_transform
        image = transform(image)
        # Convert label using 1-based indexing (0 reserved for blank)
        label_indices = [self.char_to_index[char] for char in label if char in self.char_to_index]

        return image, torch.tensor(label_indices, dtype=torch.long)


def decode_ctc_output(output, vocab):
    """
    Decode the output of a CTC model into text using best path decoding
    Args:
        output: Model output tensor of shape [seq_len, batch_size, num_classes]
        vocab: List of characters in the vocabulary
    Returns:
        decoded_text: String of decoded text
    """
    # Get probabilities and best path
    #probs = torch.nn.functional.softmax(output, dim=2)
    #predicted_indices = torch.argmax(probs, dim=2)
    predicted_indices = torch.argmax(output, dim=2)
    predicted_indices = predicted_indices.squeeze().cpu().numpy()

    # Merge repeated characters and remove blanks
    decoded_text = []
    previous_index = None  # Changed from -1 to None for clarity

    for idx in predicted_indices:
        # Convert to int to avoid any numpy type issues
        idx = int(idx)

        # Skip if it's a blank token (index 0) or repeated character
        if idx != 0 and idx != previous_index:  # blank = 0
            # Subtract 1 from index since 0 is reserved for blank
            char_idx = idx - 1
            if char_idx < len(vocab):  # Add boundary check
                decoded_text.append(vocab[char_idx])
        previous_index = idx

    return ''.join(decoded_text)  # Join characters into a string

def set_deterministic_mode():
    """Set all seeds and flags for deterministic operation"""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

def save_dataset_to_npz(labels_file, output_file):
    images = []
    targets = []

    # Open CSV and skip the header
    with open(labels_file, 'r', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = row['image_path']
            label = row['label']

            # Load the image and convert it to a numpy array
            #img_path = os.path.join(image_folder, filepath)
            image = Image.open(filepath).convert('L')  # Convert to grayscale
            image = np.array(image)

            # Append image and corresponding label
            images.append(image)
            targets.append([char for char in label])  # Assuming label is a string of digits

    # Convert lists to numpy arrays and save to npz
    images = np.array(images)
    targets = np.array(targets, dtype=object)  # Keep labels as object type since they may be variable-length sequences

    np.savez_compressed(output_file, images=images, labels=targets)
    print(f"Dataset saved to {output_file}")

# Example usage:


def load_dataset_from_zre(zre_file):
    data = np.load(zre_file, allow_pickle=True)
    images = data['images']
    targets = data['labels']
    return images, targets

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class NPZDataset(Dataset):
    def __init__(self, zre_file, vocab, is_train=True):
        """
        Args:
            zre_file (str): Path to the compressed .npz file containing 'images' and 'labels'.
            vocab (list): List of characters to create a mapping to indices.
            is_train (bool): Whether the dataset is for training or evaluation (for applying different transforms).
        """
        self.images, self.labels = load_dataset_from_zre(zre_file)
        self.vocab = vocab
        self.is_train = is_train

        # Character to index mapping (with <blank> token for CTC)
        self.char_to_index = {char: idx + 1 for idx, char in enumerate(vocab)}
        self.char_to_index['<blank>'] = 0

        # Define transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((32,), antialias=False),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, ], [0.5, ]),



        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((32,), antialias=True),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, ], [0.5, ])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and label at the specified index
        image = self.images[idx]
        label = self.labels[idx]

        # Convert NumPy image array to PIL Image for transformations
        image = Image.fromarray(image)  # Convert back to PIL image if it's a NumPy array

        # Apply the appropriate transformation (train or eval)
        transform = self.train_transform if self.is_train else self.eval_transform
        image = transform(image)

        # Convert the label (characters) to indices using char_to_index mapping
        label_indices = [self.char_to_index[char] for char in label if char in self.char_to_index]

        # Convert to tensor and ensure the type is long for CTC
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        # Return the transformed image and label tensor
        return image, label_tensor


if __name__ == "__main__":
    # Example Usage
    #image_folder = 'data/generated/train_text_lines/'  # Folder where images are stored
    #labels_file = 'data/generated/train_text_lines/labels.csv'  # Path to your CSV file
    #output_file = 'train_dataset_15k.npz'  # The output file where the dataset will be saved

    #save_dataset_to_npz(labels_file, output_file)
    #images, targets = load_dataset_from_zre('train_dataset_15k.npz')
    #print(images, targets)
    pass