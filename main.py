import torch
import torchvision.transforms as transforms
import easyocr
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from model import CRNN
from functions import decode_ctc_output

def detect_text_and_convert_to_tensors(image_path):
    # Initialize EasyOCR reader
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert image to grayscale
        transforms.Resize((32,)),  # Resize height to 28, keep width dynamic
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize if needed
    ])
    reader = easyocr.Reader(['ja'])  # Specify the languages you need

    # Read image
    image1 = Image.open(image_path)
    image = image1.convert("L")
    #image = ImageOps.invert(image)
    threshold = 183
    white = False
    if white:
        image = image.point(lambda x: 255 if x < threshold else 0, '1')
    else:
        image = image.point(lambda x: 0 if x < threshold else 255, '1')
    plt.imshow(image1, cmap="gray")
    plt.axis('off')
    plt.show()
    # Perform text detection

    results = reader.readtext(image_path)

    # If no text is detected, return an empty list
    if not results:
        return [transform(image)]

    # List to hold lists of tensor images
    tensor_images_list = []

    # Define the transformation


    for (bbox, text, prob) in results:
        # Unpack bounding box coordinates
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        # Crop the image to the bounding box
        cropped_image = image.crop((*top_left, *bottom_right))

        # Apply the transformation
        tensor_image = transform(cropped_image)
        #plt.imshow(cropped_image, cmap="gray")
        #plt.axis('off')
        #plt.show()

        # Append each tensor to a separate list
        tensor_images_list.append(tensor_image)

    return tensor_images_list


def predict(model, tensor_image, device):
    model.eval()  # Set the model to evaluation mode

    # evaluation mode

    prediction = []
    for i in tensor_image:
        img = i.unsqueeze(0).to(device)
        # print(img.shape)
        # Perform inference
        with torch.inference_mode():
            outputs = model(img)
            # print(f"Raw outputs shape: {outputs.shape}")

            outputs = outputs.permute(1, 0, 2)  # Permute to (sequence_length, batch_size, num_classes)
            outputs = torch.nn.functional.log_softmax(outputs, 2)
            # print(f"Permuted outputs shape: {outputs.shape}")

            predicted_text = decode_ctc_output(outputs, vocab)
            prediction.append(predicted_text)
            # print(predicted_text)
    for i in prediction:
        print(i)

if __name__ == "__main__":
    # Load vocab
    with open("data/generated/train_text_lines/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    vocab = list(vocab)
    model_path = "rnn_epoch_28.pth"



    #warming up the model for evaluation
    # Initialize model
    checkpoint = torch.load(model_path, weights_only=True)
    vocab = checkpoint["vocab"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(input_height=32, num_classes=len(vocab))
    model.load_state_dict(checkpoint["model"])
    model.to(device)



    # Example usage
    #image_path = "data/vn/Screenshot 2024-08-11 143327.png"
    image_path = "data/vn/2.jpg"
    tensor_image = detect_text_and_convert_to_tensors(image_path)


    predict(model, tensor_image, device)



