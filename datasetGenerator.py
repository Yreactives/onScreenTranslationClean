import random
from PIL import Image
import os
import csv
import pickle

unique_chars = []

def generate_text_line_from_etl(char_folders, output_path, text_length=15):
    selected_chars = []
    label = ""
    spacing = 1

    for _ in range(text_length):
        char_folder = random.choice(char_folders)
        image_files = [f for f in os.listdir(char_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            continue
        char_image = random.choice(image_files)
        selected_chars.append(os.path.join(char_folder, char_image))
        character = chr(int(os.path.basename(char_folder), 16))
        label += character
        if character not in unique_chars:
            unique_chars.append(character)

    if not selected_chars:
        return None

    images = [Image.open(img_path).convert('L').resize((32, 32)) for img_path in selected_chars]

    width = sum(img.width for img in images) + (len(images) - 1) * spacing
    height = max(img.height for img in images)
    canvas = Image.new('L', (width, height), color=255)

    x_offset = 5
    for img in images:
        canvas.paste(img, (x_offset, 0))
        x_offset += img.width + spacing

    canvas.save(output_path)
    return label

def generate_dataset(char_folders, num_samples=10, output_dir="testingpurpose", text_length = 15):
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "labels.csv")
    with open(csv_file, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])

        for i in range(num_samples):
            output_path = os.path.join(output_dir, f"line_{i}.png")
            label = generate_text_line_from_etl(char_folders, output_path, text_length)

            if label:
                writer.writerow([output_path, label])
            if (i + 1) % 50 == 0:
                print(f"{((i + 1) / num_samples * 100):.1f}%")

    save_variable(unique_chars, output_dir+"/vocab.pkl")


# Example usage:
def save_variable(variable, filename:str):
    with open(filename, "wb") as f:
        pickle.dump(variable, f)
if __name__ == "__main__":
    output_character_images = "data/ETL2"

    char_folders = [os.path.join(output_character_images, folder) for folder in os.listdir(output_character_images)]
    generate_dataset(char_folders, 8000, "data/generated/train_text_lines", 15)
    generate_dataset(char_folders, 2000, "data/generated/validation_text_lines", 15)
    print(len(unique_chars))