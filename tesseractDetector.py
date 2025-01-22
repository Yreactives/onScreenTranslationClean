import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

pytesseract.pytesseract.tesseract_cmd = r"F:\Tesseract-OCR\tesseract.exe"

image_path = "data/vn/1.png"
image = Image.open(image_path).convert('L')

custom_config = r'--oem 3 --psm 5 -l jpn'
data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

# Create a figure and axis for plotting
fig, ax = plt.subplots(1)
ax.imshow(image, cmap="gray")

# Loop through the results and draw rectangles for detected characters
for i in range(len(data['text'])):
    if data['text'][i].strip():  # Ignore empty results
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        # Create a rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

# Show the image with rectangles
plt.axis('off')  # Turn off axis
plt.show()