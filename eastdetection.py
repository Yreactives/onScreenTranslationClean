import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt

img = cv2.imread("data/vn/1.png")
model = cv2.dnn.readNet("model/frozen_east_text_detection.pb")

height, width, _ = img.shape
new_height = (height//32) * 32
new_width = (width//32) * 32
print(new_height, new_width)

h_ratio = height/new_height
w_ratio = width/new_width

print(h_ratio, w_ratio)


blob = cv2.dnn.blobFromImage(img, 1, (new_width, new_height), (123.68, 116.78, 103.94), True, False)

model.setInput(blob)

(geometry, scores) = model.forward(model.getUnconnectedOutLayersNames())

rectangles = []
confidence_score = []

for i in range(geometry.shape[2]):
    for j in range(geometry.shape[3]):
        if scores[0][0][i][j] < 0.1:
            continue

        bottom_x = int(j * 4 + geometry[0][1][i][j])
        bottom_y = int(i * 4 + geometry[0][2][i][j])

        top_x = int(j * 4 - geometry[0][3][i][j])
        top_y = int(i * 4 - geometry[0][0][i][j])

        rectangles.append((top_x, top_y, bottom_x, bottom_y))
        confidence_score.append(float(scores[0][0][i][j]))

fin_boxes = non_max_suppression(np.array(rectangles), probs=confidence_score, overlapThresh=0.5)

img_copy = img.copy()

for (x1, y1, x2, y2) in fin_boxes:
    x1 = int(x1 * w_ratio)
    y1 = int(y1 * h_ratio)
    x2 = int(x2 * w_ratio)
    y2 = int(y2 * h_ratio)

    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.imshow(img)
plt.show()
plt.imshow(img_copy)
plt.show()

