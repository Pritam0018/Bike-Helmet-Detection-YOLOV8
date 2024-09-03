import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import os

# Load the YOLO model
model = YOLO('D:\yolo\yolov8_helmet_detection_main\yolov8_helmet_detection_main\\best.pt')

# Define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

# Set up the window and mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video file
cap = cv2.VideoCapture('D:\yolo\yolov8_helmet_detection_main\yolov8_helmet_detection_main\\video.mp4')

# Read class names from file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

count = 0
directory = 'Predictions/'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Predict using the model
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        if c == "1":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cvzone.putTextRect(frame, 'Helmet', (x1, y1), 1, 1)
            if not os.path.exists(directory):
                os.makedirs(directory)
                
                # Save the frame as an image
            image_path = os.path.join(directory, f'{c}_{count}.jpg')
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cvzone.putTextRect(frame, 'No-Helmet', (x1, y1), 1, 1)
            cvzone.putTextRect(frame, 'WARNING', (10, 40), 3, 3, colorT=(255, 255, 255), colorR=(255, 0, 0), border=3)

    cv2.imshow("Helmet_Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
