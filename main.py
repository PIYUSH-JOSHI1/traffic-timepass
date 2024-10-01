import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import threading
import sys

# Global variables to control the process
cap = None
model = None
running = False


def start_traffic_detection(video_source):
    global running
    running = True
    cap = cv2.VideoCapture(video_source)  # For Video
    model = YOLO("yolov8l.pt")

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    limits = [
        [935, 90, 1275, 90],
        [935, 110, 1275, 110],
        [1365, 120, 1365, 360],
        [1385, 120, 1385, 360],
        [600, 70, 600, 170],
        [620, 70, 620, 170],
        [450, 500, 1240, 500],
        [450, 520, 1240, 520]
    ]

    totalCounts = [[] for _ in range(4)]

    while running:
        success, img = cap.read()
        if not success:
            print("Video ended or cannot open.")
            break

        imgRegion = cv2.bitwise_and(img, img)
        results = model(imgRegion, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        for limit in limits:
            cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (250, 182, 122), 2)

        for result in resultsTracker:
            x1, y1, x2, y2, id = map(int, result)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(111, 237, 235))
            cvzone.putTextRect(img, f' {id}', (max(0, x1), max(25, y1)), scale=1, thickness=1,
                               colorR=(56, 245, 213), colorT=(25, 26, 25), offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (22, 192, 240), cv2.FILLED)

            # Counting logic for lanes
            lane_conditions = [
                ([935, 90, 1275, 90], totalCounts[0]),
                ([935, 110, 1275, 110], totalCounts[0]),
                ([1365, 120, 1365, 360], totalCounts[1]),
                ([1385, 120, 1385, 360], totalCounts[1]),
                ([600, 70, 600, 170], totalCounts[2]),
                ([620, 70, 620, 170], totalCounts[2]),
                ([450, 500, 1240, 500], totalCounts[3]),
                ([450, 520, 1240, 520], totalCounts[3]),
            ]

            for limit, count in lane_conditions:
                if limit[0] < cx < limit[2] and limit[1] - 15 < cy < limit[1] + 15:
                    if count.count(id) == 0:
                        count.append(id)
                        cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (12, 202, 245), 3)

        # Display counts
        for i, count in enumerate(totalCounts):
            cvzone.putTextRect(img, f' Lane {i + 1}: {len(count)}', (25, 75 + (i * 70)), 2, thickness=2,
                               colorR=(147, 245, 186), colorT=(15, 15, 15))

        cv2.imshow("Traffic_Flow_Output", img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


def stop_traffic_detection():
    global running
    running = False


def main_menu(video_source):
    while True:
        print("\nOptions:")
        print("1. Start Traffic Detection")
        print("2. Stop Traffic Detection")
        print("3. Exit")
        choice = input("Choose an option (1-3): ")

        if choice == '1':
            if not running:
                print("Starting traffic detection...")
                threading.Thread(target=start_traffic_detection, args=(video_source,), daemon=True).start()
            else:
                print("Traffic detection is already running.")
        elif choice == '2':
            stop_traffic_detection()
            print("Stopping traffic detection...")
        elif choice == '3':
            stop_traffic_detection()  # Ensure the process is stopped before exiting
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please choose again.")


if __name__ == "__main__":
    video_source = "video/vehical.mp4"  # Replace with your video source
    main_menu(video_source)
