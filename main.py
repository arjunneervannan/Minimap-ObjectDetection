import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import matplotlib.pyplot as plt

def fire_detect():    
    while True:
        # ret, frame = cap.read()
        frame = cv2.imread('sample_images/fire_2.jpg')
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range for the orange color in HSV
        lower_orange = np.array([15, 50, 50])
        upper_orange = np.array([35, 255, 255])

        # Create a mask for the orange color
        blur = cv2.GaussianBlur(hsv, (15, 15), 0)
        mask = cv2.inRange(blur, lower_orange, upper_orange)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around the contours on the original image
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter out small areas to reduce noise
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the original image with bounding boxes
        cv2.imshow('Image with Bounding Boxes', frame)
        
        if cv2.waitKey(30) == 27:
            break


def yolo(use_webcam=True):
    cap = cv2.VideoCapture(0)
    
    model = YOLO("yolov8l.pt")
    
    while True:
        if use_webcam:
            ret, frame = cap.read()
        else:
            frame = cv2.imread('sample_images/people_3.jpg')
        
        # YOLO V8 BASE NETWORK BELOW
        
        result = model(frame, agnostic_nms=True)[0]
        
        detections = sv.Detections.from_ultralytics(result)
        
        
        # label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        # labeled_frame = label_annotator.annotate(
        #     scene=frame,
        #     detections=detections
        # )
        
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame,
            detections=detections
        )
        
        cv2.imshow('frame', annotated_frame)
        
        if cv2.waitKey(30) == 27:
            break


if __name__ == '__main__':
    yolo(use_webcam=False)
