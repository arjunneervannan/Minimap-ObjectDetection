import cv2
from ultralytics import YOLO
import supervision as sv

def main():
    cap = cv2.VideoCapture(0)
    model = YOLO("yolov8l.pt")
    
    while True:
        ret, frame = cap.read()
        
        
        result = model(frame, agnostic_nms=True)[0]
        
        detections = sv.Detections.from_ultralytics(result)
        
        # labels = [
        #     f"{model.model.names[class_id]}, {confidence:.2f}"
        #     for _, confidence, class_id, *_ in detections
        # ]
        
        
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        labeled_frame = label_annotator.annotate(
            scene=frame,
            detections=detections
        )
        
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        annotated_frame = bounding_box_annotator.annotate(
            scene=labeled_frame,
            detections=detections
        )
        
        cv2.imshow('frame', annotated_frame)
        
        if cv2.waitKey(30) == 27:
            break


if __name__ == '__main__':
    main()
