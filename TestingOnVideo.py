from ultralytics import YOLO
import cv2
import torch

# Loading the trained model
model = YOLO('C:\\Users\\jettb\\runs\\detect\\train23\\weights\\best.pt')

# Opening video capture (0 defaults to a webcam, could be switched to a path to .mp4)
cap = cv2.VideoCapture(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_names = ['myleft','myright','yourleft','yourright']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Runs inference
    results = model(frame, conf = 0.6, device=device)

    # Processes results if available
    if results and len(results) > 0:
        boxes = results[0].boxes # This is a list of all detections the model found in this frame
        for box in boxes:
            bbox = box.xyxy[0].cpu().numpy()  # Convert to numpy array for easier handling
            conf = box.conf.item()
            cls = int(box.cls.item())
            class_string = class_names[cls]
            
            label = f'Class: {class_string}, Conf: {conf:.2f}'

            # Draw bounding box on frame
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Detected Hands', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()