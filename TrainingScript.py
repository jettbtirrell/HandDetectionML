import torch
from ultralytics import YOLO

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize YOLO model
#model = YOLO('yolov8s.yaml')  # Use a YOLOv8 config

# Continue Training
model = YOLO('C:\\Users\\jettb\\runs\\detect\\train23\\weights\\best.pt')

if __name__ == '__main__':
    # Train the model
    model.train(
        data='egohands/data.yaml', 
        epochs=100, 
        imgsz=640, 
        device=device,
        augment=True,
        mixup=0.1
    )
