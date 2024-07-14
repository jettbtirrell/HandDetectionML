from ultralytics import YOLO
import cv2
import torch

# Load the trained model
model = YOLO('C:/Users/jettbt/runs/detect/train19/weights/best.pt')

# Load an image
img_path = 'path_to_test_image.jpg'
img = cv2.imread(img_path)

# Run inference
results = model(img)

# Print results
print(results)

# Draw bounding boxes on the image
annotated_img = results.plot()

# Display the image
cv2.imshow('Detected Hands', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
