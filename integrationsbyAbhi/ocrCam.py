import cv2
from rapidocr_onnxruntime import RapidOCR

# Initialize the ONNX-backed OCR engine
engine = RapidOCR()

# Initialize webcam (0 is usually the integrated laptop camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Warning: Could not open camera device 0. Please verify camera connection.")

frame_counter = 0
current_detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 10th frame to prevent CPU bottlenecking
    if frame_counter % 10 == 0:
        # Preprocessing can be applied here to improve accuracy
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Run inference on the current frame
        result, _ = engine(frame)
        current_detections = result if result else []

    # Draw bounding boxes and text from the latest processed frame
    for detection in current_detections:
        # RapidOCR returns: [box_coordinates, text, confidence_score]
        box = detection[0]
        text = detection[1]
        
        # Extract coordinates for the rectangle
        pt1 = (int(box[0][0]), int(box[0][1]))
        pt2 = (int(box[2][0]), int(box[2][1]))
        
        # Draw on the live feed
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(frame, text, (pt1[0], pt1[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Real-Time OCR Pipeline', frame)
    frame_counter += 1

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()