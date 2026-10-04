import cv2
import numpy as np


# We use an Indian 5 Rupee coin as the reference 
# Note: Use 25.0 if you are using the newer post-2019 coin series
REFERENCE_WIDTH_MM = 23.0 

def sort_contours_left_to_right(contours):
    """Sorts contours based on the x-coordinate of their bounding box."""
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    # zip, sort by the x-coordinate (box[0]), and unzip
    contours, _ = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0]))
    return list(contours)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Edge detection - slightly tighter thresholds for the smaller 5 Rupee coin
    edged = cv2.Canny(blurred, 40, 90)
    
    # Dilate and Erode to close any gaps in the object edges
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # 2. Find Contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small noise 
    valid_contours = [c for c in contours if cv2.contourArea(c) > 500]

    if len(valid_contours) > 0:
        # Sort contours from left to right
        valid_contours = sort_contours_left_to_right(valid_contours)
        
        pixels_per_metric = None

        # 3. Loop over the detected objects
        for i, c in enumerate(valid_contours):
            # Compute the minimum bounding box (handles rotated objects)
            box = cv2.minAreaRect(c)
            box_points = cv2.boxPoints(box)
            box_points = np.int32(box_points)
            
            # Extract dimensions from the bounding box
            # box[0] is center (x,y), box[1] is (width, height), box[2] is angle
            (cx, cy), (dim1, dim2), angle = box
            
            # The first contour (leftmost) is our reference coin
            if pixels_per_metric is None:
                # We use the smaller dimension to represent the width/diameter
                coin_pixel_width = min(dim1, dim2)
                pixels_per_metric = coin_pixel_width / REFERENCE_WIDTH_MM

            # Calculate real-world dimensions for the current object
            real_dim1 = dim1 / pixels_per_metric
            real_dim2 = dim2 / pixels_per_metric

            # 4. Visualization
            # Draw the bounding box
            if i == 0:
                # Highlight the reference object in red
                color = (0, 0, 255) 
                label = f"Ref: {real_dim1:.1f}mm"
            else:
                # Highlight target objects in green
                color = (0, 255, 0)
                label = f"{real_dim1:.1f} x {real_dim2:.1f} mm"

            cv2.drawContours(frame, [box_points], -1, color, 2)
            
            # Draw the text slightly above the object
            cv2.putText(frame, label, (int(cx) - 40, int(cy) - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.imshow("Dynamic Reference Measurement", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()