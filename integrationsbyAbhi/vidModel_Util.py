import cv2
from ultralytics import YOLO

# Expose model for the main app pipeline
_cached_vid_model = None

def get_vid_model():
    global _cached_vid_model
    if _cached_vid_model is None:
        try:
            # Using the path generated in your Colab environment
            _cached_vid_model = YOLO('/content/runs/detect/billet_detector-2/weights/best.pt') 
        except Exception as e:
            print(f"Warning: Failed to load vidModel_Util model from colab path: {e}")
            _cached_vid_model = None
    return _cached_vid_model

def run_custom_yolo_inference(input_path="Billet Video.mp4", output_path="Custom_YOLO_Output.mp4", fps=30.0):
    # 1. Load your newly fine-tuned weights
    model = get_vid_model()
    if model is None:
        print("Cannot run inference, model failed to load.")
        return

    cap = cv2.VideoCapture(input_path)
    
    # Initialize the OpenCV VideoWriter
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print(f"Running custom YOLO model on '{input_path}'...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 2. Run inference on the current frame
        # conf=0.5 sets a minimum confidence threshold of 50%
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # 3. Render the bounding boxes and labels
        # YOLO's built-in plot() method automatically draws the detected boxes 
        # and class labels onto a copy of the frame
        annotated_frame = results[0].plot()
        
        # 4. Write the annotated frame directly to the output video
        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 1000 == 0:
            print(f"Processed {frame_count} frames...")
            
    cap.release()
    out.release()
    print(f"Inference complete! Final video saved to '{output_path}'")

# Execute the inference pipeline
run_custom_yolo_inference()