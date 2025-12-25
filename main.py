import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pathlib

# STEP 1: Download model if missing
MODEL_PATH = "hand_landmarker.task"
model_path_obj = pathlib.Path(MODEL_PATH)

# STEP 2: Create HandLandmarker with absolute path
task_path = model_path_obj.resolve()
BaseOptions = python.BaseOptions
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(task_path)),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Hand connection lines (21 landmarks standard order)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),           # Palm connections
]

def visualize_results(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    h, w, _ = annotated_image.shape
    
    if detection_result.hand_landmarks:
        for idx, landmarks in enumerate(detection_result.hand_landmarks):
            if idx < len(detection_result.handedness):
                handedness = detection_result.handedness[idx][0].category_name
            else:
                handedness = f"Hand {idx}"
            
            # Collect all landmark points first
            landmark_points = []
            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                landmark_points.append((x, y))
                # Draw GREEN DOTS (landmarks)
                cv2.circle(annotated_image, (x, y), 8, (0, 0, 255), -2)
            
            # Draw RED LINES (connections)
            for connection in HAND_CONNECTIONS:
                if connection[0] < len(landmark_points) and connection[1] < len(landmark_points):
                    pt1 = landmark_points[connection[0]]
                    pt2 = landmark_points[connection[1]]
                    cv2.line(annotated_image, pt1, pt2, (255, 0, 0), 3)
            
            # Draw hand label
            cv2.putText(annotated_image, handedness, (10, 30 + idx*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return annotated_image

# STEP 3: Process image or webcam
def detect_hands(image_path=None, use_webcam=False):
    if use_webcam:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hands
            results = detector.detect(mp_image)
            
            # Visualize
            annotated_frame = visualize_results(rgb_frame, results)
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            cv2.imshow("Hand Landmarks (ESC/Q/q to exit)", annotated_frame_bgr)
            if cv2.waitKey(5) & 0xFF in [27, ord('q'), ord('Q')]:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return
    
    if image_path:
        image = mp.Image.create_from_file(image_path)
        rgb_image = image.numpy_view()
        results = detector.detect(image)
        annotated_image = visualize_results(rgb_image, results)
        cv2.imshow("Hand Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Usage:
if __name__ == "__main__":
    detect_hands(use_webcam=True)  
