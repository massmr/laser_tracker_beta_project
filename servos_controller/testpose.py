# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe pose module (same structure as hand landmarks)
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

ws, hs = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ws)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hs)

# Create a named window and set properties to ensure it appears correctly
cv2.namedWindow("Image", cv2.WINDOW_GUI_EXPANDED)

# Main loop to process each frame
while cap.isOpened():
    # Read a frame from the camera
    success, img = cap.read()
    if not success:
        break

    # Image processing:
    # Mirror view
    img = cv2.flip(img, 1)
    
    # Convert the image from BGR to RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image to detect pose landmarks
    results = mpPose.process(img_rgb)
    
    # Convert the image back to BGR for display
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    lmList = []

    # Check if any pose landmarks are detected
    if results.pose_landmarks:
        # Draw landmarks and connections on detected pose
        mpDraw.draw_landmarks(
            img,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mpDraw.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=5),
            mpDraw.DrawingSpec(color=(0, 0, 0), thickness=4)
        )
        
        # Extract pose landmark coordinates
        for lm in results.pose_landmarks.landmark:
            h, w, c = img.shape
            lm_x, lm_y = int(lm.x * w), int(lm.y * h)
            lmList.append([lm_x, lm_y])
            
        # Example: Draw a circle on the nose (landmark 0)
        if len(lmList) > 0:
            nose = lmList[0]  # Pose landmark 0 is the nose
            px, py = nose[0], nose[1]
            cv2.circle(img, (px, py), 15, (255, 0, 255), cv2.FILLED)
            cv2.putText(
                img,
                str((px, py)),
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                3
            )
            cv2.line(img, (0, py), (ws, py), (0, 0, 0), 2)
            cv2.line(img, (px, hs), (px, 0), (0, 0, 0), 2)

    # Display the processed image
    cv2.imshow("Image", img)
    
    # Properly handle the exit condition
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
