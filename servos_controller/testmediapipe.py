# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands module
mpHand = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHand.Hands(min_detection_confidence=0.8)

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
# Force window to stay on top
#cv2.setWindowProperty("Image", cv2.WND_PROP_TOPMOST, 1)

# Main loop to process each frame
while cap.isOpened():
    # Read a frame from camera
    success, img = cap.read()
    if not success:
        break

    # Image processing:
    # Mirror view
    # Convert the image from BGR to RGB format
    # Ensure img_rgb is correctly formatted
    # Process the image to detect hands
    # Convert the image back to BGR for display
    # Store detected hand landmarks
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img_rgb is None or img_rgb.ndim != 3 or img_rgb.shape[2] != 3 or img_rgb.dtype != np.uint8:
        continue
    results = hands.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    multiHandDetection = results.multi_hand_landmarks
    lmList = []

    # Check if any hand is detected
    if multiHandDetection:
        # Draw landmarks and connections on detected hands
        for id, lm in enumerate(multiHandDetection):
            mpDraw.draw_landmarks(
                img,
                lm,
                mpHand.HAND_CONNECTIONS,
                mpDraw.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=7),
                mpDraw.DrawingSpec(color=(0, 0, 0), thickness=4)
            )

        # Track landmarks for the first detected hand
        singleHandDetection = multiHandDetection[0]
        for lm in singleHandDetection.landmark:
            h, w, c = img.shape
            lm_x, lm_y = int(lm.x * w), int(lm.y * h)
            lmList.append([lm_x, lm_y])

        # Draw a circle on the tip of the index finger (landmark 8)
        myLP = lmList[8]
        px, py = myLP[0], myLP[1]
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
