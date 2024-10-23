# name : laser_tracker
# developer : Beta Adimaker

# Import necessary libraries
# OpenCV for video capture and image processing
# Mediapipe for hand detection and tracking
# NumPy for numerical operations like interpolation
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands module
# Load the hands solution from Mediapipe
# Utility to draw hand landmarks
# Initialize hands with a minimum detection confidence
mpHand = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHand.Hands(min_detection_confidence=0.8)

# Initialize video capture from the webcam
# 0 indicates the default camera => computer cam
# Set width and height for the video feed
cap = cv2.VideoCapture(0)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

# Main loop to process each frame
# Check if the video capture is open
while cap.isOpened():
    # Capture frame by frame
    # If frame is correctly captured, success is true
    success, frame = cap.read()

    if not success:
        print("Failed to capture frame")
        break

    # Frame processing begins here:

    # Convert frame to proper np array
    # Ensure the frame is contiguous in memory
    # Ensure the image is a proper numpy array
    frame = np.ascontiguousarray(frame)
    frame = np.copy(frame)
    
    # Flip the image horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # Convert the image from BGR to RGB format as Mediapipe requires RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
  
    """
    # Explicitly create an ImageFrame with copy mode to ensure safe handling of data
    try:
        image_frame_packet = mp.packet_creator.create_image_frame(
            image_format=mp.ImageFormat.SRGB, data=frame_rgb, copy=True
        )
    except Exception as e:
        print(f"Error creating ImageFrame: {e}")
        continue
    """

    # Process the ImageFrame packet to detect hands
    frame_rgb.flags.writeable = False
    results = hands.process(frame_rgb)
    frame_rgb.flags.writeable = True

    # Convert the image back to BGR for OpenCV display
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Store detected hand landmarks
    # Initialize a list to hold landmark positions
    multiHandDetection = results.multi_hand_landmarks
    lmList = []

    # Check if any hand is detected
    if multiHandDetection:
        # Draw landmarks and connections on detected hands
        for id, lm in enumerate(multiHandDetection):
            mpDraw.draw_landmarks(
                frame,
                lm,
                mpHand.HAND_CONNECTIONS,
                mpDraw.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=7),
                mpDraw.DrawingSpec(color=(0, 0, 0), thickness=4)
            )

        # Track landmarks for the first detected hand
        # Take the first hand detected
        # Loop through each landmark
        # Get the dimensions of the image
        # Convert normalized coordinates to pixel values
        # Append the landmark position to the list
        singleHandDetection = multiHandDetection[0]
        for lm in singleHandDetection.landmark:
            h, w, c = frame.shape
            lm_x, lm_y = int(lm.x * w), int(lm.y * h)
            lmList.append([lm_x, lm_y])

    # Display the processed image
    # Show the image in a window named "Image"
    # Wait for 1 millisecond for a key press
    # If 'q' is pressed, exit the loop
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release resources
# Close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
