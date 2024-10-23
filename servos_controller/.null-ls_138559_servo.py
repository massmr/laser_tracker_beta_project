# name : laser_tracker
# source-code : https://github.com/rizkydermawan1992/handtrackinga
# developer : Beta Adimaker

# Import necessary libraries
# OpenCV for video capture and image processing
# Mediapipe for hand detection and tracking
# NumPy for numerical operations like interpolation
# PyFirmata for communication between Python and Arduino
import cv2
import mediapipe as mp
import numpy as np
import pyfirmata

# Initialize Mediapipe Hands module
# Load the hands solution from Mediapipe
# Utility to draw hand landmarks
# Initialize hands with a minimum detection confidence
mpHand = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHand.Hands(min_detection_confidence=0.8)

# Initialize video capture from the webcam
# 0 indicates the default camera
# Set width and height for the video feed
# Set the width of the video feed
# Set the height of the video feed
cap = cv2.VideoCapture(0)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

# Initialize Arduino communication
# Define the COM port to which Arduino is connected
# Connect to Arduino via the specified COM port
# Define pin 9 as a servo control pin for the x-axis
# Define pin 10 as a servo control pin for the y-axis
port = "COM5"
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:9:s')
servo_pinY = board.get_pin('d:10:s')

# Main loop to process each frame
# Check if the video capture is open
while cap.isOpened():
    # Read a frame from the webcam
    # Flip the image horizontally for mirror view
    # Convert the image from BGR to RGB for processing
    # Process the image to detect hands
    # Convert the image back to BGR for display
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Store detected hand landmarks
    # Initialize a list to hold landmark positions
    multiHandDetection = results.multi_hand_landmarks
    lmList = []

    # Check if any hand is detected
    if multiHandDetection:
        # Draw landmarks and connections on detected hands
        # Draw landmarks on the hand
        # Landmark styling
        # Connection styling
        for id, lm in enumerate(multiHandDetection):
            mpDraw.draw_landmarks(
                img,
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
            h, w, c = img.shape
            lm_x, lm_y = int(lm.x * w), int(lm.y * h)
            lmList.append([lm_x, lm_y])

        # Print the list of landmarks for debugging
        print(lmList)

        # Draw a circle on the tip of the index finger (landmark 8)
        # Get the coordinates of the index finger tip
        # Extract x and y positions
        # Draw a circle at the fingertip
        # Display coordinates
        # Draw a horizontal line at y position
        # Draw a vertical line at x position
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

        # Map the finger position to servo angles
        # Map x position to 0-180 degrees
        # Map y position to 0-180 degrees
        # Draw a rectangle to display servo values
        # Display servo X value
        # Display servo Y value
        servoX = int(np.interp(px, [0, ws], [180, 0]))
        servoY = int(np.interp(py, [0, hs], [0, 180]))
        cv2.rectangle(img, (40, 20), (350, 110), (0, 255, 255), cv2.FILLED)
        cv2.putText(
            img,
            f'Servo X: {servoX} deg',
            (50, 50),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 0),
            2
        )
        cv2.putText(
            img,
            f'Servo Y: {servoY} deg',
            (50, 100),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 0),
            2
        )

        # Write the mapped values to the servos
        # Control the x-axis servo
        # Control the y-axis servo
        servo_pinX.write(servoX)
        servo_pinY.write(servoY)

        # Print the hand position and corresponding servo values for debugging
        print(f'Hand Position x: {px} y: {py}')
        print(f'Servo Value x: {servoX} y: {servoY}')

    # Display the processed image
    # Show the image in a window named "Image"
    # Wait for 1 millisecond for a key press
    # If 'q' is pressed, exit the loop
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release resources
# Close all OpenCV windows
cv2.destroyAllWindows()
