
# Import necessary libraries
# OpenCV for video capture and image processing
# Mediapipe for hand detection and tracking
# NumPy for numerical operations like interpolation
# Math for distance calculations
# PyFirmata for communication between Python and Arduino
import cv2
import mediapipe as mp
import numpy as np
import math
import pyfirmata

# Input fields for camera index and Arduino COM port
indexCam = int(input("Masukkan Index Camera: "))
portCOM = input("Masukkan Port COM: ")

# Configuration for Mediapipe
# Load the hands solution from Mediapipe
# Utility to draw hand landmarks
# Initialize hands with a minimum detection confidence
mpHand = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHand.Hands(min_detection_confidence=0.8)

# Configuration for OpenCV
# Initialize video capture with the given camera index
# Set width and height for the video feed
# Set minimum and maximum ratio values
# Set minimum and maximum percentage values
# Set minimum and maximum PWM values
cap = cv2.VideoCapture(indexCam)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)
min_rat, max_rat = 20, 220  # ratio
min_per, max_per = 0, 100  # percentage
min_out, max_out = 0, 255  # pwm value

# Configuration for PyFirmata
# Connect to Arduino board via the specified COM port
# Define pin 6 as PWM for DC Motor
# Define pin 10 as PWM for LED
# Store the pins in a list for easy access
board = pyfirmata.Arduino(portCOM)
pin_PWM_1 = board.get_pin("d:6:p")
pin_PWM_2 = board.get_pin("d:10:p")
pins = [pin_PWM_1, pin_PWM_2]

# Function to calculate the position of the finger
# Multiplies the normalized coordinates by the screen size
# Converts to integer pixel values
def posFinger(finger_x, finger_y):
    pos = tuple(np.multiply(np.array((finger_x, finger_y)), [ws, hs]).astype(int))
    return pos

# Function to calculate the distance between two points
# Uses the Euclidean distance formula
def calculateDistance(pos_1, pos_2):
    length = math.hypot(pos_1[0] - pos_2[0], pos_1[1] - pos_2[1])
    return length

# Function to mark and label the fingers on the image
# Draws lines, circles, and text on the image
# Calculates the distance between landmarks to determine values for the servos
# Draws bounding boxes and value bars
def markFinger(label, pos_1, pos_2, idx):
    if len(label) == 2:
        label = label[idx]

    # Draw a line between two positions
    # Draw circles at each endpoint
    cv2.line(img, pos_1, pos_2, (0, 0, 0), 4)
    cv2.circle(img, pos_1, 15, (0, 0, 0), cv2.FILLED)
    cv2.circle(img, pos_2, 15, (0, 0, 0), cv2.FILLED)

    # Calculate distances between specific landmarks
    # Calculate vertical and horizontal lengths for determining ratios
    # Map these ratios to percentage and PWM values
    length_ver = calculateDistance(pos_indextip, pos_thumbtip)
    length_hor = calculateDistance(pos_indexmcp, pos_pinkymcp)
    length_rat = int((length_ver / length_hor) * 100)
    length_per = int(np.interp(length_rat, [min_rat, max_rat], [min_per, max_per]))
    out_value = int(np.interp(length_rat, [min_rat, max_rat], [min_out, max_out]))

    # Draw bounding boxes around the detected hand
    # Calculate bounding box coordinates based on landmarks
    xList, yList, lmList = [], [], []
    for lm in multiHandDetection[id].landmark:
        h, w, c = img.shape
        lm_x, lm_y = int(lm.x * w), int(lm.y * h)
        xList.append(lm_x)
        yList.append(lm_y)
        lmList.append([lm_x, lm_y])
        x_min, y_min = min(xList), min(yList)
        x_max, y_max = max(xList), max(yList)
        w_box, h_box = x_max - x_min, y_max - y_min

    # Draw bounding boxes and labels on the hand
    cv2.rectangle(img, (x_min - 20, y_min - 20), (x_min + w_box + 20, y_min + h_box + 20),
                  (0, 0, 0), 4)
    cv2.rectangle(img, (x_min - 22, y_min - 20), (x_min + w_box + 22, y_min - 60),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{label}', (x_min - 10, y_min - 27), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 255, 255), 3)
    cv2.putText(img, f'{out_value}', (x_max - 45, y_min - 27),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)

    # Draw value bars indicating the percentage
    length_bar = int(np.interp(length_per, [min_per, max_per], [x_min - 20, x_min + 300]))
    cv2.rectangle(img, (x_min - 20, y_min - 110), (length_bar, y_min - 80), (0, 255, 255), cv2.FILLED)
    cv2.rectangle(img, (x_min - 20, y_min - 110), (x_min + 300, y_min - 80), (0, 0, 0), 4)
    cv2.putText(img, f'{length_per}%', (x_min + 310, y_min - 85),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)

    # Normalize the output value to a range suitable for PWM
    out_value = out_value / max_out
    return out_value

# Main loop to process each frame
# Continuously capture video until 'q' is pressed
while cap.isOpened():
    # Capture frame-by-frame
    # Flip the image horizontally
    # Convert from BGR to RGB for processing with Mediapipe
    # Process the image to detect hands
    # Convert back to BGR for display
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Extract multi-hand detection results
    # Extract handedness (left or right)
    multiHandDetection = results.multi_hand_landmarks
    handType = results.multi_handedness

    # Check if hands are detected
    if multiHandDetection:
        # Visualize detected hands
        # Draw landmarks and connections on each detected hand
        for id, lm in enumerate(multiHandDetection):
            mpDraw.draw_landmarks(
                img, 
                lm, 
                mpHand.HAND_CONNECTIONS,
                mpDraw.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=7),
                mpDraw.DrawingSpec(color=(0, 0, 0), thickness=4)
            )

            # Process each detected hand
            # Calculate positions of specific finger landmarks
            # Determine hand type (left or right)
            for idx, classification in enumerate(handType):
                indexHand = classification.classification[0].index
                pos_indextip = posFinger(
                    lm.landmark[mpHand.HandLandmark.INDEX_FINGER_TIP].x,
                    lm.landmark[mpHand.HandLandmark.INDEX_FINGER_TIP].y
                )
                pos_thumbtip = posFinger(
                    lm.landmark[mpHand.HandLandmark.THUMB_TIP].x,
                    lm.landmark[mpHand.HandLandmark.THUMB_TIP].y
                )
                pos_indexmcp = posFinger(
                    lm.landmark[mpHand.HandLandmark.INDEX_FINGER_MCP].x,
                    lm.landmark[mpHand.HandLandmark.INDEX_FINGER_MCP].y
                )
                pos_pinkymcp = posFinger(
                    lm.landmark[mpHand.HandLandmark.PINKY_MCP].x,
                    lm.landmark[mpHand.HandLandmark.PINKY_MCP].y
                )

                # Determine the hand label and mark the fingers
                # Adjust the output values for each hand type
                if len(multiHandDetection) == 2:
                    if indexHand == id:
                        label = ["Left", "Right"]
                        out_value = markFinger(label, pos_indextip, pos_thumbtip, idx)
                else:
                    label = classification.classification[0].label
                    out_value = markFinger(label, pos_indextip, pos_thumbtip, idx)

            # Send the calculated value to the corresponding Arduino pin
            pins[id].write(out_value)

    else:
        print("No Hand!!!")

    # Display the processed image
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release resources
# Close all OpenCV windows
cv2.destroyAllWindows()
