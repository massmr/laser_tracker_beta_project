
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Failed to capture image.")
        break

    # Flip and convert to RGB
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display to ensure correct handling
    cv2.imshow("RGB Image", img_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


