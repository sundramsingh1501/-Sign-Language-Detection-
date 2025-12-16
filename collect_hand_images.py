import cv2
import numpy as np
import os

LABEL = "help_me"   # change
SAVE_DIR = f"data/train/{LABEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

def extract_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    hand = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    return thresh

print("Press 's' to save hand image, 'q' to quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    roi = frame[100:400, 100:400]
    hand = extract_hand(roi)
    hand = cv2.resize(hand, (64, 64))

    cv2.rectangle(frame, (100,100), (400,400), (0,255,0), 2)
    cv2.imshow("Hand Mask", hand)
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        cv2.imwrite(f"{SAVE_DIR}/{count}.jpg", hand)
        print("Saved", count)
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
