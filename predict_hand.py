import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/hand_cnn.keras")
classes = ["help_me", "okay", "thanks"]  # auto from folder order

def extract_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70])
    upper = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    hand = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    return cv2.resize(thresh, (64,64))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    roi = frame[100:400, 100:400]
    hand = extract_hand(roi)
    img = hand / 255.0
    img = img.reshape(1,64,64,1)

    pred = model.predict(img, verbose=0)[0]
    label = classes[np.argmax(pred)]
    conf = np.max(pred)

    cv2.rectangle(frame,(100,100),(400,400),(0,255,0),2)
    cv2.putText(frame,f"{label} ({conf:.2f})",(100,90),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Hand Detection", frame)
    cv2.imshow("Mask", hand)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
