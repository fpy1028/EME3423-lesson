import cv2
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_text = "OFF "

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img, handLms, mp_hands.HAND_CONNECTIONS
            )

            # Landmark index
            # Thumb tip = 4
            # Index tip = 8
            thumb_tip = handLms.landmark[4]
            index_tip = handLms.landmark[8]

            # Convert to pixel coordinates
            thumb_y = int(thumb_tip.y * h)
            index_y = int(index_tip.y * h)

            # Thumbs Up logic
            if thumb_y < index_y:
                gesture_text = "ON "

    # Display result (any position)
    cv2.rectangle(img, (20, 20), (300, 90), (0, 0, 0), -1)
    cv2.putText(
        img,
        f"Gesture: {gesture_text}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("MP Hand Gesture Demo", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
