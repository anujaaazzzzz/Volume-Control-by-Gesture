# open cv is the wrapper package used for object detection and video processing
# mediapipe provides pre-trained ml models for hand detection
# pyautogui allows python to automate the mouse and keyboard tasks
import cv2
import mediapipe as mp
import pyautogui

# initialising variables to store coordinates of hand landmarks
x1 = y1 = x2 = y2 = 0

# initialising web camera capture
# 0 indicates default camera
webcam = cv2.VideoCapture(0)

# creating object to track hands
my_hands = mp.solutions.hands.Hands()

# utility for drawing landmarks of the hand
drawing_utils = mp.solutions.drawing_utils

# loop starts
while True:
    # reads a frame from web camera
    _, image = webcam.read()
    # flips the camera for mirror effect
    image = cv2.flip(image, 1)
    frame_height, frame_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # process the frame to detect hands
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            # drawing the landmarks in the hand
            drawing_utils.draw_landmarks(image, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                # retrieving landmarks coordinates
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:
                    # marking the tip of the thumb having landmark id = 8
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3)
                    # storing their coordinates
                    x1 = x
                    y1 = y
                if id == 4:
                    # marking the tip of the index finger having landmark id = 4
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3)
                    x2 = x
                    y2 = y

        # calculating the distance between thumb and index finger
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # adjusting the volume up or down on the basis of the distance
        # must be greater than 50 pixels
        if dist > 30:
            pyautogui.press("volumeup")
        else:
            pyautogui.press("volumedown")

    # displaying the hand with landmarks
    cv2.imshow("Hand Gesture Volume Control", image)

    # checking if there is any key press to exit the loop
    # key code 27 refers to the ESC key
    key = cv2.waitKey(10)
    if key == 27:
        break

# releasing web camera and closing all the windows
webcam.release()
cv2.destroyAllWindows()
