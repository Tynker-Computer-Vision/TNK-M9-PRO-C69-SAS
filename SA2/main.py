import cv2
# import HandDector library from cvzone
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Creating object to detect hand
detector = HandDetector(detectionCon=0.8)


while True:

    try:
        check, cameraFeedImg = cap.read()

        cameraFeedImg = cv2.flip(cameraFeedImg, 1)

        # Detect hand in cameraFeedImg
        handsDetector = detector.findHands(cameraFeedImg, flipType=False)
        hands = handsDetector[0]
        cameraFeedImg = handsDetector[1]

        if hands:
            hand1 = hands[0]
            # Get the landmarks from the detected hand
            lmList1 = hand1["lmList"]
            # Get the hand type from the detected hand ( Left and Right )
            handType1 = hand1["type"]

    except Exception as e:
        print(e)

    cv2.imshow("Image", cameraFeedImg)
    cv2.waitKey(1)
