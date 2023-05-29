import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)

while True:

    try:
        check, cameraFeedImg = cap.read()

        cameraFeedImg = cv2.flip(cameraFeedImg, 1)

        handsDetector = detector.findHands(cameraFeedImg, flipType=False)
        hands = handsDetector[0]
        cameraFeedImg = handsDetector[1]

        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            handType1 = hand1["type"]
            fingers1 = detector.fingersUp(hand1)

            currentFingerUp = ""

            if fingers1[0] == 0:
                currentFingerUp = "Thumb"
            elif fingers1[1] == 1:
                currentFingerUp = "Index Finger"
            elif fingers1[2] == 1:
                currentFingerUp = "Middle Finger"
            elif fingers1[3] == 1:
                currentFingerUp = "Ring Finger"
            elif fingers1[4] == 1:
                currentFingerUp = "Little Finger"
            else:
                currentFingerUp = ""

            cv2.putText(cameraFeedImg, handType1 + " : " + currentFingerUp, (75, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Check if handType1 is "Left"
            if handType1 == "Left":
                # Convert the color image to grayscale image
                cameraFeedImg = cv2.cvtColor(
                    cameraFeedImg, cv2.COLOR_BGR2GRAY)
            # Check if handType1 is "Right"
            if handType1 == "Right":
                # Convert to oil paint image
                cameraFeedImg = cv2.xphoto.oilPainting(
                    cameraFeedImg, size=7, dynRatio=1)

            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            handType2 = hand2["type"]
            fingers2 = detector.fingersUp(hand2)

            currentFingerUp = ""

            if fingers2[0] == 0:
                currentFingerUp = "Thumb"
            elif fingers2[1] == 1:
                currentFingerUp = "Index Finger"
            elif fingers2[2] == 1:
                currentFingerUp = "Middle Finger"
            elif fingers2[3] == 1:
                currentFingerUp = "Ring Finger"
            elif fingers2[4] == 1:
                currentFingerUp = "Little Finger"
            else:
                currentFingerUp = ""

            cv2.putText(cameraFeedImg, handType2 + " : " + currentFingerUp, (375, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Check if handType2 is "Left"
            if handType2 == "Left":
                # Convert the color image to grayscale image
                cameraFeedImg = cv2.cvtColor(
                    cameraFeedImg, cv2.COLOR_BGR2GRAY)
            # Check if handType2 is "Right"
            if handType2 == "Right":
                # Convert to oil paint image
                cameraFeedImg = cv2.xphoto.oilPainting(
                    cameraFeedImg, size=7, dynRatio=1)

    except Exception as e:
        print(e)

    cv2.imshow("Image", cameraFeedImg)
    cv2.waitKey(1)
