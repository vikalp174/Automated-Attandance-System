from cv2 import cv2
import numpy as np
import face_recognition
import os
import markAttendence as MA
path = "images"

images = []
classNames = []
myList = os.listdir(path)


for cls in myList:
    curImg = cv2.imread(f"{path}/{cls}")
    images.append(curImg)
    classNames.append(cls.split(".")[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)

print("Encoding completed.....")

cap = cv2.VideoCapture(0)
x = 25
while x:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurrentFrameLoc = face_recognition.face_locations(imgS)
    encodeingsCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrameLoc)

    for encodeFace, faceLoc in zip(encodeingsCurrentFrame, faceCurrentFrameLoc):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (205, 100, 205), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (205, 100, 205), cv2.FILLED)
            cv2.putText( img,name,(x1 + 6, y2 - 6),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),2,)
            MA.markAttendence(name)

    cv2.imshow("webcam", img)
    cv2.waitKey(1)
    x = x - 1
