# import cv2

# face_cascade=cv2.CascadeClassifier('./data-haarcascades-haarcascade_frontalface_default.xml')
# webcam=cv2.VideoCapture(0)
# while True:
#     _,img=webcam.read()
    
#     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray,1.5,4)
#     for(x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
#     cv2.imshow("face detection",img)
#     key = cv2.waitKey(10)
#     if key==27:
#         break
# webcam.release()
# cv2.destroyAllWindows()

import cv2

# Ensure correct path to the cascade classifier XML file
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default (1).xml')

# Check if the cascade classifier file was loaded successfully
if face_cascade.empty():
    print("Error loading cascade classifier file.")
    exit()


# Access webcam
webcam = cv2.VideoCapture(0)

while True:
    _, img = webcam.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces with adjusted parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Display the image
    cv2.imshow("Face Detection", img)

    # Exit on 'Esc' key press
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
