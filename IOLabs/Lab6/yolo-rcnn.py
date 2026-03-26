import cv2

pre_trained_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def find_face_in_image(image: str):
    bgr_image = cv2.imread(image)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = pre_trained_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3)

    for (x, y, z, h) in faces:
        cv2.rectangle(bgr_image, (x, y), (x + z, y + h), (255, 0, 0), 2)

    cv2.imshow("Detected faces", bgr_image)
    cv2.waitKey(0)


find_face_in_image("People.bmp")
find_face_in_image("People2.png")
find_face_in_image("People3.png")
find_face_in_image("People4.png")
find_face_in_image("People5.png")
