import cv2

yuz_tanimlari = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
kamera = cv2.VideoCapture(0)

while True:
    _, pencere = kamera.read()
    c = cv2.cvtColor(pencere, cv2.COLOR_BGR2GRAY)
    yuzler = yuz_tanimlari.detectMultiScale(c, 1.1, 4)
    for (x, y, genislik, yukseklik) in yuzler:
        cv2.rectangle(pencere, (x, y), (x+genislik, y+yukseklik), (0, 255, 0), 2)
    cv2.imshow('Kamera', pencere)
    if (cv2.waitKey(30) & 0xff == 27):
        break

kamera.release()
cv2.destroyAllWindows()
