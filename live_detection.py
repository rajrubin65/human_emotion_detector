import cv2 as cv
from prediction import EmotionDetectEnginee



capture = cv.VideoCapture(0)
org = (50, 185)
font=cv.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 0, 255)
thickness = 4


while True:
    is_true,frame = capture.read()
    cv.imwrite('image.png',frame)
    result = EmotionDetectEnginee(image_path='image.png').prediction()
    cv.putText(frame,result,org, font,fontScale, color, thickness, cv.LINE_AA)
    cv.imshow('Emotion Detector',frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()