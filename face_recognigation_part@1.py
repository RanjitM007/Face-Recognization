#DATA COLLECTION
import cv2
import numpy as np

#harcscard classifier
face_classifier=cv2.CascadeClassifier("C:/Users/Ranjit M/Desktop/pythonProject2/haarcascade_frontalface_default.xml")

#face function
def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=face_classifier.detectMultiScale(gray,1.5,5)
    if  face is ():
        return None
    for(x,y,w,h) in face:
        cropped_face=img[y:y+h,x:x+w]
    return cropped_face





camera=cv2.VideoCapture(0)
count=0
while True:
    ret,frame=camera.read()
    if face_extractor(frame) is not None:
        count+=1
        face_n=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face_n,cv2.COLOR_BGR2GRAY)
        file_name="C:/Users/Ranjit M/Desktop/pythonProject2/Face/Ranjit"+str(count)+".jpg"
        cv2.imwrite(file_name,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(2,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face Not Found")
        pass
    if cv2.waitKey(1)==13  or count==50:
        break
camera.release()
cv2.destroyAllWindows()
print("Collecting samples complete !!!!")
