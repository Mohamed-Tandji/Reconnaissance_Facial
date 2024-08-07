import cv2,numpy as np,face_recognition

#import Signatures
signatures_class=np.load('Face_Signatures_db.npy')
X=signatures_class[ : ,0:-1].astype('float')
Y=signatures_class[ : ,-1]

# Open camera
cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    if success:
        print("Capturing....")
        imgR=cv2.resize(img, (0, 0),None, fx=0.25, fy=0.25)
        imgR=cv2.cvtColor(imgR,cv2.COLOR_BGR2RGB)
        #find face locations from the camera
        facesCurrent=face_recognition.face_locations(imgR)
        
        # Get signatures from face
        encodesCurrent=face_recognition.face_encodings(imgR, facesCurrent)
        for encodeFace,faceloc in zip(encodesCurrent,facesCurrent):
            matches=face_recognition.compare_faces(X,encodeFace)
            faceDis=face_recognition.face_distance(X,encodeFace)
            matchIndex=np.argmin(faceDis)
            if matches[matchIndex]:
                name=Y[matchIndex].upper()
                y1,x2,y2,x1=faceloc
                y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img, (x1,y1), (x2,y2),(0,0,255),2)
                cv2.rectangle(img, (x1,y2-25), (x2,y2),(0,0,255),cv2.FILLED)
                cv2.putText(img, name,(x1+10,y2-20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0))
            else:
                name=Y[matchIndex].upper()
                y1,x2,y2,x1=faceloc
                y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img, (x1,y1), (x2,y2),(0,0,255),2)
                cv2.rectangle(img, (x1,y2-25), (x2,y2),(0,0,255),cv2.FILLED)
                cv2.putText(img, 'Unknown',(x1+10,y2-20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0))
                
    cv2.imshow('webcam',img)
    cv2.waitKey(1)

                