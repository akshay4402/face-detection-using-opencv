
import cv2
import os
import pysondb;

database = pysondb.db.getDb("db.json")

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id \n')
name = input('\n enter user name \n')


database.add({"name":name,"face_id": face_id})

print("\n [INFO] Initializing face capture....")
# Initialize individual sampling face count
count = 0

while(True):
    ret, img = cam.read()
    # convert to gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    # loop saved face data
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        # show image
        cv2.imshow('image', img)
    # wait for image to capture
    k = cv2.waitKey(100) & 0xff 
    # to break after wait else it will keep recording
    if k == 27:
        break
    # increase efficency by increasing count condition
    elif count >= 100: 
         break;

# Do a bit of cleanup
print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()


