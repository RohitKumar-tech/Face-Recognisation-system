import cv2
import numpy as np
import face_recognition

# function
def resize(img,size):
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimension = (width,height)
    return cv2.resize(img,dimension, interpolation=cv2.INTER_AREA)

# Load images
rohit = face_recognition.load_image_file('sample_images/rohit.jpeg')
rohit_test = face_recognition.load_image_file('sample_images/rohit_test.jpeg')

# Convert images to RGB (face_recognition library requires RGB format)
rohit = cv2.cvtColor(rohit, cv2.COLOR_BGR2RGB)
rohit_test = cv2.cvtColor(rohit_test, cv2.COLOR_BGR2RGB)

# resizing the image
rohit = resize(rohit,0.50)
rohit_test = resize(rohit_test,0.50)

# finding face location
# as this is just an image give it zero
faceLocation_rohit = face_recognition.face_locations(rohit)[0]
faceLocation_rohit_test = face_recognition.face_locations(rohit_test)[0]

# encoding the face and making the rectangle around the face
encode_rohit = face_recognition.face_encodings(rohit)[0]
encode_rohit_test = face_recognition.face_encodings(rohit)[0]
cv2.rectangle(rohit , (faceLocation_rohit[3],faceLocation_rohit[0]),(faceLocation_rohit[1],faceLocation_rohit[2]),(255,0,255),3)
cv2.rectangle(rohit_test , (faceLocation_rohit_test[3],faceLocation_rohit_test[0]),(faceLocation_rohit_test[1],faceLocation_rohit_test[2]),(255,0,255),3)

results = face_recognition.compare_faces([encode_rohit],encode_rohit_test)
print(results)
cv2.putText(rohit_test , f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX , 1 ,(0,0,255) ,2)

# Display images
cv2.imshow('main_img', rohit)
cv2.imshow('test_img1', rohit_test)
cv2.waitKey(0)
cv2.destroyAllWindows()