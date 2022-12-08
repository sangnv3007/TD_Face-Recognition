import face_recognition
import cv2
import numpy as np
import os
know_faces = './data/train'
#Check if exist folder
input_name = ''
name_folder = ''
while True:
    try:
        input_name = str(input('Enter your name:').replace(' ', ''))
        name_folder = know_faces+'\\'+input_name
        if(os.path.isdir(name_folder)):
            print("[INFO] Folder already exists...")
            continue
    except ValueError:
        print("[INFO] Please enter a valid string")
        continue
    else:
        os.mkdir(name_folder)
        print(f'[INFO] Create folder {input_name}, Waitting camera start...')
        break
video_capture = cv2.VideoCapture(0)
img_counter = 1
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    # # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    # # Loop through each face in this frame of video
    for (top, right, bottom, left) in face_locations:
        img_name = "{}_{}.jpg".format(input_name,img_counter)
        cv2.imwrite(name_folder+'\\'+img_name, frame)
        img_counter += 1
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    if(img_counter > 20): 
        print('[INFO] Get new data for {} successful'.format(input_name))
        break
    cv2.imshow('Get Data Face', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()