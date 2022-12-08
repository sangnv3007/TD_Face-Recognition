import face_recognition
import pickle
import numpy as np
import os

# Processing functions decode faces into file .dat
know_faces = './data/train'
encoded_faces = {}
# Mang chua thong tin khuon mat da ma hoa
for name in os.listdir(know_faces):
    for (i,filename) in enumerate(os.listdir(f'{know_faces}/{name}')):
        #Load images 
        images = face_recognition.load_image_file(f'{know_faces}/{name}/{filename}')
        #Indentify face location
        face_loc = face_recognition.face_locations(images)
        if(len(face_loc)>0):
            encoded_faces[name] = face_recognition.face_encodings(images, face_loc)[0]
        print("[INFO] processing image in folders {} {}/{}".format(name,i + 1,len(os.listdir(f'{know_faces}/{name}'))))
with open('./model/train.dat', 'wb') as file:
    #save file encode faces to file .dat 
    pickle.dump(encoded_faces, file)            
    print('[INFO] Encoded successful !')

### Read file .dat and preciction faces

# unknow_faces = './data/test'
# # Open file .dat
# with open('./model/train.dat', 'rb') as file:
#     encoded_faces = pickle.load(file)
#     print('Read file successful !')
# face_names = list(encoded_faces.keys())
# #Get values encoded_faces
# face_encodings = np.array(list(encoded_faces.values()))
# print(list(encoded_faces.values()))
# for name in os.listdir(unknow_faces):
#     for filename in os.listdir(f'{unknow_faces}/{name}'):
#         #load images
#         images = face_recognition.load_image_file(f'{unknow_faces}/{name}/{filename}')
#         #Identify face location
#         face_loc = face_recognition.face_locations(images)
#         test_ecodings = face_recognition.face_encodings(images, face_loc)
#         #Test image in folder test
#         for i in range(0, len(test_ecodings)):
#             for j in range(0, len(face_encodings)):
#                 face_distance = face_recognition.face_distance(face_encodings[j],test_ecodings[i])
#                 res = face_recognition.compare_faces(face_encodings[j], test_ecodings[i], tolerance=0.45)
#                 if True in res: 
#                     print("Image : {0}, Prediction => {1} )".format(filename,face_names[j]))  
#                     print(face_distance)          
#                     break