# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn
import face_recognition
from sklearn import svm
import os
import time
import pickle


# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
print("Started")
start = time.time()

encodings = []
names = []
model_save_path="trained_svn_model.clf"
# Training directory
train_dir = os.listdir('train_dir/')

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("train_dir/" + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file("train_dir/" + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        #If training image contains none or more than faces, print an error message and exit
        if len(face_bounding_boxes) != 1:
            print(person + "/" + person_img + " contains none or more than one faces and can't be used for training.")
            exit()
        else:
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)

if model_save_path is not None:
        with open(model_save_path, 'ab') as f:
            pickle.dump(clf, f)

end = time.time()
print(end - start)

print("end here")