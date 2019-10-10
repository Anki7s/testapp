# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn
import face_recognition
from sklearn import svm
import os
import time
import pickle


# Training the SVC classifier

print("Started")
start = time.time()

model_save_path="trained_svn_model.clf"

with open(model_save_path, 'rb') as f:
    svm_clf = pickle.load(f)
# Load the test image with unknown faces into a numpy array
test_image = face_recognition.load_image_file('ragini.jpg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)
end = time.time()
print(end - start)

# Predict all the faces in the test image using the trained classifier
print("Found:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = svm_clf.predict([test_image_enc])
    print(*name)
