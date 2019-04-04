import face_recognition
import os
import numpy as np
INPUT_DIR_PATH = "./input"
OUTPUT_DIR_PATH = "./output"

# Fill an array with paths of all images of a folder
image_paths = []
for image in os.listdir(INPUT_DIR_PATH):
    if image.endswith(".jpg") or image.endswith(".png"):
        image_paths.append(os.path.join(INPUT_DIR_PATH,image))


for image_path in image_paths:
    # Load the image
    image = face_recognition.load_image_file(image_path)
    # Compute the face encodings
    # WHY [0]?
    face_encoding = face_recognition.face_encodings(image)[0]

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    np.savetxt(
        os.path.join(OUTPUT_DIR_PATH,image_name+".csv"),
        face_encoding,
        fmt='%10.20f'
        )
