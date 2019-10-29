# source
# https://github.com/ageitgey/face_recognition

import face_recognition
import numpy as np
from PIL import Image, ImageDraw

from os import listdir
from os.path import isfile, join


# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
my_image = face_recognition.load_image_file("./faces/Adam.jpg")
my_face_encoding = face_recognition.face_encodings(my_image)[0]

barack_image = face_recognition.load_image_file('./faces/Barack.jpg')
barack_face_encoding = face_recognition.face_encodings(barack_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    my_face_encoding,
    barack_face_encoding
]
known_face_names = [
    "Adam",
    "Barack"
]
print('Learned encoding for', len(known_face_encodings), 'images.')

# get the list of all filenames
guess_dir = './faces/guessing/'
filenames = [f for f in listdir(guess_dir) if isfile(join(guess_dir, f))]

for f in filenames:
    print(f)
    # Load an image with an unknown face
    unknown_image = face_recognition.load_image_file(guess_dir+f)

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face using the Pillow module
        color = (255,0,0)
        draw.rectangle(((left, top), (right, bottom)), outline=(color))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(color), outline=(color))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    # pil_image.show()
    pil_image.save('./faces/out/'+f)