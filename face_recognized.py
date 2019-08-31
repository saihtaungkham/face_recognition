from pathlib import Path
from imutils import paths
import face_recognition
import os
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument(
    "--registered_faces_dir", 
    required=True, 
    type=str,
    help="Specify the directory for registered faces."
)
ap.add_argument(
    "--recognize_image", 
    type=str,
    help="Specify the directory of an image to be recognized."
)
args = vars(ap.parse_args())

def anayze_faces(args):

    # Initialize the variables for operation.
    known_face_encodings = []
    known_face_names = []
    is_video = True
    face_names = []
    process_this_frame = True

    faces_path = Path(args["registered_faces_dir"])

    # Check whether the registered_faces_dir exist
    if not os.path.exists(faces_path):
        print("Invalid registered_faces_dir")
        return

    # Check whether it is an image input or video stream.
    if args["recognize_image"]:
        recognize_image = Path(args["recognize_image"])
        is_video = False

    # Get all the registered faces.
    face_files = paths.list_images(faces_path)


    # Read image and encode it.
    for face_file in face_files:
        print("Process the file :{}".format(face_file))
        name = face_file.split(os.path.sep)[-1].split('.')[-2]
        image = face_recognition.load_image_file(face_file)
        known_face_names.append(name)
        known_face_encodings.append(face_recognition.face_encodings(image)[0])

    # Loop until the user terminate
    while True:

        if is_video:
            # Get a reference to a webcame index as 0.
            # Change parameter as required.
            video_capture = cv2.VideoCapture(0)
            # Grab a single frame of video
            _, frame = video_capture.read()
        else:
            frame = cv2.imread(str(recognize_image))

        if frame is None:
            print("Invalid Image.")
            break

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                # Tune the tolerance parameters as desired
                # Based on the tolerance ratio, we check whether the input
                # face is similar to which registered faces.
                matches = face_recognition.compare_faces(
                    known_face_encodings, 
                    face_encoding,
                    tolerance=0.5
                    )
                name = "UNKNOWN"

                # Find the face that look the most similar to the registered faces.
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                similar_rate = (1 - face_distances[best_match_index]) * 100

                # Check if the detected face is a known face
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append((name, similar_rate))

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), (name, similar_rate) in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if name == "UNKNOWN":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX

            # Draw the text on the frame
            cv2.putText(frame, "{:.2f}%".format(similar_rate), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 0), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        if not is_video:
            # Wait for any key to be pressed.
            cv2.waitKey(0)

            # Save the detected image to the same directory with suffix _detected
            new_filename = recognize_image.stem + "_detected.jpg"
            new_filepath = recognize_image.parents[0]
            cv2.imwrite(str(new_filepath/new_filename), frame)
            
            # Exit the loop as it is not a video stream.
            break

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if is_video:
        # Release handle to the webcam
        video_capture.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    anayze_faces(args)