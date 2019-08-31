# Simple Face Recognition

The python script for face recognition using dlib and face_recognition package.

Check out the full blog [here.](https://medium.com/@saihtaungkham)

Under your python virutal environment, run the following
command to install the required packages.

```shell
pip install -r requirements.txt
```

# Running the Program
```
python face_recognized.py --registered_faces_dir registered_faces --recognize_image recognize_image/aung-san-suu-kyi.jpg

−−registered_faces_dir The directory path for registered faces.

--recognize_image A single image file path for recognition.

Note: Without --recognize_image, it will run and detect as video a stream.
```

Press q key to terminat the video stream.
For image screen, press any key for termination.

## Credit
https://github.com/ageitgey/face_recognition