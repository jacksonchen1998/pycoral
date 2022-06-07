from cv2 import *

cam_port = 1
cam = VideoCapture(cam_port)

result, image = cam.read()

if result:
    imwrite("coral_src1.jpg", image)
else:
    print("No image")