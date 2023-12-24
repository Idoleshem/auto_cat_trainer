import cv2
import time
from ultralytics import YOLO
model = YOLO("yolov8n.pt")



# Run inference 
results = model('path/to/bus.jpg')


cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out_video = cv2.VideoWriter('cat_monitoring.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width,frame_height))

# training approachs
# (1) - asking politely
# (2) - try to draw attention by making cat noises
# (3) - try to draw attention by displaying an image of a food bag
# (4) - try to scary my cat with the sound of a puppy barking

# measure time until this training approach is successful
# save the results in a file and load it each time the program is run

start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

# /Users/idoleshem/Desktop/auto cat training
# docker run -it --name test_auto_cat_trainer -v /Users/idoleshem/Desktop/auto_cat_training :/project auto_cat_trainer