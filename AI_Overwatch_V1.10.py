

# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import time


# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
#IM_WIDTH = 640    #Use smaller resolution for
#IM_HEIGHT = 480   #slightly faster framerate

detected = 0
global detected
# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Define inside box coordinates (top left and bottom right)
TL_inside = (int(IM_WIDTH*0.1),int(IM_HEIGHT*0.35))
BR_inside = (int(IM_WIDTH*0.45),int(IM_HEIGHT-5))

# Define outside box coordinates (top left and bottom right)
TL_outside = (int(IM_WIDTH*0.46),int(IM_HEIGHT*0.25))
BR_outside = (int(IM_WIDTH*0.8),int(IM_HEIGHT*.85))

# Initialize control variables used for pet detector
detected_inside = False
detected_outside = False

inside_counter = 0
outside_counter = 0

pause = 0
pause_counter = 0

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.

### Picamera ###
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)
    camera.rotation = 180

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)
            

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        yes = 1
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        #NOW EMAIL ME
        # boxes[0][0] variable holds coordinates of detected objects as (ymin, xmin, ymax, xmax)
        if (((int(classes[0][0]) == 17) or (int(classes[0][0]) == 1))):
            
            detected = detected + 1
            if detected > 2:
                if np.any((scores) > 0.5):
                    
                    print("Emailing")
                    if ((int(classes[0][0]) == 17)):
                        print("Found Cat")
                        cv2.imwrite('me.JPG',frame)
                        email_user = 'willukmail@gmail.com'
                        email_password = 'ygczgasrzjodelkl'
                        email_send = 'willukmail@gmail.com'

                        subject = 'AI detected Cat'

                        msg = MIMEMultipart()
                        msg['From'] = email_user
                        msg['To'] = email_send
                        msg['Subject'] = subject

                        body = 'AI Detected Cat!'
                        msg.attach(MIMEText(body,'plain'))

                        filename='me.JPG'
                        attachment  =open(filename,'rb')

                        part = MIMEBase('application','octet-stream')
                        part.set_payload((attachment).read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition',"attachment; filename= "+filename)

                        msg.attach(part)
                        text = msg.as_string()
                        server = smtplib.SMTP('smtp.gmail.com',587)
                        server.starttls()
                        server.login(email_user,email_password)

                        
                        server.sendmail(email_user,email_send,text)
                        server.quit()
                        print("email Sent")
                        detected = 0
                            
                    elif ((int(classes[0][0]) == 1)):
                           print("Found Human")
                           cv2.imwrite('me.JPG',frame)
                           email_user = 'willukmail@gmail.com'
                           email_password = 'ygczgasrzjodelkl'
                           email_send = 'willukmail@gmail.com'

                           subject = 'AI detected Human'

                           msg = MIMEMultipart()
                           msg['From'] = email_user
                           msg['To'] = email_send
                           msg['Subject'] = subject

                           body = 'AI Detected Human!'
                           msg.attach(MIMEText(body,'plain'))

                           filename='me.JPG'
                           attachment  =open(filename,'rb')

                           part = MIMEBase('application','octet-stream')
                           part.set_payload((attachment).read())
                           encoders.encode_base64(part)
                           part.add_header('Content-Disposition',"attachment; filename= "+filename)

                           msg.attach(part)
                           text = msg.as_string()
                           server = smtplib.SMTP('smtp.gmail.com',587)
                           server.starttls()
                           server.login(email_user,email_password)

                            
                           server.sendmail(email_user,email_send,text)
                           server.quit()
                           print("email Sent")
                           detected = 0
                    
            
            
            
        
        
        
        
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

### USB webcam ###
elif camera_type == 'usb':
    # Initialize USB webcam feed
    camera = cv2.VideoCapture(0)
    ret = camera.set(3,IM_WIDTH)
    ret = camera.set(4,IM_HEIGHT)

    while(True):

        t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = camera.read()
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.85)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
        #NOW EMAIL ME
        # boxes[0][0] variable holds coordinates of detected objects as (ymin, xmin, ymax, xmax)
        if (((int(classes[0][0]) == 17) or (int(classes[0][0] == 18) or (int(classes[0][0]) == 88))) and (pause == 0)):
            x = int(((boxes[0][0][1]+boxes[0][0][3])/2)*IM_WIDTH)
            y = int(((boxes[0][0][0]+boxes[0][0][2])/2)*IM_HEIGHT)

            # Draw a circle at center of object
            cv2.circle(frame,(x,y), 5, (75,13,180), -1)

            # If object is in inside box, increment inside counter variable
            if ((x > TL_inside[0]) and (x < BR_inside[0]) and (y > TL_inside[1]) and (y < BR_inside[1])):
                inside_counter = inside_counter + 1

            # If object is in outside box, increment outside counter variable
            if ((x > TL_outside[0]) and (x < BR_outside[0]) and (y > TL_outside[1]) and (y < BR_outside[1])):
                outside_counter = outside_counter + 1

        # If pet has been detected inside for more than 10 frames, set detected_inside flag
        # and send a text to the phone.
        if inside_counter > 10:
            detected_inside = True
            print("Found")

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()

cv2.destroyAllWindows()

