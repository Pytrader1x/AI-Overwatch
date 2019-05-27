
import datetime
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
import edgetpu.classification.engine

import numpy as np
import cv2
import time

import time



import numpy as np
import cv2
import time




def capture_frame_send(target_frame):


    # capture_duration = 10

    # cap = cv2.VideoCapture(0)

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    # start_time = time.time()
    # while( int(time.time() - start_time) < capture_duration ):
    #     ret, frame = cap.read()
    #     if ret==True:
    #         frame = cv2.flip(frame,0)
    #         out.write(frame)
    #         cv2.imshow('frame',frame)
    #     else:
    #         break

    




    cv2.imwrite('me.JPG',target_frame)
    email_user = 'email@email.com'
    email_password = 'pswd'
    email_send = 'email@email.com'
    

    subject = 'AI detected Human'

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_send
    msg['Subject'] = subject

    body = 'AI Detected Human!'
    msg.attach(MIMEText(body,'plain'))
    #filename='output.avi'          
    filename='me.JPG'
    attachment  =open(filename,'rb')


    part = MIMEBase('application','octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition',"attachment; filename= "+filename)

    msg.attach(part)
    text = msg.as_string()
    server = smtplib.SMTP('smtp...',111)
    server.starttls()
    server.login(email_user,email_password)

    
    server.sendmail(email_user,email_send,text)
    server.quit()
    print("email Sent")
        
