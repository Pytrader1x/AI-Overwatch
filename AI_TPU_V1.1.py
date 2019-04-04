import argparse
import platform
import numpy as np
import cv2
import time
from PIL import Image
from time import sleep
import multiprocessing as mp
from edgetpu.detection.engine import DetectionEngine
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import datetime
import os




lastresults = None
processes = []
frameBuffer = None
results = None
fps = ""
AI_fps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0
box_color = (255, 128, 0)
box_thickness = 1
label_background_color = (125, 175, 75)
label_text_color = (255, 255, 255)
percentage = 0.0

# Function to read labels from text files.
def ReadLabelFile(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret


def camThread(label, results, frameBuffer, camera_width, camera_height, vidfps, usbcamno):

    global fps
    global AI_fps
    global framecount
    global detectframecount
    global time1
    global time2
    global lastresults
    global cam
    global window_name
    

    cam = cv2.VideoCapture(usbcamno)
    cam.set(cv2.CAP_PROP_FPS, vidfps)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    window_name = "USB Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        t1 = time.perf_counter()

        ret, color_image = cam.read()
        if not ret:
            continue
        if frameBuffer.full():
            frameBuffer.get()
        frames = color_image
        frameBuffer.put(color_image.copy())
        res = None

        if not results.empty():
            res = results.get(False)
            detectframecount += 1
            imdraw = overlay_on_image(frames, res, label, camera_width, camera_height)
            lastresults = res
        else:
            imdraw = overlay_on_image(frames, lastresults, label, camera_width, camera_height)

        cv2.imshow('USB Camera', imdraw)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

        # FPS calculation
        framecount += 1
        if framecount >= 15:
            fps       = "(Playback) {:.1f} FPS".format(time1/15)
            AI_fps = "(AI_FPS) {:.1f} FPS".format(detectframecount/time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
        time2 += elapsedTime



def inferencer(results, frameBuffer, model, camera_width, camera_height):

    engine = DetectionEngine(model)

    while True:

        if frameBuffer.empty():
            continue

        # Run inference.
        color_image = frameBuffer.get()
        prepimg = color_image[:, :, ::-1].copy()
        prepimg = Image.fromarray(prepimg)

        tinf = time.perf_counter()
        ans = engine.DetectWithImage(prepimg, threshold=0.5, keep_aspect_ratio=True, relative_coord=False, top_k=10)
        print(time.perf_counter() - tinf, "sec")
        results.put(ans)



def overlay_on_image(frames, object_infos, label, camera_width, camera_height):
    
    color_image = frames

    if isinstance(object_infos, type(None)):
        return color_image
    img_cp = color_image.copy()

    for obj in object_infos:
        box = obj.bounding_box.flatten().tolist()
        box_left = int(box[0])
        box_top = int(box[1])
        box_right = int(box[2])
        box_bottom = int(box[3])
        cv2.rectangle(img_cp, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)
        time_pause = datetime.datetime.now()
        percentage = int(obj.score * 100)
        label_score_internal = label[obj.label_id]
        label_text = label[obj.label_id] + " (" + str(percentage) + "%)" 
        frame = overlay_on_image
        if percentage > 0.8:
            print("Sufficiently accurate")
            if label_score_internal == "person":
                if datetime.datetime.now() > time_pause:
                            print("EMAILING")
                            time_pause = datetime.datetime.now() + datetime.timedelta(minutes=2)
                            print (time_pause)
                            print("Found Human")
                            cv2.imwrite('me.JPG',frame)
                            email_user = 'usr@domain.com'
                            email_password = 'pswd'
                            email_send = 'usr@domain.com'

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
                            server = smtplib.SMTP('smtp.domain.com',123)
                            server.starttls()
                            server.login(email_user,email_password)

                                
                            server.sendmail(email_user,email_send,text)
                            server.quit()
                            print("email Sent")
                

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = box_left
        label_top = box_top - label_size[1]
        if (label_top < 1):
            label_top = 1
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]
        cv2.rectangle(img_cp, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1), label_background_color, -1)
        cv2.putText(img_cp, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    cv2.putText(img_cp, fps,       (camera_width-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
    cv2.putText(img_cp, AI_fps, (camera_width-170,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)

    return img_cp

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite", help="Path of the detection model.")
    parser.add_argument("--label", default="coco_labels.txt", help="Path of the labels file.")
    parser.add_argument("--usbcamno", type=int, default=0, help="USB Camera number.")
    args = parser.parse_args()

    model    = args.model
    label    = ReadLabelFile(args.label)
    usbcamno = args.usbcamno

    camera_width = 320
    camera_height = 240
    vidfps = 150

    try:
        mp.set_start_method('forkserver')
        frameBuffer = mp.Queue(10)
        results = mp.Queue()

        # Start streaming
        p = mp.Process(target=camThread,
                       args=(label, results, frameBuffer, camera_width, camera_height, vidfps, usbcamno),
                       daemon=True)
        p.start()
        processes.append(p)

        # Activation of inferencer
        p = mp.Process(target=inferencer,
                       args=(results, frameBuffer, model, camera_width, camera_height),
                       daemon=True)
        p.start()
        processes.append(p)

        while True:
            sleep(1)

    finally:
        for p in range(len(processes)):
            processes[p].terminate()
