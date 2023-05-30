from flask import Flask,jsonify,request,render_template, Response
from utils import read_image, prepare_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    # let blow through model and detect face
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))


    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize the corresponding list faces, locations, and predicts from model
    faces = []
    locs = []
    preds = []

    # loop through the detections
    for i in range(0, detections.shape[2]):
        # get the corresponding confidence (probability,...) of each detection
        confidence = detections[0, 0, i, 2]

        # filter out detections that ensure reliability > confidence threshold
        if confidence > args["confidence"]:
            # calculate (x,y) bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # make sure the bounding box is within the frame size
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract face ROI, convert image from BGR to RGB, resize to 224x224 and preprocess
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the corresponding face and bounding box to the lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only predict if less detects at least 1 face
    if len(faces) > 0:
        # To make it faster, predict on all faces instead of predicting on each face using a loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return 2-tuple containing their respective locations and predict
    return (locs, preds)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# input parameters
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load face detector model from folder
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load face mask detector model from train
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])


def gen():
    # initiate video stream and enable webcam
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop through frames from video stream
    while True:
        # crop frame from video and resize to max width 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        # detect faces in the frame and determine as mask or no mask
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # iterate over the detected face bounding boxes and their corresponding predict iterates over the detected face bounding boxes and their corresponding predict
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # define class label and color to draw bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # attach more information about the probability of label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # rectangular display label and bounding box on output framedisplay label and rectangular bounding box on output frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show output frame
        cv2.imshow("FACE_MASK_DETECTOR_TL", frame)
        key = cv2.waitKey(1) & 0xFF

        # press q to exit
        if key == ord("q"):
            break
    # cleanup
    cv2.destroyAllWindows()
    vs.stop()

@app.route('/webcam')
def webcam():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():

    # input parameters
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
                    default="face_detector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
                    default="mask_detector.model",
                    help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load face detector model from folder
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load face mask detector model 
    print("[INFO] loading face mask detector model...")
    model = load_model(args["model"])

    # load input image preprocess
    file = request.files['image']
    # Read image
    image = read_image(file)

    orig = image.copy()
    (h, w) = image.shape[:2]

    # convert image to blob
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop through the detections
    for i in range(0, detections.shape[2]):
        # get the corresponding confidence (probability,...) of each detection
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
       
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # make sure the bounding box is within the frame size
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract face ROI, convert image from BGR to RGB, resize to 224x224 and preprocess
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Use the trained model to predict mask or no mask
            (mask, withoutMask) = model.predict(face)[0]

            
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    # Prepare image for html
    to_send = prepare_image(image)

    return render_template('index.html', image_to_show=to_send,
                           init=True)
if __name__ == '__main__':
    app.run(debug=True)
