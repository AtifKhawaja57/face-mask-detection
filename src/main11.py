from gpiozero import Buzzer, LED

# import the necessary packages

from tensorflow.keras.models import load_model
import argparse
import imutils
import time
import cv2
import os

from detect_mask import detect_and_predict_mask
from detect_temperature import TempSensor

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
                    default="models/face_detector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
                    default="models/mask_detector/mask_detector.model",
                    help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    print("[INFO] loading face detector model...")

    prototxtPath = os.path.join(os.path.dirname(__file__), "..", args["face"], "deploy.prototxt")
    weightsPath = os.path.join(os.path.dirname(__file__), "..", args["face"], "res10_300x300_ssd_iter_140000.caffemodel")
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args["model"])

    sensor = TempSensor()

    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(0)
    # vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)

    buzzer = Buzzer(21)
    red = LED(14)
    green = LED(15)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        _, frame = cap.read()
        frame = imutils.resize(frame, width=500)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        locations, predictions = detect_and_predict_mask(frame, faceNet, maskNet, args["confidence"])

        # loop over the detected face locations and their corresponding
        # locations
        for box, prediction in zip(locations, predictions):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = prediction

            frame, t = sensor.temperature(frame, (startX + endX) / 2, (startY + endY) / 2)

            # determine the class label and color we'll use to draw
            # the bounding box and text
            if mask > withoutMask:
                label = "Thank You. Mask On."
                color = (0, 255, 0)
                if t:
                    buzzer.off()
                    red.off()
                    green.on()
                else:
                    buzzer.on()
                    green.off()
                    red.on()
            else:
                label = "No Face Mask Detected"
                color = (0, 0, 255)
                buzzer.on()
                green.off()
                red.on()

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX - 50, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Face Mask Detector", frame)

        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # do a bit of cleanup
    cap.release()
    cv2.destroyAllWindows()
