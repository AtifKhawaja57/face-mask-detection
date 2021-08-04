import argparse
import imutils
import time
import cv2
import os

from mask_detection import MaskDetector
from temp_sensor import TempSensor
from gpio_management import GPIO
from email_sender import EmailSender

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
                    default="models/face_detector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
                    default="models/mask_detector/mask_detector.model",
                    help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = ap.parse_args()

    print("[INFO] initialising GPIO and temp sensor...")
    io = GPIO()
    sensor = TempSensor()

    print("[INFO] initialising SMTP...")
    sender = EmailSender()

    print("[INFO] loading face and mask detector models...")
    prototxt_path = os.path.join(os.path.dirname(__file__), "..", args.face, "deploy.prototxt")
    weights_path = os.path.join(os.path.dirname(__file__), "..", args.face, "res10_300x300_ssd_iter_140000.caffemodel")
    mask_net = os.path.join(os.path.dirname(__file__), "..", args.model)
    mask_detector = MaskDetector(prototxt_path, weights_path, mask_net, args.confidence)
    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        _, frame = cap.read()
        frame = imutils.resize(frame, width=500)
        location, prediction = mask_detector.detect_mask(frame)
        temperature = sensor.temperature()
        try:
            (startX, startY, endX, endY) = location
        except ValueError:
            cv2.imshow("Face Mask Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        wearing_mask = prediction[0] > prediction[1]
        color = (0, 255, 0)
        if wearing_mask:
            cv2.putText(
                frame, "Thank you. Mask on.", (startX - 50, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                frame, "No Face Mask Detected!", (startX - 50, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            color = (0, 0, 255)

        if temperature > 37:
            cv2.putText(frame, f'Body Temp: {temperature}C ', (startX - 50, startY - 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, 'Body Temperature too High! ', (startX - 50, startY - 35),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            color = (0, 0, 255)
        elif temperature < 34:
            cv2.putText(frame, 'Move Closer To Camera ', (startX - 50, startY - 35),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f'Body Temp: {temperature}C ', (startX - 50, startY - 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, 'Body Temperature OK! ', (startX - 50, startY - 35),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        if color == (0, 0, 255):
            io.red()
        else:
            io.green()

        if not (wearing_mask or temperature <= 37):
            cwd = os.getcwd()
            try:
                os.chdir(os.path.join(os.path.dirname(__file__), "..", "no_mask"))
                os.chdir(cwd)
            except FileNotFoundError:
                os.mkdir(os.path.join(os.path.dirname(__file__), "..", "no_mask"))
            cv2.imwrite(os.path.join(os.path.dirname(__file__), "..", "no_mask",
                                     "".join([str(i) for i in frame[230, 250:290]])) + ".jpg", frame)
            sender.send_message(temperature)
        cv2.imshow("Face Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    #sensor.cleanup()
    cap.release()
    cv2.destroyAllWindows()
