from flask import Flask, jsonify, render_template, request
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

import numpy as np
import math
app = Flask(__name__)
@app.route('/')
def index():
    return render_template("SIAN/index.html")
# def check(img):
#     cv2.destroyWindows("img")
#     return render_template("Check.html")

@app.route('/test',methods = ['GET'])
def predict():
    if request.method == 'GET':
        cap = cv2.VideoCapture(0)
        detect = HandDetector(maxHands=1)
        classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
        ofs = 20
        size = 300
        c = 0
        folder = "Data/C"
        labels = ["A", "B", "C"]
        while True:
            frame, img = cap.read()
            imgoutput = img.copy()
            hands, img = detect.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgcrop = img[y - ofs:y + h + ofs, x - ofs:x + w + ofs]
                imgcropshape = imgcrop.shape

                imgwhite = np.ones((size, size, 3), np.uint8) * 255
                ratio = h / w
                if ratio > 1:
                    k = size / h
                    wcal = math.ceil(k * w)
                    imgresize = cv2.resize(imgcrop, (wcal, size))
                    imgresizeshape = imgresize.shape
                    wgap = math.ceil((size - wcal) / 2)
                    imgwhite[:, wgap:wcal + wgap] = imgresize
                    prediction, index = classifier.getPrediction(imgwhite, draw=False)
                    print(prediction, index)
                else:
                    k = size / w
                    hcal = math.ceil(k * h)
                    imgresize = cv2.resize(imgcrop, (size, hcal))
                    imgresizeshape = imgresize.shape
                    hgap = math.ceil((size - hcal) / 2)
                    imgwhite[hgap:hcal + hgap, :] = imgresize
                    prediction, index = classifier.getPrediction(imgwhite, draw=False)
                    print(prediction, index)
                cv2.rectangle(imgoutput, (x - ofs, y - ofs - 50), (x - ofs + 90, y - ofs), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgoutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 0), 2)
                cv2.rectangle(imgoutput, (x - ofs, y - ofs), (x + w + ofs, y + h + ofs), (255, 0, 255), 4)
                # cv2.imshow("Crop", imgcrop)
                # cv2.imshow("White", imgwhite)
            cv2.imshow("Image", imgoutput)
            if cv2.waitKey(1) & 0xFF == ord('c') :
                cv2.destroyWindow("Image")
                return render_template("SIAN/check.html")


if __name__ == "__main__":
    app.run(debug=True)