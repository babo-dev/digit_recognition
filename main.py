from flask import Flask, render_template, url_for, request
from keras.models import load_model
import base64
import numpy as np
import cv2


init_Base64 = 21

app = Flask(__name__, template_folder='templates')

model = load_model('mnist.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/draw')
def draw():
    return render_template("draw.html")


@app.route('/digits')
def digits():
    return render_template("digits.html")


@app.route('/predict', methods=['POST'])
def predict():
    final_pred = None
    if request.method == 'POST':
        draw = request.form['url']
        draw = draw[init_Base64:]
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        _, threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        i = 0

        for contour in contours:

            if i == 0:
                i = 1
                continue

            area = cv2.contourArea(contour)
            if area > 1000:
                approx = cv2.approxPolyDP(
                    contour, 0.03 * cv2.arcLength(contour, True), True)

                # cv2.imwrite('threshold_image.png', threshold)
                # x, y, w, h = cv2.boundingRect(approx)
                # cv2.rectangle(threshold, (x,y), (x+w, y+h), (0,  255, 0), 5)

                if len(approx) == 3:
                    final_pred = "Üçburçluk"

                elif len(approx) == 4:
                    final_pred = "Dörtburçluk"

                elif len(approx) == 5:
                    final_pred = "Bäşburçluk"

                elif len(approx) == 6:
                    final_pred = "Atlyburçluk"

                else:
                    final_pred = "Tegelek"

    return render_template('results.html', prediction =final_pred)


@app.route('/detect_digit', methods=['POST'])
def detect_digit():
    final_pred = None
    if request.method == 'POST':
        digit = request.form['url']
        digit = digit[init_Base64:]
        draw_decoded = base64.b64decode(digit)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (28, 28))
        cv2.imwrite('image.png', image)
        image = image.astype('float32') / 255.0

        img2 = np.expand_dims(image, axis=(0, -1))

        prediction = model.predict(img2)
        final_pred = str(np.argmax(prediction))

    return render_template('digit_result.html', prediction=final_pred)


if __name__ == '__main__':
    app.run(debug=True)
