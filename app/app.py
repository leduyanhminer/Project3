from flask import Flask, render_template, request
import sys
import os
from PIL import Image
from io import BytesIO
import requests
import base64
from predict import predict_image

app = Flask(__name__)

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")  # Hoặc format khác tùy thuộc vào hình ảnh
    return base64.b64encode(buffered.getvalue()).decode()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_data = None  # Biến này sẽ lưu trữ chuỗi base64 của hình ảnh

    if request.method == "POST":
        if "image" in request.files:
            file = request.files["image"]
            if file.filename != "":
                image = Image.open(file.stream)
                prediction = predict_image(image)
                image_data = image_to_base64(image)

        elif "url" in request.form:
            url = request.form["url"]
            response = requests.get(url)
            image_stream = BytesIO(response.content)
            image = Image.open(image_stream)
            prediction = predict_image(image)
            image_data = image_to_base64(image)

    return render_template('index.html', prediction=prediction, image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)
