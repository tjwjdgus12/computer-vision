import cv2
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from helper import help

app = Flask(__name__)

@app.route('/')
def upload_render():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file_bytes = np.asarray(bytearray(file.stream.read()), dtype="uint8")
        file_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result = help(file_img)
        cv2.imwrite('static/result.png', result)
        return render_template('result.html')
    
app.run(debug=True)