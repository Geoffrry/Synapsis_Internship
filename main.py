# Import Library
from flask import Flask, jsonify, render_template, request
from os.path import join, dirname, realpath
import os
import base64, cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)
app.config['DEBUG'] = True

# Load the model
model = load_model('model_MobileNet.model')

# Configure the upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index_data():
    return render_template('input.html')

@app.route('/data', methods = ['POST'])
def uploadFiles():
    uploaded_file = request.files['file']
    if uploaded_file.filename !=  '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
    
    with open(file_path) as f:
        b64 = f.readline()

    
    decoded_data = base64.b64decode(b64)
    np_data = np.frombuffer(decoded_data,np.uint8)
    test_img = cv2.imdecode(np_data,cv2.IMREAD_COLOR)

    feature = cv2.resize(test_img ,(128, 128))

    test_feature = np.array(feature)/255.0
    test_feature = np.reshape(test_feature, (1, 128, 128, 3))

    label_dict = {'incorrect_mask': 0, 'without_mask': 1, 'with_mask': 2}
    result= model.predict(test_feature)
    label= np.argmax(result,axis=1)[0]
    classification = list(label_dict.keys())[label]


    json_response= {
        'status_code': 200,
        'description': 'Face Mask Classification',
        'data' : classification,
    }
    return jsonify(json_response)

if (__name__ == '__main__'):
    app.run(port = 5000)