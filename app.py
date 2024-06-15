from flask import Flask,request,jsonify
import cv2
import numpy as np
import base64
import cv2
# from keras.models import model_from_json
import tensorflow as tf
import numpy as np
from flask_cors import CORS
# from keras_preprocessing.image import load_img
# json_file = open("facialemotionmodel.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)

model = tf.keras.models.load_model("facialemotionmodel.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

app = Flask(__name__)
CORS(app,origins='*')

@app.route('/')
def home():
    return "Hello World! & Server is Up :-)"

labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
@app.route('/predictemotion',methods=['POST'])
def predictemotion():
    data = request.get_json()
    if data is None:
        return jsonify({"error":"invalid request"}) , 400
    
    string_image = data['image']
    string_image = string_image.split(',')
    if len(string_image) == 2:
        string_image = string_image[1]
    else:
        string_image = string_image[0]
    
    string_image = base64.b64decode(string_image)
    nparr_image = np.frombuffer(string_image,np.uint8)
    img = cv2.imdecode(nparr_image,cv2.IMREAD_GRAYSCALE)
    faces=face_cascade.detectMultiScale(img,1.3,5)
    for (p,q,r,s) in faces:
        image = img[q:q+s,p:p+r]
        image = cv2.resize(image,(48,48))
        im = extract_features(image)
        pred = model.predict(im)
        prediction_label = labels[pred.argmax()]
        return jsonify({"emotion":prediction_label,"x1":str(p),"y1":str(q),"x2":str(r),"y2":str(s)}) , 200
    
    return jsonify({"emotion": "face not detected","x1":str(0),"y1":str(0),"x2":str(0),"y2":str(0)}) , 200

if __name__ == "__main__":
    app.run()