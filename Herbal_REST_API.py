from json import load

from flask import Flask, request, json, jsonify
import time 
import base64
import cv2
import numpy as np

from PlantClassificationModels.EnsemblePredictor import EnsemblePredictor 
from NLPModel.NLPPredictor import NLPPredictor 
from SpeechToText.SpeechToTextConverter import SpeechToTextConverter


app = Flask(__name__)

DEPLOY_MODELS = True

ENSEMBLE_MODEL = None 
NLP_MODEL = None 
SPEECH_TO_TEXT_CONVERTER = None 

#Filename of the JSON database
filename = 'HerbalFinder/data/accounts.json'

with open("Plant_Data.json", "r", encoding="utf-8") as f:
    plant_data = load(f)

def readb64(raw_64_string):
    image_file_name = "input_img.jpg"
    s = base64.b64decode(raw_64_string)
    # nparr = np.fromstring(base64.b64decode(raw_64_string), np.uint8)
    nparr = np.frombuffer(s, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite(image_file_name, img)
    return img, image_file_name 

def image_resize(image, width = None, height = None):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return resized

@app.route('/login', methods = ['POST'])
def CheckAccount():
    f = open(filename)
    db = json.load(f)
    input = request.json
    #Initially set flag to False
    flag = False
    for i in db['users']:
        if ((db['users'][i]['username'] == input['username']) or (db['users'][i]['email'] == input['username'])) and db['users'][i]['password'] == input['password']:
            #Set flag to True if there is an account match and break the loop
            flag = True
            break
    f.close()
    response = {}
    response['match'] = flag
    print(jsonify(response))
    return jsonify(response)

#EDIT FOR SIGNUP
@app.route('/signup', methods = ['POST'])
def SignUpAccount():
    f = open(filename)
    db = json.load(f)
    print(db)
    input = request.json
    #Initially set the flag to false, assume that there is no match
    flag = False
    for i in db['users']:
        if (db['users'][i]['username'] == input['username']) or (db['users'][i]['email'] == input['email']):
            flag = True
            break
    f.close()
    f = open(filename, 'w')
    if flag == False:
        #Add Entries in Database
        db_size = len(db['users'])
        db_size = str(db_size)
        db['users'][db_size] = {'id': db_size, 'username': input['username'], 'email': input['email'], 'password': input['password']}
    print(db)
    json.dump(db, f, indent = 4)
    f.close()
    response = {}
    response['match'] = flag
    print(jsonify(response))
    return jsonify(response)

# Testing
@app.route('/test', methods = ['GET'])
def test_connection():
    response = {"text" : "Hello World"}
    return jsonify(response)

# Post Image Data (Test Version)
@app.route('/test_predict_image', methods=['POST'])
def predict_given_image():
    input_json = request.json
    input_image = input_json["input_image"]
    print(len(input_image))
    img = readb64(input_image)

    
    # Change this to only the unique ones
    response = {"predictions" : ["Jackfruit", "Mango", "Jackfruit", "Mango"]}
    return jsonify(response)

# Post NLP Data (Test Version)
@app.route('/test_predict_text', methods=['POST'])
def predict_given_text():
    input_json = request.json 
    input_text = input_json["input_text"]
    print(len(input_text))
    response = {"predictions" : ["Jackfruit", "Mango"]}
    return jsonify(response)

# Post Audio Data (Test Version)
@app.route('/test_predict_audio', methods=['POST'])
def convert_given_audio():
    input_json = request.json 
    input_audio = input_json["input_audio"]
    print(len(input_audio))
    response = {"predictions" : ["HIGH RISK HIGH BLOOD"]}
    return jsonify(response)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    input_json = request.json 

    input_image = input_json["input_image"]
    print(len(input_image))
    img, img_file_name = readb64(input_image)
    img = image_resize(img, height=500)
    _img = ENSEMBLE_MODEL.preprocess_image(img_file_name)
    predictions = ENSEMBLE_MODEL.predict_image(_img)
    response = {"predictions" : predictions}
    return jsonify(response)

@app.route('/predict_nlp', methods=['POST'])
def predict_text():
    input_json = request.json 
    print(input_json)
    input_text = input_json["input_text"]

    _predictions = NLP_MODEL.predict_given_text(input_text)
    predictions = NLP_MODEL.transform_predictions_result(_predictions)

    response = {"predictions" : predictions}
    return jsonify(response)

@app.route('/convert_audio', methods=['POST'])
def convert_input_audio():
    input_data = request.get_data()
    recording_filename = "input_recording.m4a"
    with open(recording_filename,"wb") as file:
        file.write(input_data)
    
    transcription = SPEECH_TO_TEXT_CONVERTER.transform(recording_filename)

    response = {"result" : transcription}
    return jsonify(response)

@app.route("/plant_data", methods=['POST'])
def get_plant_data():
    global plant_data
    requested_plant = request.json["plant_name"]
    response = {"text": plant_data[requested_plant]}

    return jsonify(response)


if __name__ == '__main__':
    if DEPLOY_MODELS:

        ENSEMBLE_MODEL  = EnsemblePredictor()
        NLP_MODEL = NLPPredictor()
        SPEECH_TO_TEXT_CONVERTER = SpeechToTextConverter()

    app.run(host="0.0.0.0", port=4000)