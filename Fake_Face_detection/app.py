from flask import Flask, request,jsonify, render_template, redirect, url_for,Response
import requests
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import urllib.parse
import yaml
import faiss
import shutil
import cv2
import math
from tqdm import tqdm
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from utils.model_loader import ModelLoader
from backbone.backbone_def import BackboneFactory

# Set the environment variable to avoid the OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
app = Flask(__name__)
app.config['STATIC_FOLDER'] =  app.static_folder
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['CLAIM_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'claim_identity')
app.config['TEST_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'test')
app.config['CROP_CLAIM_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'crop_claim')
app.config['CROP_TEST_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'crop_test')
app.config['FEATURE_CLAIM_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'feature_claim')
app.config['FEATURE_TEST_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'feature_test')

folders = [app.config['UPLOAD_FOLDER'], app.config['STATIC_FOLDER'] ,app.config['CLAIM_FOLDER'], app.config['TEST_FOLDER'], app.config['CROP_CLAIM_FOLDER'], app.config['CROP_TEST_FOLDER'], app.config['FEATURE_CLAIM_FOLDER'], app.config['FEATURE_TEST_FOLDER']]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Encode the username and password
username = urllib.parse.quote_plus('aryanpopalghat4')
password = urllib.parse.quote_plus('DeepFake@2004')

# # MongoDB Atlas setup with SSL/TLS configurations
# client = MongoClient(
#     f'mongodb+srv://{username}:{password}@cluster0.gv2ym21.mongodb.net/',
#     tls=True,
#     tlsAllowInvalidCertificates=True  # Only use this for testing purposes
# )
# db = client.face_data

# Define the feature extraction model parameters
model_conf_file = './config/model_conf.yaml'
backbone_type = 'AttentionNet'
backbone_conf_file = './config/backbone_conf.yaml'
model_path = 'backbone/Epoch_17.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mean = 127.5
std = 128.0

# Load the model
backbone_factory = BackboneFactory(backbone_type, backbone_conf_file)
model_loader = ModelLoader(backbone_factory)
model = model_loader.load_model(model_path).eval().to(device)

# Transformations for images
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean / 255.0, mean / 255.0, mean / 255.0], std=[std / 255.0, std / 255.0, std / 255.0])
])

def extract_features(image_path):
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (112, 112))
    image = (image.transpose((2, 0, 1)) - mean) / std
    image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy().flatten()

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def distance2center(x1, y1, x2, y2, image):
    im_cx = int(image.shape[1] / 2)
    im_cy = int(image.shape[0] / 2)
    cx = ((x2 + x1) / 2).astype(int)
    cy = ((y1 + y2) / 2).astype(int)
    return math.sqrt(math.pow(im_cx - cx, 2) + math.pow(im_cy - cy, 2))

def Filter2centerBox(boundingBoxes, frame):
    min_distance = float('inf')
    min_idx = -1
    for i, det in enumerate(boundingBoxes):
        distance = distance2center(det[0], det[1], det[2], det[3], frame)
        if distance < min_distance:
            min_idx = i
            min_distance = distance
    return np.array([boundingBoxes[min_idx]])

def AlignedOneImageUsingFaceXAlignment(input_root, out_root, image_path, faceDetModelHandler, faceAlignModelHandler, face_cropper):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        input_height, input_width, _ = image.shape
    except:
        return
    dets = faceDetModelHandler.inference_on_image(image)
    if len(dets) > 0:
        dets = Filter2centerBox(dets, image)
        for i, det in enumerate(dets):
            assert (i != 1)  # only one face in picture
            landmarks = faceAlignModelHandler.inference_on_image(image, det)
            cropped_image = face_cropper.crop_image_by_mat(image, landmarks.reshape(-1))
            out_path = image_path.replace(input_root, out_root)
            if os.path.exists(os.path.dirname(out_path)) is False:
                os.makedirs(os.path.dirname(out_path))
            cv2.imwrite(out_path, cropped_image)
    else:
        out_path = image_path.replace(input_root, out_root)
        cv2.imwrite(out_path, image)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_claim', methods=['POST'])
def upload_claim_images():
    claim_images = request.files.getlist('claim')

    if len(claim_images) != 5:
        return 'Please upload exactly 5 claim images.'

    clear_directory(app.config['CLAIM_FOLDER'])

    for image in claim_images:
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['CLAIM_FOLDER'], filename)
        image.save(image_path)
       # db.images.insert_one({'name': filename, 'type': 'claim', 'path': image_path})

    # Automatically crop faces and extract features
    crop_faces_from_folder(app.config['CLAIM_FOLDER'], app.config['CROP_CLAIM_FOLDER'])
    extract_features_from_folder(app.config['CROP_CLAIM_FOLDER'], app.config['FEATURE_CLAIM_FOLDER'])

    return 'Claim images uploaded successfully.'

@app.route('/upload_test', methods=['POST'])
def upload_test_image():
    test_image = request.files.get('test')

    if not test_image:
        return 'Please upload a test image.'

    clear_directory(app.config['TEST_FOLDER'])
    clear_directory(app.config['STATIC_FOLDER'])

    filename = secure_filename(test_image.filename)
    test_image_path = os.path.join(app.config['TEST_FOLDER'], filename)
    static_image_path = os.path.join(app.config['STATIC_FOLDER'], filename)
    test_image.save(test_image_path)
    test_image.seek(0)
    test_image.save(static_image_path)
   # db.images.insert_one({'name': filename, 'type': 'test', 'path': test_image_path})
    
    
    # Automatically crop faces and extract features
    crop_faces_from_folder(app.config['TEST_FOLDER'], app.config['CROP_TEST_FOLDER'])
    extract_features_from_folder(app.config['CROP_TEST_FOLDER'], app.config['FEATURE_TEST_FOLDER'])

    return 'Test image uploaded successfully.'

def crop_faces_from_folder(input_folder, output_folder):
    with open('./config/model_conf.yaml') as f:
        model_conf = yaml.load(f, yaml.FullLoader)

    model_path = 'models'
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name = model_conf[scene][model_category]
    faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
    modelDet, cfgDet = faceDetModelLoader.load_model()
    faceDetModelHandler = FaceDetModelHandler(modelDet, device, cfgDet)

    model_category = 'face_alignment'
    model_name = model_conf[scene][model_category]
    faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
    modelAli, cfgAli = faceAlignModelLoader.load_model()
    faceAlignModelHandler = FaceAlignModelHandler(modelAli, device, cfgAli)

    face_cropper = FaceRecImageCropper()

    clear_directory(output_folder)

    for image in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image)
        AlignedOneImageUsingFaceXAlignment(input_folder, output_folder, image_path, faceDetModelHandler, faceAlignModelHandler, face_cropper)

def extract_features_from_folder(input_folder, output_folder):
    clear_directory(output_folder)

    for image in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image)
        features = extract_features(image_path)
        feature_path = os.path.join(output_folder, os.path.splitext(image)[0] + '.npy')
        np.save(feature_path, features)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    claim_features = []
    for feature_file in os.listdir(app.config['FEATURE_CLAIM_FOLDER']):
        feature_path = os.path.join(app.config['FEATURE_CLAIM_FOLDER'], feature_file)
        claim_features.append(np.load(feature_path))

    test_features = []
    for feature_file in os.listdir(app.config['FEATURE_TEST_FOLDER']):
        feature_path = os.path.join(app.config['FEATURE_TEST_FOLDER'], feature_file)
        test_features.append(np.load(feature_path))

    claim_features = np.array(claim_features)
    test_features = np.array(test_features)

    index = faiss.IndexFlatL2(claim_features.shape[1])  # Use the number of features in train_set
    index.add(claim_features)
    k_value = 4

    # Perform similarity search
    D, _ = index.search(test_features, k_value)
    distances = np.sum(D, axis=1)

    # Convert avg_similarity to Python float before jsonify
    avg_dissimilarity = float((distances[0])/  k_value)/100

    image_status = 'fake' if avg_dissimilarity > 0.55 else 'real'
    filename=os.listdir(app.config['STATIC_FOLDER'])
    result = {
        'nearest_dissimilarity': avg_dissimilarity,
        'image_status': image_status,
        'file_path': filename[0]
    }

    return render_template('result.html', result=result)

API_KEY = '$2a$10$B5k1PY5NCDNguz.l2zt0A.G.L3Z/smrrFZ3VgTcfJaLJGxtEpc3BC'


@app.route('/get_presigned_url', methods=['POST'])
def get_presigned_url():
    data = request.json
    if data is None or not isinstance(data, dict):
        return jsonify({'error': 'Invalid JSON format'}), 400

    file_name = data.get('fileName')
    file_size = data.get('fileSize')
    show_audio_result = False

    payload = {
        "fileName": file_name,
        "fileSize": file_size,
        "showAudioResult": show_audio_result
    }

    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': API_KEY,
        'Email': 'abhishek@meetkiwi.co',
        'password': 'SanFrancisco@2024',
        'Host': 'api.prd.realitydefender.xyz',
        'fileName': file_name
    }

    response = requests.post('https://api.prd.realitydefender.xyz/api/files/aws-presigned', json=payload, headers=headers)

    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({'error': 'Failed to get presigned URL'}), response.status_code

@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    file_name = request.form.get('fileName')

    if file is None or file_name is None:
        return jsonify({'error': 'No file uploaded or filename provided'}), 400

    file_size = len(file.read())
    file.seek(0)  # Reset file pointer to the beginning after reading its size

    # Get presigned URL
    try:
        presigned_url_response = requests.post('http://localhost:3000/get_presigned_url', json={
            "fileName": file_name,
            "fileSize": file_size,
            "showAudioResult": False
        })
    except requests.ConnectionError as e:
        return jsonify({'error': 'Connection to the presigned URL service failed', 'details': str(e)}), 500

    if presigned_url_response.status_code != 200:
        return Response(response=presigned_url_response.text, status=presigned_url_response.status_code, content_type='application/json')

    presigned_url = presigned_url_response.json().get('response', {}).get('signedUrl')
          
    if not presigned_url:
        return jsonify({'error': 'Failed to get presigned URL'}), 500

    # Upload file to presigned URL
    headers = {'Content-Type': 'application/octet-stream'}
    upload_response = requests.put(presigned_url, data=file, headers=headers)

    if upload_response.status_code == 200:
        return jsonify({'message': 'File uploaded successfully'})
    else:
        return jsonify({'error': 'Failed to upload file'}), upload_response.status_code

@app.route('/get_all_files', methods=['GET'])
def get_all_files():
    try:
        headers = {
            'X-API-KEY': API_KEY,
            'Host': 'api.prd.realitydefender.xyz'
        }

        response = requests.get('https://api.prd.realitydefender.xyz/api/media/users', headers=headers)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch media data'}), response.status_code

        media_data = response.json()
        media_list = media_data.get('mediaList', [])  # Updated to 'mediaList' instead of 'response'

        file_statuses = []
        for media in media_list:
            models_info = []
            for model in media.get('models', []):
                models_info.append({
                    'name': model.get('name'),
                    'status': model.get('status'),
                    'score': model.get('finalScore')
                })

            file_statuses.append({
                'fileName': media.get('filename'),  # Updated to 'filename' instead of 'fileName'
                'status': media.get('overallStatus'),
                'models': models_info
            })

        return jsonify({'files': file_statuses})
    except Exception as e:
        return jsonify({'error': f'Error in get_all_files: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=3000)
    
    