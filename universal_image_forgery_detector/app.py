import os
import uuid
import logging
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from copy import deepcopy
from sklearn.metrics import average_precision_score, accuracy_score
from models import get_model, VALID_NAMES
from models.imagenet_models import CHANNELS as IMAGENET_CHANNELS

import random

def set_seed(seed=40):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(40)
# Initialize the Flask application
app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

logging.basicConfig(level=logging.DEBUG)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def delete_all_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

import numpy as np

def find_best_threshold(y_true, y_pred):
    N = y_true.shape[0]
    sorted_pred = np.sort(y_pred)
    thresholds = (sorted_pred[:-1] + sorted_pred[1:]) / 2

    best_acc = 0
    best_thres = 0
    
    for thres in thresholds:
        temp = y_pred >= thres
        acc = np.mean(temp == y_true)
        if acc > best_acc:
            best_thres = thres
            best_acc = acc

    return best_thres


def calculate_best_threshold(model, data_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images).sigmoid().flatten()
            y_pred.extend(outputs.tolist())
            y_true.extend(labels.tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return find_best_threshold(y_true, y_pred)

def get_dummy_data_loader():
    images = torch.randn(100, 3, 224, 224)
    labels = torch.cat([torch.zeros(50), torch.ones(50)])
    dataset = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

def load_model(arch, ckpt):
    logging.debug(f"Loading model architecture: {arch}")
    model = get_model(arch)
    state_dict = torch.load(ckpt, map_location='cpu')

    num_ftrs = IMAGENET_CHANNELS[arch.split(':')[1]]

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    )

    # Filter the state_dict to match the shape of the model's fc layer
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model.fc.state_dict() and v.shape == model.fc.state_dict()[k].shape:
            new_state_dict[k] = v

    model.fc.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def get_transform(arch):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    selected_architectures = [
        #'Imagenet:resnet18',
        #'Imagenet:resnet34',
        'Imagenet:resnet50',
        #'Imagenet:resnet101',
        #'Imagenet:resnet152',
        #'Imagenet:vgg11',
        #'Imagenet:vgg19'
    ]
    ckpt = './pretrained_weights/fc_weights.pth'

    logging.debug(f"Selected architectures: {selected_architectures}")
    logging.debug(f"Checkpoint path: {ckpt}")

    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        delete_all_files(UPLOAD_FOLDER)
        
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # List to store results across all iterations
        all_results = []
        
        # Loop through the whole process 5 times
        for _ in range(5):
            iteration_results = []
            for arch in selected_architectures:
                model = load_model(arch, ckpt)
                transform = get_transform(arch)
                data_loader = get_dummy_data_loader()
                best_threshold = calculate_best_threshold(model, data_loader)
                print(f"Dynamically calculated best threshold for {arch}: {best_threshold}")
                
                # Run the model and get prediction
                result = predict_image(filepath, model, transform, best_threshold)
                iteration_results.append(result)
            
            # Collect the majority result from this iteration and store it
            final_result_for_iteration = max(set(iteration_results), key=iteration_results.count)
            all_results.append(final_result_for_iteration)
        
        # Find the overall result that appears the most across all iterations
        final_result = max(set(all_results), key=all_results.count)
        return redirect(url_for('result', filename=unique_filename, result=final_result))
    return redirect(request.url)


@app.route('/result/<filename>/<result>')
def result(filename, result):
    return render_template('result.html', filename=filename, result=result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def predict_image(image_path, model, transform, best_threshold):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image).sigmoid().item()
    print(f"Model output: {output}, Threshold: {best_threshold}")
    #Initially I was trying to set dynamic threshold. So here code to find dynamic threshold is available. But it was not giving good results. So changing it to fixed threshold.
    #return 'Real' if output < best_threshold else 'Fake'
    #return 'Real' if output < best_threshold else 'Fake'
    return 'Real' if output < 0.50 else 'Fake'

if __name__ == '__main__':
    app.run(debug=True)
