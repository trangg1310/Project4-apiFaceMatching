from flask import Flask, request, jsonify
import cv2
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)
face_detect = cv2.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')
resnets = InceptionResnetV1(pretrained='vggface2').eval()

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Two images not uploaded'}), 400
        
        image_file1 = request.files['image1']
        image_np1 = np.frombuffer(image_file1.read(), np.uint8)
        image1 = cv2.imdecode(image_np1, cv2.IMREAD_COLOR)

        image_file2 = request.files['image2']
        image_np2 = np.frombuffer(image_file2.read(), np.uint8)
        image2 = cv2.imdecode(image_np2, cv2.IMREAD_COLOR)

        face1 = detect_and_extract_face(image1)
        face2 = detect_and_extract_face(image2)

        if face1 is None or face2 is None:
            return jsonify({'error': 'Face not detected in one or both images'}), 400

        face_tensor1 = preprocess_image(face1)
        face_tensor2 = preprocess_image(face2)
        batch_tensor = torch.stack([face_tensor1, face_tensor2])
        embeddings = resnets(batch_tensor).detach().numpy()
        distance = np.linalg.norm(embeddings[0] - embeddings[1])
        similarity_percentage = (1.0 - distance) * 100

        return jsonify({'similarity_percentage': similarity_percentage}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def detect_and_extract_face(image):
    faces = face_detect.detectMultiScale(image, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return image[y:y+h, x:x+w]
    return None

def preprocess_image(face):
    face_tensor = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_tensor = cv2.resize(face_tensor, (160, 160))
    face_tensor = np.transpose(face_tensor, (2, 0, 1))
    face_tensor = torch.Tensor(face_tensor)
    return face_tensor

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8800)
