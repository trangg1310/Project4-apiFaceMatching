import cv2
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
from opencv.fr import FR
BACKEND_URL = "https://us.opencv.fr"
DEVELOPER_KEY = "8sxYofjNGRmNmQ4MTEtMzFiOS00YWU0LTkxZDQtOWM5YjMyMDQ0Nzdl"
sdk = FR(BACKEND_URL, DEVELOPER_KEY)
from opencv.fr.compare.schemas import CompareRequest
from opencv.fr.search.schemas import SearchMode
from pathlib import Path

face_detect = cv2.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0) #0- cammera của máy/ 1-cam rời gắn với máy/ hoặc 1 đường link video

resnets = InceptionResnetV1(pretrained='vggface2').eval() #khởi tạo đối tượng InceptionResnetV1 để trích xuất đặc trưng khuôn mặt, với mô hình được huấn luyện trước trên vggface2, eval là để đánh giá/ không huấn luyện

#đọc khuôn mặt cho trước
face_reference_path = 'imgs\source\source.jpg'
face_reference = cv2.imread(face_reference_path)
face_reference = cv2.cvtColor(face_reference, cv2.COLOR_BGR2LAB)

#chuyển kích thước khuôn mặt cho trước về (3,160,160)
face_reference = cv2.resize(face_reference, (160,160))
face_reference = np.transpose(face_reference, (2,0,1))
face_reference_tensor = torch.Tensor(face_reference)

while True:
    _, frame = video.read()
    faces = face_detect.detectMultiScale(frame,1.3,5) # scaleFactor - tỉ lệ kích thước ảnh đầu vào và ảnh sử dụng để phát hiện khuôn mặt (thường từ 1.01 - 1.5)
    # minNeighbors - số lượng khuôn mạt được phát hiện trong khu vực gần nhau trên ảnh

    for(x,y,w,h) in faces:
        #cv2.imwrite('imgs/p1_{}.jpg'.format(count),frame[y: y+h, x: x+w]) #lưu ảnh khuôn mặt đã cắt   
        cv2.rectangle(frame,(x,y),(x+w, y+h), (0,255,0))
        face_tensor = cv2.cvtColor(frame[y: y+h, x: x+w], cv2.COLOR_BGR2RGB) # chuyển đổi màu sắc từ BGR sang RGB????
        face_tensor = cv2.resize(face_tensor, (160, 160))
        face_tensor = np.transpose(face_tensor, (2, 0, 1))
        face_tensor = torch.Tensor(face_tensor)
        #face_tensors.append(face_tensor.transpose((2, 0, 1))) # chuyển đổi kích thước của ảnh thành (3, 160, 160) để phù hợp với mô hình
        #facr_tensor có kích thước (height, width, channels), transpose để hoán vị các trục sao cho dữ liệu phù hợp với thuật toán
           
        # Chuyển đổi face_tensor và reference_face_tensor thành batch tensor
        batch_tensor = torch.stack([face_reference_tensor, face_tensor])
        
        # Trích xuất đặc trưng khuôn mặt từ batch tensor
        embeddings = resnets(batch_tensor).detach().numpy()
    
        # Tính khoảng cách Euclidean giữa hai vector đặc trưng
        
        compare_same_object = CompareRequest([frame], [face_reference_path], search_mode=SearchMode.FAST)

        
         # Điều kiện so sánh để xác định hai khuôn mặt giống nhau
        cv2.putText(frame,sdk.compare.compare_image_sets(compare_same_object), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           
       
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(30) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

