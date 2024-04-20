import cvlib as cv0
from cvlib.object_detection import draw_bbox
import cv2
import time
import numpy as np
import serial
import sklearn
import joblib
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import albumentations
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
#import argparse


lb = joblib.load('lb.pkl')

    
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # changed 3 to 1
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, len(lb.classes_))
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #print("Conv1 output shape:", x.shape)
        
        x = self.pool(F.relu(self.conv2(x)))
        #print("Conv2 output shape:", x.shape)
        
        x = self.pool(F.relu(self.conv3(x)))
        #print("Conv3 output shape:", x.shape)
        
        x = self.pool(F.relu(self.conv4(x)))
        #print("Conv4 output shape:", x.shape)
        
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


print('Loading model and label binarizer...')
lb = joblib.load('lb.pkl')
model = CustomCNN()
print('Model Loaded...')
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()
print('Loaded model state_dict...')
aug = albumentations.Compose([
    albumentations.Resize(224, 224),
    ])

t0 = time.time() 

def detectDrowning(source):
    isDrowning = False
    swimming_status=None
    fram=0
    #input from the camera
    cap = cv2.VideoCapture("videos/" + source)
    
    if not cap.isOpened():
        print('Error while trying to read video')
        
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    
    try:
            while cap.isOpened():
                status, frame = cap.read()

                # Apply object detection
                bbox, label, conf = cv0.detect_common_objects(frame)
                
                # If only one person is detected, use model-based detection
                if len(bbox) == 1:
                    bbox0 = bbox[0]
                    centre = [(bbox0[0] + bbox0[2]) / 2, (bbox0[1] + bbox0[3]) / 2]

                    start_time = time.time()
                    model.eval()
                    with torch.no_grad():
                        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        pil_image = aug(image=np.array(pil_image))['image']
                        if fram == 500:
                            break
                        fram += 1
                        pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
                        pil_image = torch.tensor(pil_image, dtype=torch.float).cpu()
                        pil_image = pil_image.unsqueeze(0)
                        outputs = model(pil_image)
                        _, preds = torch.max(outputs.data, 1)

                    print("Swimming status : ", lb.classes_[preds])
                    swimming_status = lb.classes_[preds]

                    isDrowning = swimming_status == 'drowning'

                    out = draw_bbox(frame, bbox, label, conf, isDrowning)
                    
                # If more than one person is detected, use logic-based detection
                elif len(bbox) > 1:
                    # Calculate the centroid of each bounding box
                    centres = []
                    for i in range(len(bbox)):
                        bbox_i = bbox[i]
                        centre_i = [(bbox_i[0] + bbox_i[2]) / 2, (bbox_i[1] + bbox_i[3]) / 2]
                        centres.append(centre_i)
                    
                    # Calculate the distance between each pair of centroids
                    distances = []
                    for i in range(len(centres)):
                        for j in range(i + 1, len(centres)):
                            dist = np.sqrt((centres[i][0] - centres[j][0]) ** 2 + (centres[i][1] - centres[j][1]) ** 2)
                            distances.append(dist)
                    
                    # If the minimum distance is less than a threshold, consider it as drowning
                    isDrowning = len(distances) > 0 and min(distances) < 50

    except ImportError as e:
            print("An error occurred:", str(e))

    finally:
            return swimming_status
            
 

#detectDrowning(args.source)



