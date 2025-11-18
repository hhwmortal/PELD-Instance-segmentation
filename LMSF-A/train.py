import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='home/data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=8,
                close_mosaic=0,
                workers=8,
                optimizer='SGD',
                project='runs/train',
                name='exp',
                )