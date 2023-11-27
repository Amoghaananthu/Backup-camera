#Installing Yolov8
# !pip install ultralytics

####################################################################################################################################
#Downloading dataset
# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="##########") # Use API-key for your device/download the dataset from Roboflow website  
# project = rf.workspace("roboflow-gw7yv").project("vehicles-openimages")
# dataset = project.version(1).download("yolov8")
####################################################################################################################################
#training using Yolov8
from ultralytics import YOLO
# !yolo task=detect mode=train model=yolov8m.pt data=Vehicles-OpenImages.v1-416x416.yolov8/data.yaml epochs=25 imgsz=416 %while training

####################################################################################################################################
#Video validation
model = YOLO('best.pt') #Copy the file from the runs/detect/train/weights to your main folder
source = 'Video1.mp4'
# source = 'Video2.mp4'
# source = 'Video3.avi'

results = model(source, show=True, conf=0.3, save=True)