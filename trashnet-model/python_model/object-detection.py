from imageai.Detection import ObjectDetection
import os
import torch
import tensorflow as tf

execution_path = str(os.path.dirname(os.path.abspath(__file__)))

def detectObjects(img_dir=execution_path + "\\data\\unseen\\demo20.jpg"):
   detector = ObjectDetection()
   detector.setModelTypeAsRetinaNet()
   detector.setModelPath(os.path.join(execution_path , "retina.pth"))
   detector.loadModel()
   detections = detector.detectObjectsFromImage(input_image=img_dir, output_image_path=os.path.join(execution_path , "imagenew.jpg"), extract_detected_objects=True, minimum_percentage_probability=30, output_type='file')
   print(len(detections))
   # temp=torch.from_numpy(detections[0])
   # temp=tf.transpose(temp, perm=[2,0,1])
   # return temp
detectObjects()