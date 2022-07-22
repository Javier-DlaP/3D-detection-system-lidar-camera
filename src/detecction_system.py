import os

import torch
import numpy as np
import pandas as pd

from distance_approximation import DistanceApprox
from pcl_img_utils import Pcl_Img_Utils
from bb3d_detector_fpp import FrustumPP_Detector
from bb2d_detector_yolo import Yolo_Detector

from time import time

class Detector3D:

    def __init__(self):
        self.current_dir = os.getcwd()
        self.time_dict = {'yolov5':[], 'distance_approximation':[], 'frustum_generation':[], 'frustum_pointpillars': []}

    def set_calibration_matrices(self, matrices):
        self.P2, self.R0_rect, self.Tr_velo_to_cam = matrices
        self.Tr_cam_to_velo = np.linalg.inv(self.Tr_velo_to_cam)

    def set_image_pointcloud(self, img_pcl):
        img, self.pcl = img_pcl
        self.img, self.np_img = img

    def load_models(self):
        """
        Load the models.
        """
        self.load_yolo_model()
        self.load_fpp_model()

    def detect_3d(self):
        """
        Obtain the 3D detections.
        """
        # Obtain 2D detections using YOLOv5
        t0 = time()
        self.detect_yolo()
        self.time_dict['yolov5'].append(time()-t0)
        # Approximate the distance of the detections
        t0 = time()
        self.approximate_distance()
        self.time_dict['distance_approximation'].append(time()-t0)
        if self.detections.shape[0] != 0:
            # Obtain the frustum points
            t0 = time()
            self.calculate_frustum_points()
            self.time_dict['frustum_generation'].append(time()-t0)
            # Obtain 3D detections using Frustum Pointpillars
            t0 = time()
            self.detect_frustum_pointpillars()
            self.time_dict['frustum_pointpillars'].append(time()-t0)
        else:
            self.final_detections = pd.DataFrame({'type':[],'x':[],'y':[],'z':[],'width':[],'length':[],'heigth':[],'rot_y':[]})

    def load_yolo_model(self, threshold=0.6):
        """
        Load the YOLOv5 model.
        """
        self.yolo_detector = Yolo_Detector(str(self.current_dir)+"/../ultralytics_yolov5_master/runs/train/yolov5m_ft/weights/best.pt",
                                          str(self.current_dir)+"/../ultralytics_yolov5_master/data/custom.yaml")
        self.yolo_detector.build_model()

    def detect_yolo(self):
        """
        Obtain the detections from the YOLOv5 model.
        """
        self.pred = self.yolo_detector.detect_yolo(self.np_img, 0.6)
        self.pred = np.array(self.pred[0].cpu())
        self.__process_detections()

    def __process_detections(self):
        """
        Preprocess the detections.
        """
        names = ['Car', 'Pedestrian', 'Cyclist']
        self.detections = pd.DataFrame(data=self.pred, columns=['xmin','ymin','xmax','ymax','score','name'])
        self.detections['name'] = list(map(lambda x: names[int(x)], self.detections['name'].tolist()))
        self.detections['complete_bb'] = (self.detections['xmin'] > 0) & (self.detections['xmax'] < self.img.size[0]-1)

    def approximate_distance(self):
        """
        Approximate the distance of the detections.
        """
        self.pcl_img_utils = Pcl_Img_Utils(self.pcl, self.img, self.P2, self.R0_rect, self.Tr_velo_to_cam,
                                           self.Tr_cam_to_velo, self.detections)
        self.distance_approx = DistanceApprox(self.pcl_img_utils)
        self.distance_approx.approximate_distance()
        self.detections = self.distance_approx.detections

    def calculate_frustum_points(self):
        """
        Calculate the frustum points.
        """
        self.pcls_bb, self.points3d, self.rots_y = self.pcl_img_utils.get_frustum(self.detections)

    def load_fpp_model(self):
        """
        Load the Frustum PointPillars model.
        """
        self.fpp_detector = FrustumPP_Detector(str(self.current_dir)+"/../Frustum_PointPillars/second/saved_models/frustum_pointpillars_all_12_base/")
        self.fpp_detector.build_model()
        self.fpp_detector.generate_anchors()

    def detect_frustum_pointpillars(self):
        """
        Obtain the detections from the Frustum PointPillars model.
        """
        self.fpp_detector.load_pointclouds(self.pcls_bb, self.detections['name'].tolist())
        fpp_detections = self.fpp_detector.detect()

        obj_type, x, y, z, width, length, heigth, rot_y = self.detections['name'].tolist(), [], [], [], [], [], [], []
        for i in range(len(fpp_detections)):
            x.append(float(fpp_detections[i]['box3d_lidar'][0][0]))
            y.append(float(fpp_detections[i]['box3d_lidar'][0][1]))
            z.append(float(fpp_detections[i]['box3d_lidar'][0][2]))
            width.append(float(fpp_detections[i]['box3d_lidar'][0][3]))
            length.append(float(fpp_detections[i]['box3d_lidar'][0][4]))
            heigth.append(float(fpp_detections[i]['box3d_lidar'][0][5]))
            rot_y.append(float(fpp_detections[i]['box3d_lidar'][0][6]))
        self.final_detections = pd.DataFrame({'type':obj_type,'x':x,'y':y,'z':z,'width':width,'length':length,'heigth':heigth,'rot_y':rot_y})

    def show_times(self):
        """
        Visualize mean inference time for each step
        """
        print('\nExecution time for each step:')
        total = 0
        for key in self.time_dict.keys():
            self.time_dict[key] = np.mean(self.time_dict[key])
            print('\tAverage time to execute',key,':',self.time_dict[key],'s')
            total += self.time_dict[key]
        print('\tAverage time to execute all pipeline:',total,'s ('+str(1/total)+' Hz)')

