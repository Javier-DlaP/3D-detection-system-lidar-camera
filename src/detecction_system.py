import os

import torch

from distance_approximation import DistanceApprox
from pcl_img_utils import Pcl_Img_Utils

class Detector3D:

    def set_calibration_matrices(self, matrices):
        self.P2, self.R0_rect, self.Tr_velo_to_cam = matrices

    def set_image_pointcloud(self, img_pcl):
        self.img, self.pcl = img_pcl

    def load_models(self):
        """
        Load the models.
        """
        self.load_yolo_model()

    def detect_3d(self):
        """
        Obtain the 3D detections.
        """
        # Obtain 2D detections using YOLOv5
        self.detect_yolo()
        # Approximate the distance of the detections
        self.approximate_distance()

    def load_yolo_model(self, threshold=0.6):
        """
        Load the YOLOv5 model.
        """
        torch.hub.set_dir(os.getcwd()+'/..')
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l')#, force_reload=True)
        yolo_model.classes = [0,1,2]
        yolo_model.conf = threshold
        self.yolo_model = yolo_model

    def detect_yolo(self):
        """
        Obtain the detections from the YOLOv5 model.
        """
        self.yolo_detections = self.yolo_model(self.img).pandas().xyxy[0]
        self.__preprocess_detections()

    def __preprocess_detections(self):
        """
        Preprocess the detections.
        """
        dict_names = {'car':'Car', 'person':'Pedestrian', 'bicycle':'Cyclist'}
        self.yolo_detections['name'] = self.yolo_detections['name'].map(lambda x: dict_names[x])
        self.yolo_detections['complete_bb'] = (self.yolo_detections['xmin'] > 0) & (self.yolo_detections['xmax'] < self.img.size[0]-1)

    def approximate_distance(self):
        """
        Approximate the distance of the detections.
        """
        self.pcl_img_utils = Pcl_Img_Utils(self.pcl, self.img, self.P2, self.R0_rect, self.Tr_velo_to_cam, self.yolo_detections)
        # self.pcl_img_utils.calculate_point_cloud_projected()
        # pcl_projected_bbs = self.pcl_img_utils.get_projected_pcs_bb()
        self.distance_approx = DistanceApprox(self.pcl_img_utils)
        self.distance_approx.approximate_distance()

    # def proyect_2d_to_3d_points(self, points_2d):
    #     """
    #     Proyect 2D points to the 3D pointcloud coordinate system.
    #     """
    #     def solve_equation(q):  
    #         xi, yi, d = q
    #         def equations(p):
    #             xc, yc, zc, w = p
    #             eq1 = (self.P2[0,0]*xc + self.P2[0,1]*yc + self.P2[0,2]*zc + self.P2[0,3])/ xi - w
    #             eq2 = (self.P2[1,0]*xc + self.P2[1,1]*yc + self.P2[1,2]*zc + self.P2[1,3])/ yi - w
    #             eq3 = (self.P2[2,0]*xc + self.P2[2,1]*yc + self.P2[2,2]*zc + self.P2[2,3]) - w
    #             eq4 = xc**2 + yc**2 + zc**2 - d**2
    #             return (eq1,eq2,eq3,eq4)
    #         values = fsolve(equations, (0,0,0,0))
    #         return values

    #     p = Pool(8)
    #     start = time.time()
    #     xd = list(p.map(solve_equation, [(100,200,20)]*2))
    #     print(time.time()-start)
    #     p.close()
        

