import os
import pickle

import numpy as np

class DistanceApprox:

    def __init__(self, pcl_img_utils):
        self.pcl_img_utils = pcl_img_utils
        self.detections = pcl_img_utils.detections
        self.pcl_img_utils.calculate_point_cloud_projected()
        self.models_dir = os.getcwd()+"/../Distance approximation/"
        self.__load_models()
        
    def __load_models(self):
        """
        Load the regression models.
        """
        # Load the models
        height_model = None
        pc_proj_refinement_model = None
        error_model = None

        with open(self.models_dir+'height_model.pkl', 'rb') as f:
            height_model = pickle.load(f)
        with open(self.models_dir+'pc_proj_refinement_model.pkl', 'rb') as f:
            pc_proj_refinement_model = pickle.load(f)
        with open(self.models_dir+'error_model.pkl', 'rb') as f:
            error_model = pickle.load(f)
            
        self.distance_models = {'height_model':height_model,
                                'pc_proj_refinement_model':pc_proj_refinement_model,
                                'error_model':error_model}

    def approximate_distance(self):
        """
        Approximate the distance of the detections.
        """
        self.approximate_distance_height()
        # Get pointcloud by bounding box ordered by distance obtained from the height model
        self.pcl_projected_bbs, self.detections = self.pcl_img_utils.get_projected_pcs_bb()
        self.approximate_distance_pcl()
        self.approximate_distance_ensemble()

    def approximate_distance_height(self):
        """
        Approximate the distance of the detections using the height model.
        """
        model = self.distance_models['height_model']

        def aux_approximate_distance_height(type_name, top, bottom, bb_complete):
            """
            Auxiliary function to approximate the distance of the detections using the height model.
            """
            height = bottom - top
            if type_name == 'Car':
                if bb_complete:
                    f = model.car.bb_complete.function
                    return f(height, *model.car.bb_complete.variables)
                else:
                    f = model.car.bb_incomplete.function
                    return f(height, *model.car.bb_incomplete.variables)
            elif type_name == 'Pedestrian':
                f = model.pedestrian.function
                return f(height, *model.pedestrian.variables)
            elif type_name == 'Cyclist':
                f = model.cyclist.function
                return f(height, *model.cyclist.variables)
            else:
                raise ValueError("Type must be Car, Pedestrian or Cyclist")

        distances = list(map(aux_approximate_distance_height, self.detections['name'], self.detections['ymin'],
                                                              self.detections['ymax'], self.detections['complete_bb']))
        self.detections['distance_height'] = distances

    def approximate_distance_pcl(self):
        """
        Approximate the distance of the detections using the pointcloud projection model.
        """
        # Get model
        model = self.distance_models['pc_proj_refinement_model']
        # # Sort the detections by their aprroximated distance using the height projection model
        # self.detections = self.detections.sort_values(by='distance_height')

        def get_distances_pcl(type_name, pc_bb):
            """
            Auxiliary function to approximate the distance of the detections using the pointcloud projection model.
            """
            pc_distance_bb = pc_bb[2,:]
            if pc_distance_bb.size != 0: # Checks if there's any point on the bb 2D
                # Approximate distance
                distance = np.median(np.squeeze(np.asarray(pc_distance_bb)))
                if type_name == 'Car':
                    f = model.car.function
                    rectification = f(distance, *model.car.variables)
                    distance += rectification
                elif type_name == 'Pedestrian':
                    f = model.pedestrian.function
                    rectification = f(distance, *model.pedestrian.variables)
                    distance += rectification
                elif type_name == 'Cyclist':
                    f = model.cyclist.function
                    rectification = f(distance, *model.cyclist.variables)
                    distance += rectification
                else:
                    print("Type must be Car, Pedestrian or Cyclist")
                return distance
            else:
                return None

        # Map over bbs to get distance
        distances = list(map(lambda type_name, pc_distance_bb: get_distances_pcl(type_name, pc_distance_bb),
                             self.detections['name'], self.pcl_projected_bbs))

        self.detections['distance_pcl'] = distances
    
    def approximate_distance_ensemble(self):
        """
        Approximate the distance of the detections using the ensemble model.
        """
        # Get model
        error_model = self.distance_models['error_model']

        def ajust_distance(height_distance, bottom2d, top2d, pointcloud_distance, pointcloud_n_points, class_name):
            """
            Auxiliary function to approximate the distance of the detections using the ensemble model.
            """
            if pointcloud_distance == None:
                return height_distance
            else:
                height_2d = bottom2d - top2d
                height_distance_model, height_2d_model, pointcloud_distance_model, pointcloud_n_points_model = None, None, None, None
                # Obtain regression models based on the object type
                if class_name == 'Car':
                    height_distance_model = error_model.height_distance.car
                    height_2d_model = error_model.height_2d.car
                    pointcloud_distance_model = error_model.pointcloud_distance.car
                    pointcloud_n_points_model = error_model.pointcloud_n_points.car
                elif class_name == 'Cyclist':
                    height_distance_model = error_model.height_distance.cyclist
                    height_2d_model = error_model.height_2d.cyclist
                    pointcloud_distance_model = error_model.pointcloud_distance.cyclist
                    pointcloud_n_points_model = error_model.pointcloud_n_points.cyclist
                elif class_name == 'Pedestrian':
                    height_distance_model = error_model.height_distance.pedestrian
                    height_2d_model = error_model.height_2d.pedestrian
                    pointcloud_distance_model = error_model.pointcloud_distance.pedestrian
                    pointcloud_n_points_model = error_model.pointcloud_n_points.pedestrian
                else:
                    raise ValueError("Type must be Car, Pedestrian or Cyclist")   
                # Obtain an approximate error based on different metrics
                height_distance_error = height_distance_model.function(height_distance, *height_distance_model.variables)
                height_2d_error = height_2d_model.function(height_2d, *height_2d_model.variables)
                pointcloud_distance_error = pointcloud_distance_model.function(pointcloud_distance, *pointcloud_distance_model.variables)
                pointcloud_n_points_error = pointcloud_n_points_model.function(pointcloud_n_points, *pointcloud_n_points_model.variables)
                # Obtain final distance to use
                sum_error = height_distance_error + height_2d_error + pointcloud_distance_error + pointcloud_n_points_error
                distance = (1 - (height_distance_error + height_2d_error)/sum_error) * height_distance +\
                        (1 - (pointcloud_distance_error + pointcloud_n_points_error)/sum_error) * pointcloud_distance
                return distance

        distance = list(map(ajust_distance, self.detections['distance_height'], self.detections['ymax'], self.detections['ymin'],
                            self.detections['distance_pcl'], self.detections['n_points_bb'], self.detections['name']))

        self.detections['distance_ensemble'] = distance
