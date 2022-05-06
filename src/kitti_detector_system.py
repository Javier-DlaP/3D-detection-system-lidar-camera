import numpy as np
from PIL import Image
import pandas as pd

class KittiDetector3D:
    def __init__(self, path_kitti):
        self.path_kitti = path_kitti

    def get_id_calib(self, id):
        """
        Returns the calibration data for the given id.
        """
        name = '%06d'%id
        P2, R0_rect, Tr_velo_to_cam = self.get_calibration_data(self.path_kitti+'calib/'+name+'.txt')
        return P2, R0_rect, Tr_velo_to_cam

    def get_id_img_pcl(self, id):
        """
        Returns the pointcloud and image data for the given id.
        """
        name = '%06d'%id
        img = self.set_image(self.path_kitti+'image_2/'+name+'.png')
        pcl = self.get_pointcloud(self.path_kitti+'velodyne/'+name+'.bin')
        return img, pcl

    def get_calibration_data(self, calibration_data_file):
        """
        Extracts the calibration data from the calibration data file.
        """
        calib = None
        with open(calibration_data_file,'r') as f:
            calib = f.readlines()

        # P2 (3 x 4) for left eye (intrinsic matrix in homogeneous coordinates)
        P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
        
        R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
        # Add a 1 in bottom-right, reshape to 4 x 4
        R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
        R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
        
        Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
        Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)

        return P2, R0_rect, Tr_velo_to_cam

    def get_pointcloud(self, pointcloud_data_file):
        """
        Extracts the pointcloud data from the pointcloud data file.
        """
        return np.fromfile(pointcloud_data_file, dtype=np.float32).reshape((-1,4))

    def set_image(self, camera_image_file):
        """
        Extracts the image data from the image data file.
        """
        return Image.open(camera_image_file)

    def load_groundtruth(self):
        """
        Loads the groundtruth data from the groundtruth data file.
        """
        groundtruth_file = self.path_kitti+'kitti_gt.csv'
        df = pd.read_csv (groundtruth_file)
        # Rename some columns
        df.rename(columns={'left':'xmin', 'top':'ymin', 'right':'xmax', 'bottom':'ymax'}, inplace=True)
        self.groundtruth = df

    def get_groundtruth_frame(self, frame):
        """
        Returns the groundtruth data for the given frame.
        """
        return self.groundtruth.loc[self.groundtruth['frame'] == frame]
