import time
import numpy as np
from scipy.optimize import fsolve

class Pcl_Img_Utils:

    def __init__(self, pcl, img, P2, R0_rect, Tr_velo_to_cam, detections):
        self.pcl = pcl
        self.img = img
        self.P2 = P2
        self.R0_rect = R0_rect
        self.P2_R0_rect = self.P2 * self.R0_rect
        self.Tr_velo_to_cam = Tr_velo_to_cam
        self.Tr_cam_to_velo_rect = np.linalg.inv(R0_rect * Tr_velo_to_cam)
        self.detections = detections

    def __init__(self, pcl, img, P2, R0_rect, Tr_velo_to_cam, Tr_cam_to_velo, detections):
        self.pcl = pcl
        self.img = img
        self.P2 = P2
        self.R0_rect = R0_rect
        self.P2_R0_rect = self.P2 * self.R0_rect
        self.Tr_velo_to_cam = Tr_velo_to_cam
        self.Tr_cam_to_velo_rect = Tr_cam_to_velo
        self.detections = detections

    def set_detections(self, detections):
        self.detections = detections
    
    def calculate_point_cloud_projected(self):
        """
        Calculate the pointcloud projected on the image.
        """
        points = self.pcl[:, 0:3] # lidar xyz (front, left, up)
        
        # Delete the points that are behind the camera
        velo = np.insert(points,3,1,axis=1).T
        velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
        
        # Transform the points to the camera frame
        self.pcl = self.Tr_velo_to_cam * velo
        cam = self.P2 * self.R0_rect * self.pcl
        # self.pcl = velo
        # cam = self.P2 * self.R0_rect * self.Tr_velo_to_cam * velo
        cam[:2] /= cam[2,:]
        # Create a mask with the points that are not on the image
        filter_cam = ~((cam[0,:]<self.img.size[0]) & (cam[0,:]>=0) & (cam[1,:]<self.img.size[1]) | (cam[1,:]>=0))
        filter_cam = np.array(filter_cam).flatten()

        # Apply the mask to the projected points
        cam = np.delete(cam,np.where(filter_cam),axis=1)
        
        # Save the mask to be used later
        self.pcl_projected_filter = filter_cam
        # Save the projected points
        self.pcl_projected = cam

    def get_point_cloud_projected(self):
        """
        Get the pointcloud projected on the image.
        """
        self.__calculate_point_cloud_projected()
        return self.pcl_projected

    def calculate_projected_pcs_bb(self):
        """
        Calculate the projected pointcloud that correspond to the bounding box of the detections.
        """
        projected_pcs_bb = []
        lefts2d, tops2d, rights2d, bottoms2d = self.detections['xmin'], self.detections['ymin'], self.detections['xmax'], self.detections['ymax']
        projected_pc = self.pcl_projected
        
        pcl_projected_filter_objs = []
        for left, top, right, bottom in zip(lefts2d, tops2d, rights2d, bottoms2d):
            filter_bb = (left > projected_pc[0,:]) | (top > projected_pc[1,:]) |\
                        (projected_pc[0,:] > right) | (projected_pc[1,:] > bottom)
            filter_bb = np.array(filter_bb).flatten()
            # Save the mask to be used later
            pcl_projected_filter_objs.append(filter_bb)
            # Apply the mask to the projected points
            pc_distance_bb = np.delete(projected_pc,np.where(filter_bb),axis=1)
            # Points on bb
            projected_pcs_bb.append(pc_distance_bb)
            # Deleting saved points
            projected_pc = np.delete(projected_pc,np.where(~filter_bb),axis=1)

        # Count the number of points on the bb
        self.detections['n_points_bb'] = list(map(lambda x: x.shape[1], projected_pcs_bb))

        self.pcl_projected_bbs_filters = pcl_projected_filter_objs
        self.pcl_projected_bbs = projected_pcs_bb

    def get_projected_pcs_bb(self):
        """
        Get the projected pointcloud that correspond to the bounding box of the detections.
        """
        self.detections = self.detections.sort_values(by='distance_height')
        self.calculate_projected_pcs_bb()
        return self.pcl_projected_bbs, self.detections

    def get_pcls_bb(self):
        """
        Calculate the pointcloud that correspond to the bounding box of the detections.
        """
        self.pcl = np.delete(self.pcl,np.where(self.pcl_projected_filter),axis=1)
        pcl_bbs = []
        # Apply filters to the pointcloud and save the points that are on the bb
        for filterbb in self.pcl_projected_bbs_filters:
            pcl_bbs.append(np.delete(self.pcl,np.where(filterbb),axis=1))
            self.pcl = np.delete(self.pcl,np.where(~filterbb),axis=1)
        return pcl_bbs

    def __solve_equation(self, q):
        """
        Auxiliary function to solve the equation for obtaining the points 3d.
        """
        xi, yi, d = q
        def equations(p):
            xc, yc, zc, w = p
            #xc, yc, zc = min(80,max(0,xc)), min(80,max(0,yc)), min(80,max(0,zc))
            eq1 = (self.P2_R0_rect[0,0]*xc + self.P2_R0_rect[0,1]*yc + self.P2_R0_rect[0,2]*zc + self.P2_R0_rect[0,3])/ xi - w
            eq2 = (self.P2_R0_rect[1,0]*xc + self.P2_R0_rect[1,1]*yc + self.P2_R0_rect[1,2]*zc + self.P2_R0_rect[1,3])/ yi - w
            eq3 = (self.P2_R0_rect[2,0]*xc + self.P2_R0_rect[2,1]*yc + self.P2_R0_rect[2,2]*zc + self.P2_R0_rect[2,3]) - w
            eq4 = xc**2 + yc**2 + zc**2 - d**2
            return (eq1,eq2,eq3,eq4)
        
        values = fsolve(equations, (0,0,0,0)).reshape(4,-1)
        values[3] = 0
        return values

    def __calculate_points_3d(self, noise_f = lambda x: x):
        """
        Calculate the points 3d from a 2d point and the distance to that point.
        """
        # Calculate xmean and ymean in the detection dataframe
        self.detections['xmean'] = (self.detections['xmin'] + self.detections['xmax'])/2
        self.detections['ymean'] = (self.detections['ymin'] + self.detections['ymax'])/2

        # Apply the noise to the points
        self.detections['distance_ensemble_noise'] = self.detections.apply(lambda x: noise_f(x['distance_ensemble']), axis=1)

        # Get the points 2d from the detections with format [(xmean,ymean,distance),...]
        points_2d = self.detections[['xmean','ymean','distance_ensemble_noise']].values
        
        # Solve the equation to get the points 3d
        points_3d = list(map(self.__solve_equation, points_2d))

        self.points_3d = points_3d

    def get_points_3d(self, noise_f = lambda x: x):
        """
        Get the points 3d from a 2d point and the distance to that point.
        """
        self.__calculate_points_3d(noise_f = noise_f)
        return self.points_3d

    def __calculate_gt_points_3d(self):
        """
        Calculate the points 3d from the detections dataframe.
        """
        # Create a numpy array with zeros that has the same length as the detections dataframe
        zeros = np.zeros((self.detections.shape[0]))

        # Create a numpy array with the points 3d
        points_3d = np.array([np.asarray(self.detections['x'].tolist()),
                              np.asarray(self.detections['y'].tolist()),
                              np.asarray(self.detections['z'].tolist()), zeros], dtype=np.float32).T.tolist()
        self.points_3d = points_3d

    def get_gt_points_3d(self):
        """
        Get the points 3d from the detections dataframe.
        """
        self.__calculate_gt_points_3d()
        return self.points_3d

    def move_pcs_bb(self, points_3d, pcl_bbs):
        """
        Move each pcl_bb to the point_3d.
        """
        points_3d = np.array(points_3d).reshape(len(points_3d),4).T
        # Change the fourth column to 0 (to not substract by the fourth dimension)
        points_3d[3,:] = 0
        points_3d = np.reshape(np.array(points_3d).T,(points_3d.shape[1],4,1))
        # Move the pointcloud
        pcl_bbs = list(map(lambda pcl_bb, point_3d: pcl_bb - point_3d, pcl_bbs, points_3d))
        return pcl_bbs

    def rotate_pcl_bb_y(self, points3d, pcl_bbs, offset):
        """
        Rotate each pcl_bb to the point_3d in the y axis.
        """
        # Get the angle between the point3d and (0,0,0) on the y axis
        anglesy = list(map(lambda point3d: -np.arctan2(point3d[0],point3d[2]), points3d))
        # Get the rotation over the y axis
        rots_y = list(map(lambda angle: np.array([[np.cos(angle),0,np.sin(angle),0],
                                                  [0,1,0,0],
                                                  [-np.sin(angle),0,np.cos(angle),0],
                                                  [0,0,0,1]], dtype=np.float), anglesy))
        # Apply the rotation over the y axis in the pcl_bb
        pcl_bbs = list(map(lambda pcl_bb, rot_z: np.dot(rot_z,pcl_bb), pcl_bbs, rots_y))
        # Apply the rotation over the point3d
        points3dy = np.array(list(map(lambda point3d, rot_y: np.dot(rot_y,point3d), points3d, rots_y)))

        points3dy[:,1] = 0
        points3dy[:,2] -= offset

        return points3dy, anglesy, pcl_bbs

    def filtrate_frustums(self, max_distance, pcl_bbs, offset):
        """
        Filtrate the pointcloud to eliminate points out of bounds
        """
        pcl_bbs = list(map(lambda pcl_bb: np.delete(pcl_bb,np.where(abs(pcl_bb[0,:]-offset)>max_distance),axis=1), pcl_bbs))
        return pcl_bbs

    def get_frustum(self, distance):
        """
        Get the frustum that correspond to the bounding box of the detections.
        """
        pcls_bb = self.get_pcls_bb()
        pcls_bb = list(map(lambda pcl_bb: np.dot(self.R0_rect,pcl_bb), pcls_bb))
        points3d = self.get_points_3d()
        points3d, rots_y, pcls_bb = self.rotate_pcl_bb_y(points3d, pcls_bb, 0)
        pcls_bb = self.move_pcs_bb(points3d, pcls_bb)
        pcls_bb = list(map(lambda pcl_bb: np.dot(self.Tr_cam_to_velo_rect ,pcl_bb), pcls_bb))
        pcls_bb = self.filtrate_frustums(5.5, pcls_bb, 0)
        return pcls_bb, points3d, rots_y

    def get_gt_frustum(self, noise_distance_f, max_distance, offset):
        """
        Get the frustum that correspond to the bounding box of the detections.
        """
        start = time.time()
        pcls_bb = self.get_pcls_bb()
        pcls_bb = list(map(lambda pcl_bb: np.dot(self.R0_rect,pcl_bb), pcls_bb))
        points3d = self.get_points_3d(noise_f = noise_distance_f)
        points3d, rots_y, pcls_bb = self.rotate_pcl_bb_y(points3d, pcls_bb, offset)
        pcls_bb = self.move_pcs_bb(points3d, pcls_bb)
        pcls_bb = list(map(lambda pcl_bb: np.dot(self.Tr_cam_to_velo_rect ,pcl_bb), pcls_bb))
        pcls_bb = self.filtrate_frustums(max_distance, pcls_bb, offset)
        end = time.time()
        return pcls_bb, points3d, rots_y, end-start