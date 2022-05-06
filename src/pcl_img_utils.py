import numpy as np

class Pcl_Img_Utils:

    def __init__(self, pcl, img, P2, R0_rect, Tr_velo_to_cam, detections):
        self.pcl = pcl
        self.img = img
        self.P2 = P2
        self.R0_rect = R0_rect
        self.Tr_velo_to_cam = Tr_velo_to_cam
        self.detections = detections
    
    def calculate_point_cloud_projected(self):
        """
        Calculate the pointcloud projected on the image.
        """
        points = self.pcl[:, 0:3] # lidar xyz (front, left, up)
        
        # Delete the points that are behind the camera
        velo = np.insert(points,3,1,axis=1).T
        velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
        self.pcl = velo.copy()
        
        # Transform the points to the camera frame
        cam = self.P2 * self.R0_rect * self.Tr_velo_to_cam * velo
        cam[:2] /= cam[2,:]
        # Create a mask with the points that are not on the image
        filter_cam = (cam[0,:]>=self.img.size[0]) & (cam[0,:]<0) & (cam[1,:]>=self.img.size[1]) & (cam[1,:]<0)
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

    def __calculate_projected_pcs_bb(self):
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
        self.__calculate_projected_pcs_bb()
        return self.pcl_projected_bbs, self.detections

########################################################################################################################
##################################1. APLICAR LOS FILTROS SOBRE LA NUBE DE PUNTOS########################################
#########################2. GENERAR NUEVO DATASET A PARTIR DEL GT Y LAS SECCIONES DE LA PC##############################
########################################################################################################################