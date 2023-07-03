import json
import os
import cv2
import numpy as np
import math

SCALED_X = 384
SCALED_Y = 256

MCS_IMG_HEIGHT = 600
MCS_IMG_WIDTH = 400

class Camera(object):

    def __init__(self):

        self.camera_aspect_ratio = None
        self.camera_clipping_planes = None
        self.camera_field_of_view = None
        self.camera_height = None
        self.camera_position = None
        self.intrinsic_mat = None
        self.extrinsic_mat = None

    def load_cam_params(self, data):

        self.camera_aspect_ratio = data['camera_aspect_ratio']    
        self.camera_clipping_planes = data['camera_clipping_planes']
        self.camera_field_of_view = data['camera_field_of_view']
        self.camera_height = data['camera_height']
        self.camera_position = data['position']

    def set_intrinsic_mat(self):

        cx = self.camera_aspect_ratio[0] / 2
        cy = self.camera_aspect_ratio[1] / 2
        aspect_ratio = self.camera_aspect_ratio[0] / self.camera_aspect_ratio[1]

        fov_y = np.deg2rad(self.camera_field_of_view)
        fov_x = 2 * math.atan(aspect_ratio * math.tan(fov_y / 2.0))

        fx = cx / math.tan(fov_x / 2.0)
        fy = cy / math.tan(fov_y / 2.0)

        self.intrinsic_mat = [
                [fx, 0, cx, 0],
                [0, fy, cy, 0],
                [0, 0, 1, 0],
            ]
        self.focal_length_x = fx
        self.focal_length_y = fy
        # print(fx, fy)

    def set_extrinsic_mat(self):
        
        # currently hard-coded
        self.extrinsic_mat = [
                [1, 0, 0, self.camera_position['x']],
                [0, -1, 0, self.camera_position['y']],
                [0, 0, 1, -self.camera_position['z']],
                [0, 0, 0, 1]
            ]

class Object(object):

    def __init__(self):
        # 3D info
        self.bbox_corners = None
        self.cent_pos = None
        self.dims = None
        self.rotation = None        
        self.segment_color = None
        self.obj_id = None
        self.frame = None

    def set_object_info(self, data, frame, obj_id):
        self.bbox_corners = data[obj_id]['dimensions']
        self.cent_mass = data[obj_id]['position'] # the position data isn't aligned with the center of 3D bbox. Looks like this is the center of mass.
        self.rotation = data[obj_id]['rotation']
        self.segment_color = data[obj_id]['segment_color']
        self.obj_id = obj_id
        self.frame = frame   

class Track(object):

    def __init__(self, obj_id):

        self.track = []
        self.frames = []
        self.obj_id = obj_id
    
    def update_track(self, obj, frame):

        self.track.append(obj)
        self.frames.append(frame)

class Sequence(object):

    def __init__(self):
        self.cam = {}
        self.obj_by_frame = {}
        self.obj_by_id = {}

    def add_camera(self, frame, cam):
        self.cam[frame] = cam
    
    def add_obj_by_frame(self, frame, obj):
        if frame in self.obj_by_frame.keys():
            self.obj_by_frame[frame].append(obj)
        else:
            self.obj_by_frame[frame] = [obj]
    
    def add_obj_by_id(self, obj_id, obj):
        if obj_id in self.obj_by_id.keys():
            self.obj_by_id[obj_id].append(obj)
        else:
            self.obj_by_id[obj_id] = [obj]


def read_cam_params(data):

    cam = Camera()
    cam.load_cam_params(data)
    cam.set_intrinsic_mat()
    cam.set_extrinsic_mat()
    
    return cam

def read_objs(data, frame):
    
    obj_list = []
    for obj_id in data['object_list']:
        if data['object_list'][obj_id]['visible'] == True:
            obj = Object()
            obj.set_object_info(data['object_list'], frame, obj_id)
            obj_list.append(obj)

    exclude_id = ['floor', 'wall_left', 'wall_right', 'wall_front', 'wall_back']
    for obj_id in data['structural_object_list']:
        if obj_id in exclude_id:
            continue
        obj = Object()
        obj.set_object_info(data['structural_object_list'], frame, obj_id)
        obj_list.append(obj)

    return obj_list

def visualize_obj_center_mass(img, objs, cam):

    for obj in objs:
        world_coords = [obj.cent_mass['x'], obj.cent_mass['y'], obj.cent_mass['z'], 1]
        img_coords = np.dot(cam.intrinsic_mat, np.dot(cam.extrinsic_mat, world_coords))
        x = int(img_coords[0] / img_coords[2])
        y = int(img_coords[1] / img_coords[2])
        
        img = cv2.circle(img, (x, y), 1, list(obj.segment_color.values()), thickness=2)

    return img

def visualize_obj_center(img, objs, cam):

    for obj in objs:
        x = 0
        y = 0 
        z = 0
        for bbox in obj.bbox_corners:
            x += bbox['x']
            y += bbox['y']
            z += bbox['z']
        
        world_coords = [x / 8, y / 8, z / 8, 1]
        img_coords = np.dot(cam.intrinsic_mat, np.dot(cam.extrinsic_mat, world_coords))
        x = int(img_coords[0] / img_coords[2])
        y = int(img_coords[1] / img_coords[2])
        
        img = cv2.circle(img, (x, y), 1, list(obj.segment_color.values()), thickness=2)

    return img

def visualize_3D_bbox_corners(img, objs, cam):

    for obj in objs:
        for bbox in obj.bbox_corners:
            world_coords = [bbox['x'], bbox['y'], bbox['z'], 1]
            img_coords = np.dot(cam.intrinsic_mat, np.dot(cam.extrinsic_mat, world_coords))
            x = int(img_coords[0] / img_coords[2])
            y = int(img_coords[1] / img_coords[2])
            
            img = cv2.circle(img, (x, y), 1, list(obj.segment_color.values()), thickness=2)

    return img

def visualize_2D_bbox_corners(img, objs, cam):

    for obj in objs:
        x_all = []
        y_all = []
        for bbox in obj.bbox_corners:
            world_coords = [bbox['x'], bbox['y'], bbox['z'], 1]
            img_coords = np.dot(cam.intrinsic_mat, np.dot(cam.extrinsic_mat, world_coords))
            x = int(img_coords[0] / img_coords[2])
            y = int(img_coords[1] / img_coords[2])
            
            x_all.append(x)
            y_all.append(y)

        xy_all = [[max(x_all), max(y_all)], [max(x_all), min(y_all)],
                  [min(x_all), max(y_all)], [min(x_all), min(y_all)]]
        for xy in xy_all:
            x = xy[0]
            y = xy[1]
            img = cv2.circle(img, (x, y), 1, list(obj.segment_color.values()), thickness=2)

    return img

def obtain_amodal_center(objs, cam):

    img_shape = (MCS_IMG_HEIGHT, MCS_IMG_WIDTH)
    rescaled_shape = (SCALED_X, SCALED_Y)
    scale = np.divide(rescaled_shape, img_shape)
    
    amodal_center_all = {}
    # obj_number = 1
    obj_all=[]
    for obj in objs:
        x = 0
        y = 0 
        z = 0
        for bbox in obj.bbox_corners:
            x += bbox['x']
            y += bbox['y']
            z += bbox['z']
        
        world_coords = [x / 8, y / 8, z / 8, 1]

        cam_coords = np.dot(cam.extrinsic_mat, world_coords)
        assert(cam_coords[3] == 1)

        img_coords = np.dot(cam.intrinsic_mat, np.dot(cam.extrinsic_mat, world_coords))
        x = int(img_coords[0] / img_coords[2])
        y = int(img_coords[1] / img_coords[2])
        amodal_center_all[obj.obj_id] = [x, y]
        obj_all.append([x*scale[0], y*scale[1], cam_coords[2]])
        # obj_number+=1

    return amodal_center_all, obj_all

def obtain_amodel_center_depth(objs, cam):
    
    amodal_center_depth_all = {}
    for obj in objs:
        x = 0
        y = 0 
        z = 0
        for bbox in obj.bbox_corners:
            x += bbox['x']
            y += bbox['y']
            z += bbox['z']
        
        world_coords = [x / 8, y / 8, z / 8, 1]
        cam_coords = np.dot(cam.extrinsic_mat, world_coords)
        assert(cam_coords[3] == 1)

        amodal_center_depth_all[obj.obj_id] = cam_coords[2]
    
    return amodal_center_depth_all, cam_coords[2]

def backproject_from_2D_to_3D_amodal_ctr(amodal_center_all, amodal_center_depth_all, cam):
    
    reconstructed_3D_amodal_ctr = {}
    for obj_id in amodal_center_all.keys():
        assert(obj_id in amodal_center_depth_all.keys())
        amodal_center = amodal_center_all[obj_id]
        amodal_depth = amodal_center_depth_all[obj_id]
        Z_c = amodal_depth
        X_c = (amodal_center[0]*Z_c - (cam.camera_aspect_ratio[0] / 2)*Z_c) / cam.focal_length_x
        Y_c = (amodal_center[1]*Z_c - (cam.camera_aspect_ratio[1] / 2)*Z_c) / cam.focal_length_y
        reconstructed_3D_amodal_ctr[obj_id] = [X_c, Y_c, Z_c]

    return reconstructed_3D_amodal_ctr

def compare_original_and_reconstructed_3D_amodal_ctr(reconstructed_3D_amodal_ctr, objs, cam):

    for obj in objs:
        x = 0
        y = 0 
        z = 0
        for bbox in obj.bbox_corners:
            x += bbox['x']
            y += bbox['y']
            z += bbox['z']
        
        world_coords = [x / 8, y / 8, z / 8, 1]
        cam_coords = np.dot(cam.extrinsic_mat, world_coords)
        rec_coords = np.array(reconstructed_3D_amodal_ctr[obj.obj_id])

        print(cam_coords[:3], rec_coords)
        assert(np.sum(np.isclose(cam_coords[:3], rec_coords, rtol=0.1)) == rec_coords.shape[0])

    
def read_objs_new(data, frame, obj_name):
    
    obj_list = []
    for obj_id in data['object_list']:
        if obj_id in obj_name:
            if data['object_list'][obj_id]['visible'] == True:
                obj = Object()
                obj.set_object_info(data['object_list'], frame, obj_id)
                obj_list.append(obj)

    exclude_id = ['floor', 'wall_left', 'wall_right', 'wall_front', 'wall_back']
    for obj_id in data['structural_object_list']:
        if obj_id in exclude_id:
            continue
        if obj_id in obj_name:
            obj = Object()
            obj.set_object_info(data['structural_object_list'], frame, obj_id)
            obj_list.append(obj)

    return obj_list

