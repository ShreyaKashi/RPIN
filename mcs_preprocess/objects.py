class Object(object):
    def __init__(self):
        # 3D info
        self.bbox_corners = None
        self.bbox_2d = None
        self.cent_pos = None
        self.dims = None
        self.rotation = None
        self.segment_color = None
        self.obj_id = None
        self.frame = None
        self.obj_type = None
        self.is_pole = None
        self.is_stationary = None

    def set_object_info(self, data, frame, obj_id):
        # self.bbox_corners = data[obj_id]['dimensions']
        self.bbox_2d = data[obj_id]["2dbbox"]
        # self.cent_mass = data[obj_id]['position'] # the position data isn't aligned with the center of 3D bbox. Looks like this is the center of mass.
        self.rotation = data[obj_id]["rotation"]
        self.segment_color = data[obj_id]["segment_color"]
        self.obj_id = obj_id
        self.frame = frame
        self.obj_type = data[obj_id]["role"]
        self.is_pole = data[obj_id]["is_pole"]
        self.is_stationary = data[obj_id]["is_stationary"]


class Sequence(object):
    def __init__(self):
        self.obj_by_frame = {}
        self.obj_by_id = {}
        self.mask_per_frame = {}
        self.depth_per_frame = {}

    def add_mask_for_frame(self, frame, mask):
        self.mask_per_frame[frame] = mask

    def add_depth_for_frame(self, frame, depth_img):
        self.depth_per_frame[frame] = depth_img

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