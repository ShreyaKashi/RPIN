import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
import json
import tqdm
import utils as utils
import shutil
import pickle
import cv2
from pathlib import Path
import cam_help as cam_help
import copy
import random
# from dotenv import load_dotenv

# load_dotenv(dotenv_path=Path("/home/kashis/Desktop/Eval7/RPIN/.env"))
MCS_ROOT_DIR = './after_eval'
STORE_PATH = os.getenv("STORE_PATH")
# OUTPUT_DIR = os.getenv("OUTPUT_DIR")
# OUTPUT_DIR = os.getenv("OUTPUT_DIR")
# OUTPUT_DIR = './after_eval_process_3d_center'

# MCS_ROOT_DIR = os.getenv("MCS_ROOT_DIR")
# STORE_PATH = os.getenv("STORE_PATH")
# OUTPUT_DIR = os.getenv("OUTPUT_DIR")

MAX_OBJS = 3
SCALED_X = 384
SCALED_Y = 256

MIN_VID_LEN = 30

MCS_IMG_HEIGHT = 600
MCS_IMG_WIDTH = 400

def get_obj_entrance_events(seq):
        objEntranceEventDict = {}

        for obj_id, obj_details in seq.obj_by_id.items():
            objEntranceEventDict[obj_id] = obj_details[0].frame

        return objEntranceEventDict

def get_obj_exit_events(seq):
        objEntranceEventDict = {}

        for obj_id, obj_details in seq.obj_by_id.items():
            objEntranceEventDict[obj_id] = obj_details[-1].frame

        return objEntranceEventDict

def get_reqd_scenes_list():
    scene_list = []
    for scene_name in os.listdir(MCS_ROOT_DIR):

        # Get only plaus scenes with no occluders
        if "implaus" in scene_name:
            continue

        folder_path = os.path.join(MCS_ROOT_DIR, scene_name)
        
        step_output_folder = os.path.join(folder_path, "Step_Output")
        step_out_initial = step_output_folder + "/" + os.listdir(step_output_folder)[0]

        with open(step_out_initial) as f:
            step_out_content = json.load(f)
            if any("occluder" in key for key in step_out_content["structural_object_list"]):
                continue

        scene_list.append(scene_name)
    # print(scene_list)
    # print(sorted(scene_list))
    random.seed(0)
    random.shuffle(scene_list)
    return scene_list
    
def get_occ_scenes_list():
    scene_list = []
    for scene_name in os.listdir(MCS_ROOT_DIR):

        # Get only plaus scenes with no occluders
        if "implaus" in scene_name:
            continue

        folder_path = os.path.join(MCS_ROOT_DIR, scene_name)
        
        step_output_folder = os.path.join(folder_path, "Step_Output")
        step_out_initial = step_output_folder + "/" + os.listdir(step_output_folder)[0]

        with open(step_out_initial) as f:
            step_out_content = json.load(f)
            if any("occluder" in key for key in step_out_content["structural_object_list"]):
                scene_list.append(scene_name)
    # print(scene_list)
    # print(sorted(scene_list))
    random.seed(0)
    random.shuffle(scene_list)
    return scene_list
    

def get_step_processed_out(scene_name):
    
    scene_path = f"{MCS_ROOT_DIR}/{scene_name}"
    scene_metadata = utils.get_scene_metadata(
        scene_path, scene_name, store_path=STORE_PATH, load=False, save=False
    )
    expected_tracks, vid_len_details = utils.get_tracklets(
        scene_path,
        scene_name,
        store_path=STORE_PATH,
        load=False,
        save=False,
        provide_shape=True,
    )

    # print("expected_tracks",expected_tracks)

    seq = utils.get_metadata_from_pipleine(expected_tracks, scene_metadata, vid_len_details)

    # print("seq",seq)
    states_dict_2 = utils.preprocessing(vid_len_details, seq, scene_metadata)

    return expected_tracks, scene_metadata, seq, states_dict_2   

def get_vid_start_len(scene_name, seq, states_dict_2):
    # Find start frame no and vid len
    focused_objs = [k for k, v in states_dict_2.items() if v["obj_type"] == "focused"]
    obj_entrance_events = get_obj_entrance_events(seq)
    obj_exit_events = get_obj_exit_events(seq)

    if "grav" in scene_name:
        # start_frame is frame when obj detaches from placer
        placer_obj_id = [k for k, v in states_dict_2.items() if v["is_pole"]][0]
        placer_entrance_frame = obj_entrance_events[placer_obj_id]

        all_focused_obj_ids = [k for k, v in states_dict_2.items() if v["obj_type"] == "focused" and not v["is_stationary"]]
        # print(all_focused_obj_ids)
        assert len([k for k, v in states_dict_2.items() if v["obj_type"] == "focused" ]) == 2
        assert len(all_focused_obj_ids) == 1
        focused_obj_id = all_focused_obj_ids[0]

        print("Scene name in grav: ", scene_name)
        for frame_id in range(placer_entrance_frame+2, len(seq.obj_by_frame)):
            if not utils.isPlacerAttached_v1(focused_obj_id, frame_id, states_dict_2):
                detached_frame = frame_id
                break

        start_frame = detached_frame
        first_exit_frame = min([frame_no for obj_id, frame_no in obj_exit_events.items() if states_dict_2[obj_id]["obj_type"] == "focused"])
        vid_len_details = (start_frame, first_exit_frame - start_frame)

    elif (len(focused_objs) == 1):
         start_frame = obj_entrance_events[focused_objs[0]]
         end_frame = obj_exit_events[focused_objs[0]]
         vid_len_details = (start_frame, end_frame - start_frame)
    else:
         common_entrance_frame = max([frame_no for obj_id, frame_no in obj_entrance_events.items() if states_dict_2[obj_id]["obj_type"] == "focused"])
         first_exit_frame = min([frame_no for obj_id, frame_no in obj_exit_events.items() if states_dict_2[obj_id]["obj_type"] == "focused"])
         vid_len_details = (common_entrance_frame, first_exit_frame - common_entrance_frame)

    return vid_len_details[0], vid_len_details[1]

def get_max_vid_len(reqd_scenes, recompute=False):
    if (recompute):
        common_vid_len = []
        for scene_name in reqd_scenes:
            expected_tracks, scene_metadata, seq, states_dict_2 = get_step_processed_out(scene_name)
            start_frame, vid_len = get_vid_start_len(scene_name, seq, states_dict_2)
            common_vid_len.append(vid_len)
        return min(common_vid_len)
 
    else:
        return MIN_VID_LEN


img_shape = (MCS_IMG_HEIGHT, MCS_IMG_WIDTH)
rescaled_shape = (SCALED_X, SCALED_Y)
scale = np.divide(rescaled_shape, img_shape)

scene_folder_name_init = '0000'

# empty_vals_scenes = ["grav_new6_000004_01_zz_plaus_6n"]

# reqd_scenes = get_reqd_scenes_list()
reqd_scenes = get_occ_scenes_list()
max_vid_len = get_max_vid_len(reqd_scenes, False)

scenes_generated = 0

print('reqd_scenes',reqd_scenes)

occ_1_name = []
occ_2_name = []
occ_3_name = []

scene_name_all = {}
for scene_name in reqd_scenes:

    scene_folder_path = os.path.join(MCS_ROOT_DIR, scene_name)

    # print('scene_folder_path',scene_folder_path)
    rgb_folder = os.path.join(scene_folder_path, "RGB")
    seg_mask = os.path.join(scene_folder_path, "Mask")
    depth_folder = os.path.join(scene_folder_path, "Depth")

    step_output_folder = os.path.join(scene_folder_path, "Step_Output")
    
    expected_tracks, scene_metadata, seq, states_dict_2 = get_step_processed_out(scene_name)
    # vid_len = len(seq.mask_per_frame)

    focused_objs = [k for k, v in states_dict_2.items() if v["obj_type"] == "focused"]
    # obj re-enters the scene
    if len(focused_objs) == 1:
        occ_1_name.append(scene_name)
    if len(focused_objs) == 2:
        occ_2_name.append(scene_name)
    if len(focused_objs) >= 3:
        occ_3_name.append(scene_name)
    # if not list(seq.obj_by_frame.keys()) == list(range(min(seq.obj_by_frame.keys()), max(seq.obj_by_frame.keys())+1)): continue
    
    # vid_start_frame, trimmed_vid_len = get_vid_start_len(scene_name, seq, states_dict_2)
print("len(occ_1_name)",len(occ_1_name))
print("len(occ_2_name)",len(occ_2_name))
print("len(occ_3_name)",len(occ_3_name))

if len(occ_1_name)!=0:
    with open('./occ_1_name.json', 'w') as f:
        json.dump(occ_1_name, f)

if len(occ_2_name)!=0:
    with open('./occ_2_name.json', 'w') as f:
        json.dump(occ_2_name, f)

if len(occ_3_name)!=0:
    with open('./occ_3_name.json', 'w') as f:
        json.dump(occ_3_name, f)