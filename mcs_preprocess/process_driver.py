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
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path("/home/kashis/Desktop/Eval7/RPIN/.env"))


MCS_ROOT_DIR = os.getenv("MCS_ROOT_DIR")
STORE_PATH = os.getenv("STORE_PATH")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

MAX_OBJS = 3
SCALED_X = 384
SCALED_Y = 256

MIN_VID_LEN = 30

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

    seq = utils.get_metadata_from_pipleine(expected_tracks, scene_metadata, vid_len_details)
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


def copy_process_images(frame_id, idx, folder_path, scene_folder_name_init, depth_prefix=""):
    # TODO: Resize images to 224x224
    src = folder_path + "/" + str(frame_id).zfill(6) + ".png"
    dst = OUTPUT_DIR + "/" + scene_folder_name_init + depth_prefix
    target_file_name = str(idx).zfill(3) + ".png"
    loaded_src_img = cv2.cvtColor(
                   cv2.imread(src),
                   cv2.COLOR_BGR2RGB,
               )
    resized_img = cv2.resize(loaded_src_img, (SCALED_X, SCALED_Y))

    if not os.path.exists(str(OUTPUT_DIR) + "/" + scene_folder_name_init + depth_prefix):
       os.makedirs(str(OUTPUT_DIR) + "/" + scene_folder_name_init + depth_prefix)
    cv2.imwrite(os.path.join(dst, target_file_name), resized_img)


# def viz(bbox_new_val, frame_id, idx, rgb_folder, scene_folder_name_init):
#     # TODO: Resize images to 224x224
#     src = rgb_folder + "/" + str(frame_id).zfill(6) + ".png"
#     dst = OUTPUT_DIR + "/" + scene_folder_name_init
#     target_file_name = str(idx).zfill(3) + ".png"
#     loaded_src_img = cv2.cvtColor(
#                    cv2.imread(src),
#                    cv2.COLOR_BGR2RGB,
#                )
#     resized_img = cv2.resize(loaded_src_img, (SCALED_X, SCALED_Y))
#     print("test")


scene_folder_name_init = '0000'

# empty_vals_scenes = ["grav_new6_000004_01_zz_plaus_6n"]

reqd_scenes = get_reqd_scenes_list()
max_vid_len = get_max_vid_len(reqd_scenes, False)

scenes_generated = 0

for scene_name in reqd_scenes:

    scene_folder_path = os.path.join(MCS_ROOT_DIR, scene_name)
    rgb_folder = os.path.join(scene_folder_path, "RGB")
    seg_mask = os.path.join(scene_folder_path, "Mask")
    depth_folder = os.path.join(scene_folder_path, "Depth")
    
    expected_tracks, scene_metadata, seq, states_dict_2 = get_step_processed_out(scene_name)
    vid_len = len(seq.mask_per_frame)

    # obj re-enters the scene
    if not list(seq.obj_by_frame.keys()) == list(range(min(seq.obj_by_frame.keys()), max(seq.obj_by_frame.keys())+1)): continue
    
    vid_start_frame, trimmed_vid_len = get_vid_start_len(scene_name, seq, states_dict_2)
    if trimmed_vid_len < MIN_VID_LEN: continue

    obj_bbox_list = []
    obj_mask_list = []

    # Trim videos and create a new dir
    for idx, frame_id in enumerate(range(vid_start_frame, min(vid_start_frame + max_vid_len, vid_len))):
        
        copy_process_images(frame_id, idx, rgb_folder, scene_folder_name_init)
        copy_process_images(frame_id, idx, depth_folder, scene_folder_name_init, depth_prefix="_depth")

        temp_obj_bbox_dict = []
        temp_obj_mask_list = []
        # Get bbox and mask
        for k, v in states_dict_2.items():
            if v["obj_type"] == "focused":
                bbox_vals = v["2dbbox"][frame_id]
                
                # bbox values reshaped
                bbox_new_val = utils.bbox_scaler(scene_name, frame_id, bbox_vals, SCALED_X, SCALED_Y)
                
                temp_obj_bbox_dict.append([k, bbox_new_val[0], bbox_new_val[1], bbox_new_val[0] + bbox_new_val[2], bbox_new_val[1] + bbox_new_val[3]])
                
                bbox_send_vals = [bbox_new_val[0], bbox_new_val[1], bbox_new_val[0] + bbox_new_val[2], bbox_new_val[1] + bbox_new_val[3]]
                # viz(bbox_send_vals, frame_id, idx, rgb_folder, scene_folder_name_init)

                seg_color_frame = expected_tracks[k]["content"][frame_id]["segment_color"]
                mask_img = cv2.cvtColor(
                     cv2.imread(f"{scene_folder_path}/Mask/{frame_id:06d}.png"),
                     cv2.COLOR_BGR2RGB,
                )
                selected_mask = np.logical_and.reduce(
                (
                    mask_img[:, :, 0] == seg_color_frame["r"],
                    mask_img[:, :, 1] == seg_color_frame["g"],
                    mask_img[:, :, 2] == seg_color_frame["b"],
                )
                )
                if np.sum(selected_mask) == 0:
                    continue
                cropped_image = selected_mask[bbox_vals[1]:bbox_vals[1]+bbox_vals[3], bbox_vals[0]:bbox_vals[0]+bbox_vals[2]]
                resized_cropped_img = cv2.resize(cropped_image.astype("float32"), (28, 28))
                temp_obj_mask_list.append(resized_cropped_img)
        
        temp_obj_np = np.asarray(temp_obj_bbox_dict, dtype=np.float64)
        temp_bbox_padded_boxes = utils.padding_bboxes(temp_obj_np, MAX_OBJS)
        obj_bbox_list.append(temp_obj_bbox_dict)
        obj_mask_list.append(temp_obj_mask_list)
    
    if len(obj_bbox_list)!= 0:
        print("Scene name: ", scene_name)
        obj_bbox_np = np.asarray(obj_bbox_list, dtype=np.float64)
        bbox_dst = OUTPUT_DIR + "/" + scene_folder_name_init + "_boxes.pkl"
        pickle.dump(obj_bbox_np, open(bbox_dst, "wb"))

        obj_mask_np = np.asarray(obj_mask_list, dtype=np.float64)
        mask_dst = OUTPUT_DIR + "/" + scene_folder_name_init + "_masks.pkl"
        pickle.dump(obj_mask_np, open(mask_dst, "wb"))

        scenes_generated += 1
    else:
        # TODO: Check why flow goes here
        import pdb; pdb.set_trace()

    scene_folder_name_init = str(int(scene_folder_name_init) + 1).zfill(4)



print("SCENES GENERATED: ", scenes_generated)
