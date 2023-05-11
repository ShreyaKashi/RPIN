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

mcs_root_dir = "/home/kashis/Desktop/mcs/gen_scenes/eval5_validation_set"
output_dir = "/home/kashis/Desktop/RPIN/data/MCS_1/"
STORE_PATH="/home/kashis/Desktop/RPIN/data/MCS_misc/"
output_dir_SS = "/home/kashis/Desktop/RPIN/data/MCS_SS/train/"
vid_start_end = []
MAX_OBJS = 4

def getObjEntranceEvents(seq):
        objEntranceEventList = []

        for obj_id, obj_details in seq.obj_by_id.items():
            if obj_details[0].obj_type == "focused":
                objEntranceEventList.append([obj_id, obj_details[0].frame])

        return objEntranceEventList

def getObjExitEvents(seq):
        objEntranceEventList = []

        for obj_id, obj_details in seq.obj_by_id.items():
            if obj_details[0].obj_type == "focused":
                objEntranceEventList.append([obj_id, obj_details[-1].frame])

        return objEntranceEventList

scene_folder_name_init = '0000'

for scene_name in os.listdir(mcs_root_dir):

    # Get only plaus scenes with no occluders
    if "implaus" in scene_name:
        continue

    folder_path = os.path.join(mcs_root_dir, scene_name)
    rgb_folder = os.path.join(folder_path, "RGB")
    seg_mask = os.path.join(folder_path, "Mask")
    
    step_output_folder = os.path.join(folder_path, "Step_Output")
    step_out_initial = step_output_folder + "/" + os.listdir(step_output_folder)[0]

    with open(step_out_initial) as f:
        step_out_content = json.load(f)
        if any("occluder" in key for key in step_out_content["structural_object_list"]):
            continue
    
    

    # Find number of objs in scene
    scene_path = f"{mcs_root_dir}/{scene_name}"
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
    obj_entrance_events = getObjEntranceEvents(seq)
    obj_exit_events = getObjExitEvents(seq)

    # Find start frame no and vid len
    focused_objs = [k for k, v in states_dict_2.items() if v["obj_type"] == "focused"]


    if (len(focused_objs) == 1):
         focused_obj_id = focused_objs[0]
         start_frame = obj_entrance_events[0][1]
         end_frame = obj_exit_events[0][1]
         vid_start_end.append((start_frame, end_frame))
         vid_len_details = (start_frame, end_frame - start_frame)
    else:
         common_entrance_frame = max([frame_no for obj_id, frame_no in obj_entrance_events])
         first_exit_frame = min([frame_no for obj_id, frame_no in obj_exit_events])
         vid_start_end.append((common_entrance_frame, first_exit_frame))
         vid_len_details = (common_entrance_frame, first_exit_frame - common_entrance_frame)

    # print(scene_name, ": ", vid_len)
    max_vid_len = 34



    obj_bbox_list = []
    obj_mask_list = []
    # Trim videos and create a new dir
    if not os.path.exists(str(output_dir_SS) + scene_folder_name_init):
        os.makedirs(str(output_dir_SS) + scene_folder_name_init)
        for idx, frame_id in enumerate(range(vid_len_details[0], vid_len_details[0] + max_vid_len)):
             src = rgb_folder + "/" + str(frame_id).zfill(6) + ".png"

             # resized_src_img = cv2.resize(src.astype("float32"), (224, 224))
             dst = output_dir_SS + scene_folder_name_init
             target_file_name = str(idx).zfill(3) + ".png"
            #  shutil.copyfile(resized_src_img, os.path.join(dst, target_file_name)) 

             loaded_src_img = cv2.cvtColor(
                            cv2.imread(src),
                            cv2.COLOR_BGR2RGB,
                        )
             cv2.imwrite(os.path.join(dst, target_file_name), loaded_src_img)

             temp_obj_bbox_dict = []
             temp_obj_mask_list = []
 

             # Get bbox and mask
             for k, v in states_dict_2.items():
                  if v["obj_type"] == "focused":
                       bbox_vals = v["2dbbox"][frame_id]
                       temp_obj_bbox_dict.append([k, bbox_vals[0], bbox_vals[1], bbox_vals[0] + bbox_vals[2], bbox_vals[1] + bbox_vals[3]])
                       seg_color_frame = expected_tracks[k]["content"][frame_id]["segment_color"]
                       mask_img = cv2.cvtColor(
                            cv2.imread(f"{scene_path}/Mask/{frame_id:06d}.png"),
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
                    #    plt.imshow(selected_mask)
                    #    plt.savefig(output_dir_SS + scene_folder_name_init + str(k) + str(frame_id)+'.png')
                       cropped_image = selected_mask[bbox_vals[1]:bbox_vals[1]+bbox_vals[3], bbox_vals[0]:bbox_vals[0]+bbox_vals[2]]
                       resized_cropped_img = cv2.resize(cropped_image.astype("float32"), (28, 28))
                       temp_obj_mask_list.append(resized_cropped_img)

             obj_bbox_list.append(temp_obj_bbox_dict)
             obj_mask_list.append(temp_obj_mask_list)
    
    obj_bbox_np = np.asarray(obj_bbox_list, dtype=np.float64)
    bbox_dst = output_dir_SS + scene_folder_name_init + "_boxes.pkl"
    pickle.dump(obj_bbox_np, open(bbox_dst, "wb"))

    obj_mask_np = np.asarray(obj_mask_list, dtype=np.float64)
    mask_dst = output_dir_SS + scene_folder_name_init + "_masks.pkl"
    pickle.dump(obj_mask_np, open(mask_dst, "wb"))


             

    scene_folder_name_init = str(int(scene_folder_name_init) + 1).zfill(4)


    # Convert image to npy
    # if not os.path.exists(str(output_dir)+str(scene_name)):
    #     os.makedirs(str(output_dir)+str(scene_name))


    # for file_name in os.listdir(rgb_folder):
        
    #     file_path = os.path.join(rgb_folder, file_name)
    #     image = Image.open(file_path)
    #     image_array = np.array(image)
    #     output_file = os.path.join(output_dir, scene_name, file_name.replace(".png", ".npy"))

        
    #     np.save(output_file, image_array)

    #     img_array = np.load(output_file)
    #     plt.imshow(img_array)

# print("Vid len: ", vid_start_end)
# vid_start = max([start for start, _ in vid_start_end])
# vid_end = min([end for _, end in vid_start_end])
# print(vid_start, vid_end)


