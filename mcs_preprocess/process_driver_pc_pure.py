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
import torch
# import open3d as o3d
import matplotlib.pyplot as plt
# from dotenv import load_dotenv

# load_dotenv(dotenv_path=Path("/home/kashis/Desktop/Eval7/RPIN/.env"))
MCS_ROOT_DIR = './after_eval'
STORE_PATH = os.getenv("STORE_PATH")
# OUTPUT_DIR = os.getenv("OUTPUT_DIR")
# OUTPUT_DIR = os.getenv("OUTPUT_DIR")
OUTPUT_DIR = './after_eval_process_pc'

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


def copy_process_images(frame_id, idx, folder_path, scene_folder_name_init, depth_prefix=""):
    # TODO: Resize images to 224x224
    src = folder_path + "/" + str(frame_id).zfill(6) + ".png"
    dst = OUTPUT_DIR + "/" + scene_folder_name_init + depth_prefix
    target_file_name = str(idx).zfill(3) + ".png"
    loaded_src_img = cv2.cvtColor(
                   cv2.imread(src),
                   cv2.COLOR_BGR2RGB,
               )
    # print('loaded_src_img',loaded_src_img.shape)
    # resized_img = cv2.resize(loaded_src_img, (SCALED_X, SCALED_Y))

    # if depth_prefix != "_depth":
    #     resized_img =  cv2.cvtColor(
    #                resized_img,
    #                cv2.COLOR_RGB2BGR,
    #            )

    # if not os.path.exists(str(OUTPUT_DIR) + "/" + scene_folder_name_init + depth_prefix):
    #    os.makedirs(str(OUTPUT_DIR) + "/" + scene_folder_name_init + depth_prefix)
    # cv2.imwrite(os.path.join(dst, target_file_name), resized_img)

    return loaded_src_img.copy()


def occulder_process_images(input, selected_mask):
    # get rgb and depth of image
    input_new = input * selected_mask[...,np.newaxis]
    
    return input_new


def occulder_save_images(input, idx, scene_folder_name_init, depth_prefix=""):
    # save rgb and depth of image
    dst = OUTPUT_DIR + "/" + scene_folder_name_init + depth_prefix
    target_file_name = str(idx).zfill(3) + ".png"

    resized_img = cv2.resize(input, (SCALED_X, SCALED_Y))

    if depth_prefix == "_occulder":
        resized_img =  cv2.cvtColor(
                   resized_img,
                   cv2.COLOR_RGB2BGR,
               )

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

img_shape = (MCS_IMG_HEIGHT, MCS_IMG_WIDTH)
rescaled_shape = (SCALED_X, SCALED_Y)
scale = np.divide(rescaled_shape, img_shape)

scene_folder_name_init = '0000'

# empty_vals_scenes = ["grav_new6_000004_01_zz_plaus_6n"]

reqd_scenes = get_reqd_scenes_list()
max_vid_len = get_max_vid_len(reqd_scenes, False)

scenes_generated = 0

print('reqd_scenes',reqd_scenes)

scene_name_all = {}
for scene_name in reqd_scenes:

    scene_folder_path = os.path.join(MCS_ROOT_DIR, scene_name)

    # print('scene_folder_path',scene_folder_path)
    rgb_folder = os.path.join(scene_folder_path, "RGB")
    seg_mask = os.path.join(scene_folder_path, "Mask")
    depth_folder = os.path.join(scene_folder_path, "Depth")

    step_output_folder = os.path.join(scene_folder_path, "Step_Output")
    
    expected_tracks, scene_metadata, seq, states_dict_2 = get_step_processed_out(scene_name)
    vid_len = len(seq.mask_per_frame)

    # obj re-enters the scene
    if not list(seq.obj_by_frame.keys()) == list(range(min(seq.obj_by_frame.keys()), max(seq.obj_by_frame.keys())+1)): continue
    
    vid_start_frame, trimmed_vid_len = get_vid_start_len(scene_name, seq, states_dict_2)
    if trimmed_vid_len < MIN_VID_LEN: continue

    obj_bbox_list = []
    obj_mask_list = []

    obj_3dcenter_2d_list = []
    obj_3dcenter_real_list = []

    # obj_rgbd_pc_world_list = []
    # obj_rgbd_pc_ind_list = []
    obj_rgbd_pc_f_ind_list =[]
    obj_rgbd_pc_f_ind_help = torch.zeros((1))
    obj_rgbd_pc_f_ind_list.append(obj_rgbd_pc_f_ind_help)
    # Trim videos and create a new dir
    for idx, frame_id in enumerate(range(vid_start_frame, min(vid_start_frame + max_vid_len, vid_len))):

        # print('idx',idx,'frame_id',frame_id)
        # if idx <23:
        #     continue
        
        rgb_help=copy_process_images(frame_id, idx, rgb_folder, scene_folder_name_init)
        depth_help=copy_process_images(frame_id, idx, depth_folder, scene_folder_name_init, depth_prefix="_depth")

        temp_obj_bbox_dict = []
        temp_obj_mask_list = []
        temp_obj_3dcenter_2d_list = []
        temp_obj_3dcenter_real_list = []
        temp_obj_rgbd_pc_world_list = []
        temp_obj_rgbd_pc_ind_list = []
        temp_obj_rgbd_pc_ind_help = torch.zeros((1,1))
        temp_obj_rgbd_pc_ind_list.append(temp_obj_rgbd_pc_ind_help)

        occ_rgb_help = 0
        occ_rgb_help_mark = 0
        occ_depth_help = 0
        occ_depth_help_mark = 0

        depth_img = cv2.imread(f"{scene_folder_path}/Depth/{frame_id:06d}.png",cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.cvtColor(
                     cv2.imread(f"{scene_folder_path}/Mask/{frame_id:06d}.png"),
                     cv2.COLOR_BGR2RGB,
                )
        rgb_img = cv2.cvtColor(
                     cv2.imread(f"{scene_folder_path}/Mask/{frame_id:06d}.png"),
                     cv2.COLOR_BGR2RGB,
                )
        # if scene_name == "grav_new6_000045_01_zz_plaus_6n":
            # print(states_dict_2)
        # Get bbox and mask
        for k, v in states_dict_2.items():
            # print('k',k)
            # print('v',v)
            # if scene_name == "grav_new6_000045_01_zz_plaus_6n":
                # print(expected_tracks[k]['obj_name'])
                # print(v["obj_type"])
            if v["obj_type"] == "focused":
                bbox_vals = v["2dbbox"][frame_id]

                # if scene_name == "grav_new6_000045_01_zz_plaus_6n":
                #     print(bbox_vals)
                
                # bbox values reshaped
                bbox_new_val = utils.bbox_scaler(scene_name, frame_id, bbox_vals, SCALED_X, SCALED_Y)
                
                temp_obj_bbox_dict.append([k, bbox_new_val[0], bbox_new_val[1], bbox_new_val[0] + bbox_new_val[2], bbox_new_val[1] + bbox_new_val[3]])
                
                bbox_send_vals = [bbox_new_val[0], bbox_new_val[1], bbox_new_val[0] + bbox_new_val[2], bbox_new_val[1] + bbox_new_val[3]]
                # viz(bbox_send_vals, frame_id, idx, rgb_folder, scene_folder_name_init)

                seg_color_frame = expected_tracks[k]["content"][frame_id]["segment_color"]

                # print('mask_img',mask_img.shape)
                selected_mask = np.logical_and.reduce(
                (
                    mask_img[:, :, 0] == seg_color_frame["r"],
                    mask_img[:, :, 1] == seg_color_frame["g"],
                    mask_img[:, :, 2] == seg_color_frame["b"],
                )
                )
                if np.sum(selected_mask) == 0:
                    continue

                # print('selected_maskshape',selected_mask.shape)
                # print('sum_selected_mask',np.sum(selected_mask))
                # print('selected_mask',selected_mask)
                occ_rgb_help = occ_rgb_help + occulder_process_images(rgb_help, selected_mask)
                occ_rgb_help_mark = 1
                occ_depth_help =  occ_depth_help + occulder_process_images(depth_help, selected_mask)
                occ_depth_help_mark = 1

                cropped_image = selected_mask[bbox_vals[1]:bbox_vals[1]+bbox_vals[3], bbox_vals[0]:bbox_vals[0]+bbox_vals[2]]
                resized_cropped_img = cv2.resize(cropped_image.astype("float32"), (28, 28))
                # print(resized_cropped_img)
                temp_obj_mask_list.append(resized_cropped_img)

                json_file = os.path.join(step_output_folder, 'step_%06d.json' % frame_id)
                f = open(json_file)
                data_json = json.load(f)
                f.close()
                cam = cam_help.read_cam_params(data_json)
                objs = cam_help.read_objs_new(data_json, frame_id, expected_tracks[k]['obj_name'])
                _, temp_amodal_center, temp_3d_center = cam_help.obtain_amodal_center(objs, cam)
                # print('temp_amodal_center',temp_amodal_center)
                # print('temp_3d_center',temp_3d_center)
                # print("")
                temp_obj_3dcenter_2d_list.append([k, temp_amodal_center[0][0],temp_amodal_center[0][1],temp_amodal_center[0][2]])
                temp_obj_3dcenter_real_list.append([k,temp_3d_center[0][0],temp_3d_center[0][1],temp_3d_center[0][2]])
                # print('temp_obj_bbox_dict',temp_obj_bbox_dict)
                # print('temp_obj_3dcenter_2d_list',temp_obj_3dcenter_2d_list)
                # print('temp_obj_3dcenter_real_list',temp_obj_3dcenter_real_list)

                d_obj_tensor = torch.from_numpy((depth_img * selected_mask).astype(np.float32))
                # print('d_obj_tensor',d_obj_tensor)
                d_obj_img_pc = utils.obj_tensor_point(d_obj_tensor)
                # print("d_obj_img_pc.shape",d_obj_img_pc)
                d_obj_pc_cam = utils.obj_d_point(d_obj_tensor, cam)
                d_obj_pc_world = utils.pc_cam_to_pc_world(d_obj_pc_cam, cam)
                # print("d_obj_pc_cam.shape",d_obj_pc_cam.shape)
                # print("d_obj_pc_cam",d_obj_pc_cam)
                # print("d_obj_pc_cam",d_obj_pc_cam[:,2].max())
                

                rgb_pc=occulder_process_images(rgb_help, selected_mask)
                rgb_pc=torch.Tensor(rgb_pc[d_obj_img_pc[:,0].int(), d_obj_img_pc[:,1].int(),:])
                if len(rgb_pc.shape)==1:
                    rgb_pc=rgb_pc.unsqueeze(0)
                # utils.point_visualize(d_obj_pc_cam, rgb_pc)
                # utils.point_visualize(d_obj_pc_world, rgb_pc)
                
                # plt.imshow(rgb_help)
                # depth_img_point=utils.tensor_point(depth_img)
                # depth_img_point_cloud = utils.img_d_point(depth_img_point,cam)
                # rgb_img_point=utils.tensor_point(rgb_help)
                # xyz = np.array(depth_img_point_cloud)
                # # print('xyz.shape',xyz.shape)
                # rgb = np.array(rgb_img_point[:,2:])
                # # print('rgb.shape',rgb.shape)
                # pcd = np.concatenate((xyz, rgb), axis=1)
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=pcd[:, 3:] / 255)
                # plt.show()
                obj_rgbd_pc_world=torch.cat([d_obj_pc_cam,rgb_pc],1)
                temp_obj_rgbd_pc_world_list.append(obj_rgbd_pc_world)
                temp_obj_rgbd_pc_ind_help = temp_obj_rgbd_pc_ind_help + obj_rgbd_pc_world.shape[0]
                temp_obj_rgbd_pc_ind_list.append(temp_obj_rgbd_pc_ind_help)
                
                # assert(1==2)
        
        # if occ_rgb_help_mark !=0:
        #     occulder_save_images(occ_rgb_help, idx, scene_folder_name_init, depth_prefix="_occulder")
        # if occ_depth_help_mark !=0:
        #     occ_depth_help_new = np.where(occ_depth_help==0,255,occ_depth_help).astype("uint8")
        #     occulder_save_images(occ_depth_help_new, idx, scene_folder_name_init, depth_prefix="_occulder_depth")    

        # print('temp_obj_bbox_dict',temp_obj_bbox_dict)
        temp_obj_np = np.asarray(temp_obj_bbox_dict, dtype=np.float64)
        temp_bbox_padded_boxes = utils.padding_bboxes(temp_obj_np, MAX_OBJS)
        obj_bbox_list.append(temp_obj_bbox_dict)
        obj_mask_list.append(temp_obj_mask_list)

        # print('temp_obj_3dcenter_2d_list',temp_obj_3dcenter_2d_list)
        temp_obj_3dcenter_2d_np=np.asarray(temp_obj_3dcenter_2d_list, dtype=np.float64)
        obj_3dcenter_2d_list.append(temp_obj_3dcenter_2d_np)

        assert(len(obj_3dcenter_2d_list) == len(obj_bbox_list))
        # obj_amodal_center_all.append(amodal_center_all)
        obj_3dcenter_real_list.append(temp_obj_3dcenter_real_list)
        assert(len(obj_3dcenter_real_list) == len(obj_bbox_list))
        # print('amodal_center_all',amodal_center_all)

        temp_obj_rgbd_pc_world_ts=torch.cat(temp_obj_rgbd_pc_world_list,0).double()
        # obj_rgbd_pc_world_list.append(temp_obj_rgbd_pc_world_ts)
        temp_obj_rgbd_pc_ind_ts=torch.cat(temp_obj_rgbd_pc_ind_list,1).int()
        # obj_rgbd_pc_ind_list.append(temp_obj_rgbd_pc_ind_ts)
        # print('temp_obj_rgbd_pc_ind_ts',temp_obj_rgbd_pc_ind_ts)

        if not os.path.exists(OUTPUT_DIR + "/" + scene_folder_name_init +"_pc"+ "/"):
            os.makedirs(OUTPUT_DIR + "/" + scene_folder_name_init +"_pc"+ "/")
        temp_obj_rgbd_pc_world_dst = OUTPUT_DIR + "/" + scene_folder_name_init +"_pc"+ "/" + str(idx).zfill(3) +".pkl"
        pickle.dump(temp_obj_rgbd_pc_world_ts, open(temp_obj_rgbd_pc_world_dst, "wb"))

        if not os.path.exists(OUTPUT_DIR + "/" + scene_folder_name_init +"_pc_ind"+ "/"):
            os.makedirs(OUTPUT_DIR + "/" + scene_folder_name_init +"_pc_ind"+ "/")
        temp_obj_rgbd_pc_ind_dst = OUTPUT_DIR + "/" + scene_folder_name_init +"_pc_ind"+ "/" + str(idx).zfill(3) +".pkl"
        pickle.dump(temp_obj_rgbd_pc_ind_ts, open(temp_obj_rgbd_pc_ind_dst, "wb"))

        obj_rgbd_pc_f_ind_help = obj_rgbd_pc_f_ind_help + temp_obj_rgbd_pc_world_ts.shape[0]
        obj_rgbd_pc_f_ind_list.append(obj_rgbd_pc_f_ind_help)

    if len(obj_bbox_list)!= 0:
        print("Scene name: ", scene_name)
        # obj_bbox_np = np.asarray(obj_bbox_list, dtype=np.float64)
        # bbox_dst = OUTPUT_DIR + "/" + scene_folder_name_init + "_boxes.pkl"
        # pickle.dump(obj_bbox_np, open(bbox_dst, "wb"))

        # obj_mask_np = np.asarray(obj_mask_list, dtype=np.float64)
        # mask_dst = OUTPUT_DIR + "/" + scene_folder_name_init + "_masks.pkl"
        # pickle.dump(obj_mask_np, open(mask_dst, "wb"))

        # obj_3dcenter_np = np.asarray(obj_3dcenter_2d_list, dtype=np.float64)
        # center3d_2d_dst = OUTPUT_DIR + "/" + scene_folder_name_init + "_3dcenter_2d.pkl"
        # pickle.dump(obj_3dcenter_np, open(center3d_2d_dst, "wb"))

        obj_3dcenter_real_np = np.asarray(obj_3dcenter_real_list, dtype=np.float64)
        center3d_real_dst = OUTPUT_DIR + "/" + scene_folder_name_init + "_3dcenter_real.pkl"
        pickle.dump(obj_3dcenter_real_np, open(center3d_real_dst, "wb"))

        # obj_rgbd_pc_world_ts = torch.cat(obj_rgbd_pc_world_list, 0).double()
        # obj_rgbd_pc_world_dst = OUTPUT_DIR + "/" + scene_folder_name_init + "_rgbd_pc.pkl"
        # pickle.dump(obj_rgbd_pc_world_ts, open(obj_rgbd_pc_world_dst, "wb"))

        # obj_rgbd_pc_ind_ts = torch.cat(obj_rgbd_pc_ind_list, 0).int()
        # print('obj_rgbd_pc_ind_ts',obj_rgbd_pc_ind_ts)
        # print('obj_rgbd_pc_ind_ts',obj_rgbd_pc_ind_ts.shape)
        # obj_rgbd_pc_ind_dst = OUTPUT_DIR + "/" + scene_folder_name_init + "_pc_o_ind.pkl"
        # pickle.dump(obj_rgbd_pc_ind_ts, open(obj_rgbd_pc_ind_dst, "wb"))

        obj_rgbd_pc_f_ind_ts = torch.cat(obj_rgbd_pc_f_ind_list, 0).int()
        # print('obj_rgbd_pc_f_ind_ts',obj_rgbd_pc_f_ind_ts)
        # print('obj_rgbd_pc_f_ind_ts',obj_rgbd_pc_f_ind_ts.shape)
        obj_rgbd_pc_f_ind_dst = OUTPUT_DIR + "/" + scene_folder_name_init + "_pc_find.pkl"
        pickle.dump(obj_rgbd_pc_f_ind_ts, open(obj_rgbd_pc_f_ind_dst, "wb"))

        scenes_generated += 1
    else:
        # TODO: Check why flow goes here
        import pdb; pdb.set_trace()

    scene_name_all[scene_folder_name_init] = scene_name
    scene_folder_name_init = str(int(scene_folder_name_init) + 1).zfill(4)

with open('scene_name_dict.json', 'w') as f:
    json.dump(scene_name_all, f)

print("SCENES GENERATED: ", scenes_generated)
