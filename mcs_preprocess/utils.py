import copy
import json
import os
import pickle
from collections import OrderedDict
from collections import defaultdict

import cv2
import numpy as np
from loguru import logger
from scipy import ndimage

from objects import *

DELTA_DISTANCE = 50
DELTA_STATIONARY = 0.5

def distance_between_two_points(left, right):
    return sum((l - r) ** 2 for l, r in zip(left, right)) ** 0.5

def save_bbox_result_mcs(seg_label, cate_label, cate_score):
    bboxes = [None] * seg_label.shape[0]
    for idx in range(seg_label.shape[0]):
        cur_mask, cur_cate, cur_score = (
            seg_label[idx, :, :],
            cate_label[idx],
            cate_score[idx],
        )
        obj_slice = ndimage.measurements.find_objects(cur_mask)[0]
        y0, y1 = obj_slice[0].start, obj_slice[0].stop
        x0, x1 = obj_slice[1].start, obj_slice[1].stop
        ht, wd = (y1 - y0), (x1 - x0)
        # fixed to not follow MOT format
        # x0, y0    = x0 + 1, y0 + 1
        bboxes[idx] = [x0, y0, wd, ht]

    return np.asarray(bboxes)


def process_step(mask_rgbI=None):
    # TODO(Mazen): change this to remove everything that is large
    maskI = (
        mask_rgbI[..., 0] * 1e6 + mask_rgbI[..., 1] * 1e3 + mask_rgbI[..., 2]
    )
    mask_vals = np.unique(maskI)
    rpl_dict = {val: k for k, val in enumerate(mask_vals)}
    maskI = np.vectorize(rpl_dict.get)(maskI)
    maskI_onehot = np.eye(len(mask_vals))[maskI].transpose(2, 0, 1)
    # # -- assumption 2: floor and wall_back has the longest image boundary pixel
    # if maskI_onehot.shape[0] == 2:
    #     FG_maskI, seg_bboxes = None, None
    # else:
    #     boundaryI = np.zeros(maskI_onehot.shape[1:])[None, ...]
    #     boundaryI[0, 0, :] = 1  # top
    #     boundaryI[0, -1, :] = 1  # bot
    #     boundaryI[0, :, 0] = 1  # lft
    #     boundaryI[0, :, -1] = 1  # rht

    #     edge_size = (maskI_onehot * boundaryI).sum((1, 2))
    #     idxes = sorted(
    #         range(maskI_onehot.shape[0]), key=lambda i: -edge_size[i]
    #     )
    #     # NOTE(Mazen): remove largest two bounding boxes
    #     FG_maskI = maskI_onehot[idxes[2:], ...].astype(np.uint8)

    #     num_obj = FG_maskI.shape[0]
    #     seg_bboxes = save_bbox_result_mcs(
    #         FG_maskI, [0] * num_obj, [1.0] * num_obj
    #     )

    # NOTE(Mazen): keep everything and do removal afterward
    boundaryI = np.zeros(maskI_onehot.shape[1:])[None, ...]
    boundaryI[0, 0, :] = 1  # top
    boundaryI[0, -1, :] = 1  # bot
    boundaryI[0, :, 0] = 1  # lft
    boundaryI[0, :, -1] = 1  # rht

    edge_size = (maskI_onehot * boundaryI).sum((1, 2))
    idxes = sorted(range(maskI_onehot.shape[0]), key=lambda i: -edge_size[i])
    FG_maskI = maskI_onehot[idxes[:], ...].astype(np.uint8)

    num_obj = FG_maskI.shape[0]
    seg_bboxes = save_bbox_result_mcs(FG_maskI, [0] * num_obj, [1.0] * num_obj)

    return {"bboxes": seg_bboxes, "masks": FG_maskI}


def what_is_below_me(pole, others, frame_num):
    pole_bbox = pole["content"][frame_num]["2dbbox"]
    below_point = [
        pole_bbox[0] + pole_bbox[2] / 2,
        pole_bbox[1] + pole_bbox[3],
    ]
    closest_idx, closest_other = min(
        others.items(),
        key=lambda other: distance_between_two_points(
            [
                other[1]["content"][frame_num]["2dbbox"][0]
                + other[1]["content"][frame_num]["2dbbox"][2] / 2,
                other[1]["content"][frame_num]["2dbbox"][1],
            ],
            below_point,
        ),
    )
    closest_bbox = closest_other["content"][frame_num]["2dbbox"]
    closest_top_point = [
        closest_bbox[0] + closest_bbox[2] / 2,
        closest_bbox[1],
    ]

    distance = distance_between_two_points(closest_top_point, below_point)

    # TODO(Mazen): discuss with Chanho about the radius of change
    # an issue happened when the object is asysmetric and the placer was shifted
    # to either side
    if (
        distance > DELTA_DISTANCE
        or abs(below_point[1] - closest_top_point[1]) > DELTA_DISTANCE
    ):
        return None, None
    return closest_idx, closest_other


def what_is_left_me(pole, others, frame_num):
    pole_bbox = pole["content"][frame_num]["2dbbox"]
    right_point = [
        pole_bbox[0] + pole_bbox[2],
        pole_bbox[1] + pole_bbox[3] / 2,
    ]
    closest_idx, closest_other = min(
        others.items(),
        key=lambda other: distance_between_two_points(
            [
                other[1]["content"][frame_num]["2dbbox"][0],
                other[1]["content"][frame_num]["2dbbox"][1]
                + other[1]["content"][frame_num]["2dbbox"][3] / 2,
            ],
            right_point,
        ),
    )

    return closest_idx, closest_other


def what_is_right_me(pole, others, frame_num):
    pole_bbox = pole["content"][frame_num]["2dbbox"]
    left_point = [pole_bbox[0], pole_bbox[1] + pole_bbox[3] / 2]
    closest_idx, closest_other = min(
        others.items(),
        key=lambda other: distance_between_two_points(
            [
                other[1]["content"][frame_num]["2dbbox"][0]
                + other[1]["content"][frame_num]["2dbbox"][2],
                other[1]["content"][frame_num]["2dbbox"][1]
                + other[1]["content"][frame_num]["2dbbox"][3] / 2,
            ],
            left_point,
        ),
    )

    return closest_idx, closest_other


def mask2bbox(scene_metadata, verbose=False):
    seq_res = {}
    for step_num, step_rgb_frame in enumerate(scene_metadata["rgb"]):
        res = process_step(scene_metadata["mask"][step_num])
        if res["bboxes"] is not None:
            bboxes = np.copy(res["bboxes"])
            masks = np.copy(res["masks"])
            # NOTE(Mazen): remove small bboxes
            keep_no_small_bboxes = np.logical_and(
                bboxes[:, 2] >= 2, bboxes[:, 3] >= 2
            )

            bboxes = bboxes[keep_no_small_bboxes]
            masks = masks[keep_no_small_bboxes]

            # NOTE(Mazen): remove background objects
            # remove ceiling
            keep_no_ceiling = np.logical_or.reduce(
                (
                    bboxes[:, 1] != 0,
                    np.logical_and.reduce(
                        (bboxes[:, 1] == 0, bboxes[:, 2] <= 400)
                    ),
                )
            )
            if verbose and np.sum(keep_no_ceiling == False):
                print("keep_no_ceiling", bboxes[keep_no_ceiling == False])
            # remove floor
            keep_no_floor = np.logical_or.reduce(
                (
                    bboxes[:, 1] + bboxes[:, 3] != 400,
                    np.logical_and.reduce(
                        (
                            bboxes[:, 1] + bboxes[:, 3] == 400,
                            bboxes[:, 2] <= 400,
                        )
                    ),
                )
            )
            if verbose and np.sum(keep_no_floor == False):
                print("keep_no_floor", bboxes[keep_no_floor == False])
            keep_no_floor_and_ceiling = np.logical_and.reduce(
                (
                    keep_no_ceiling,
                    keep_no_floor,
                )
            )
            bboxes = bboxes[keep_no_floor_and_ceiling]
            masks = masks[keep_no_floor_and_ceiling]
            # remove left wall
            keep_no_left_wall = np.logical_or.reduce(
                (
                    bboxes[:, 0] != 0,
                    np.logical_and.reduce(
                        (bboxes[:, 0] == 0, bboxes[:, 3] <= 100)
                    ),
                )
            )
            if verbose and np.sum(keep_no_left_wall == False):
                print("keep_no_left_wall", bboxes[keep_no_left_wall == False])
            # remove right wall
            keep_no_right_wall = np.logical_or.reduce(
                (
                    bboxes[:, 0] + bboxes[:, 2] != 600,
                    np.logical_and.reduce(
                        (
                            bboxes[:, 0] + bboxes[:, 2] == 600,
                            bboxes[:, 3] <= 100,
                        )
                    ),
                )
            )
            if verbose and np.sum(keep_no_right_wall == False):
                print(
                    "keep_no_right_wall", bboxes[keep_no_right_wall == False]
                )
            keep_no_walls = np.logical_and.reduce(
                (
                    keep_no_left_wall,
                    keep_no_right_wall,
                )
            )
            # NOTE(Mazen): assumption - if there is no left or right wall, then
            # don't remove any wall. We assume symetry
            if np.sum(keep_no_walls == False) == 2:
                bboxes = bboxes[keep_no_walls]
                masks = masks[keep_no_walls]

            # # remove back wall
            # TODO: what if an object covers back wall?
            removed_objects = res["bboxes"].shape[0] - bboxes.shape[0]
            # print(removed_objects)
            # import pdb; pdb.set_trace()
            if removed_objects == 4:
                # NOTE(Mazen): remove back-wall based on floor, ceiling, side-walls
                # expected_back_wall_position = np.array(
                #     [
                #         left_wall[0] + left_wall[2],
                #         ceiling[1] + ceiling[3],
                #         right_wall[0] - (left_wall[0] + left_wall[2]),
                #         floor[1] - (ceiling[1] + ceiling[3]),
                #     ]
                # )
                # arr = {i: bbox for i, bbox in enumerate(bboxes)}
                # keep_no_back_wall = [True] * bboxes.shape[0]
                # cloest_idx, _ = min(
                #     arr.items(),
                #     key=lambda x: distance_between_two_points(
                #         x[1], expected_back_wall_position
                #     ),
                # )

                # NOTE(Mazen): remove the biggest area
                # 1. sort based on area
                sorted_idx = np.argsort(
                    np.apply_along_axis(
                        lambda x: (-1)
                        * x[2]
                        * x[3],  # get area (-1 to get the largest)
                        1,  # sort based on row
                        bboxes,
                    )
                )
                bboxes = bboxes[sorted_idx]
                masks = masks[sorted_idx]

                # 2. remove based on largest area
                keep_no_back_wall = [True] * bboxes.shape[0]
                keep_no_back_wall[0] = False
                if verbose and np.sum(keep_no_back_wall == False):
                    print(
                        "keep_no_back_wall", bboxes[keep_no_back_wall == False]
                    )

                bboxes = bboxes[keep_no_back_wall]
                masks = masks[keep_no_back_wall]

            # NOTE(Mazen): remove large objects as they could be walls
            # img_height, img_width = step_rgb_frame.shape[:2]
            # img_area = img_width * img_height

            # keep_no_large_bboxes = bboxes[:, 2] * bboxes[:, 3] / img_area < 0.2
            # if verbose and np.sum(keep_no_large_bboxes == False):
            #     print(
            #         "keep_no_large_bboxes",
            #         bboxes[keep_no_large_bboxes == False],
            #     )
            # bboxes = bboxes[keep_no_large_bboxes]
            # masks = masks[keep_no_large_bboxes]

            seq_res[step_num] = {"bboxes": bboxes, "masks": masks}
    return seq_res


def get_accuracy_rule_based_agent(history):
    list_matches = [int(scene["correctness"]) for scene in history.values()]
    acc = sum(list_matches) / len(list_matches)
    return acc


def get_scene_metadata(
    scene_path, scene_name, store_path, load=False, save=False
):
    """Returns RGB, Mask, and Depth for every frame in a scene
    Args:
        scene_path (str): Scene path
        scene_name (str): Scene name
        store_path (str): Where to store metadata in the pickle file
        load (bool, optional): Whether to load saved metadata from pickle file or not. Defaults to False.
        save (bool, optional): Whether to save the metadata or not. Defaults to False.
    Returns:
        scene_metadata (dict):  RGB, Mask, and Depth for every frame in a scene
            scene_metdata = {
                "rgb": [w x h x 3],
                "mask": [w x h x 3],
                "depth": [w x h x 1]
            }
    """
    scene_metadata = None
    if not load:
        scene_metadata = {
            "rgb": [],
            "mask": [],
            "depth": [],
            "direct_rgb": [],
            "direct_mask": [],
        }
        num_frames = len(os.listdir(f"{scene_path}/RGB"))
        for frame_num in range(1, num_frames + 1):
            rgb_image = cv2.cvtColor(
                cv2.imread(f"{scene_path}/RGB/{frame_num:06n}.png"),
                cv2.COLOR_BGR2RGB,
            )
            rgb_image_direct = cv2.imread(
                f"{scene_path}/RGB/{frame_num:06n}.png"
            )
            mask_image = cv2.cvtColor(
                cv2.imread(f"{scene_path}/Mask/{frame_num:06n}.png"),
                cv2.COLOR_BGR2RGB,
            )
            depth_image = cv2.imread(f"{scene_path}/Depth/{frame_num:06n}.png")
            mask_image_direct = cv2.imread(
                f"{scene_path}/Mask/{frame_num:06n}.png"
            )
            scene_metadata["rgb"].append(rgb_image)
            scene_metadata["mask"].append(mask_image)
            scene_metadata["depth"].append(depth_image)
            scene_metadata["direct_rgb"].append(rgb_image_direct)
            scene_metadata["direct_mask"].append(mask_image_direct)

        if save:
            with open(
                f"{store_path}/{scene_name}_scene_metadata.pkl", "wb"
            ) as f:
                pickle.dump(scene_metadata, f)

    else:
        with open(f"{store_path}/{scene_name}_scene_metadata.pkl", "rb") as f:
            scene_metadata = pickle.load(f)

    return scene_metadata


def get_bbox(selected_mask):
    selected_area = np.where(selected_mask, (1), (0))
    min_x = np.min(np.where(selected_area)[1])
    min_y = np.min(np.where(selected_area)[0])
    max_x = np.max(np.where(selected_area)[1])
    max_y = np.max(np.where(selected_area)[0])
    # +1 because width and height are exclusive
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    return [min_x, min_y, w, h]


def get_center_3d_bbox(dimensions):
    x_3d_dims = [dimensions[i]["x"] for i in range(len(dimensions))]
    y_3d_dims = [dimensions[i]["y"] for i in range(len(dimensions))]
    z_3d_dims = [dimensions[i]["z"] for i in range(len(dimensions))]

    x_center_3d = sum(x_3d_dims) / len(x_3d_dims)
    y_center_3d = sum(y_3d_dims) / len(y_3d_dims)
    z_center_3d = sum(z_3d_dims) / len(z_3d_dims)

    return {"x": x_center_3d, "y": y_center_3d, "z": z_center_3d}


def prepare_obj_data(mask_img, obj):
    selected_mask = np.logical_and.reduce(
        (
            mask_img[:, :, 0] == obj["segment_color"]["r"],
            mask_img[:, :, 1] == obj["segment_color"]["g"],
            mask_img[:, :, 2] == obj["segment_color"]["b"],
        )
    )
    if np.sum(selected_mask) == 0:
        return None

    min_x, min_y, w, h = get_bbox(selected_mask)

    return {
        "2dbbox": [min_x, min_y, w, h],
        "segment_color": {
            "r": obj["segment_color"]["r"],
            "g": obj["segment_color"]["g"],
            "b": obj["segment_color"]["b"],
        },
    }


def get_object_list(
    data, frame_num, mask_img, tracking_results, provide_shape
):
    for obj_name, obj in data["object_list"].items():
        assert all(
            [
                not background_obj in obj_name
                for background_obj in [
                    "floor",
                    "wall_back",
                    "wall_front",
                    "wall_left",
                    "wall_right",
                ]
            ]
        ), "expected focused object, got background"
        selected_mask = np.logical_and.reduce(
            (
                mask_img[:, :, 0] == obj["segment_color"]["r"],
                mask_img[:, :, 1] == obj["segment_color"]["g"],
                mask_img[:, :, 2] == obj["segment_color"]["b"],
            )
        )
        if np.sum(selected_mask) == 0:
            continue

        shape = obj["shape"]
        texture_color_list = obj["texture_color_list"]
        shape_str = (
            shape.replace(",", "---").replace("_", "--").replace(" ", "-")
        )
        textures_str = "_".join(
            [
                texture_color.replace(",", "---")
                .replace("_", "--")
                .replace(" ", "-")
                for texture_color in texture_color_list
            ]
        )
        obj_unique_name = f"{obj_name}_{shape_str}_{textures_str}"

        if obj_unique_name not in tracking_results:
            role = "focused"
            assert not (
                "occluder" in obj_name
            ), "expected focused object, got occluder"
            assert not (
                "placer" in obj_name
            ), "expected focused object, got placer"
            tracking_results[obj_unique_name] = {
                "content": {},
                "role": role,
                "is_pole": False,
            }
            if provide_shape:
                tracking_results[obj_unique_name].update(
                    {
                        "shape": shape,
                        "texture": texture_color_list,
                        # NOTE(Mazen): agent doesn't have any color
                        "color": texture_color_list[0]
                        if texture_color_list != []
                        else None,
                    }
                )
        obj_data = prepare_obj_data(mask_img, obj)
        if obj_data == None:
            continue
        tracking_results[obj_unique_name]["content"][frame_num] = obj_data
        if shape == "chest":
            if "open_at" not in tracking_results[obj_unique_name]:
                tracking_results[obj_unique_name].update(
                    {
                        "open_at": None,
                    }
                )
            if (
                obj["is_open"]
                and tracking_results[obj_unique_name]["open_at"] == None
            ):
                tracking_results[obj_unique_name]["open_at"] = frame_num

    return tracking_results


def get_structural_object_list(
    data, frame_num, mask_img, tracking_results, provide_shape
):
    for obj_name, obj in data["structural_object_list"].items():
        if any(
            [
                background_obj in obj_name
                for background_obj in [
                    "floor",
                    "wall_back",
                    "wall_front",
                    "wall_left",
                    "wall_right",
                    # NOTE(Mazen): additional objects for interactive tasks
                    "ceiling",
                ]
            ]
        ):
            continue
        selected_mask = np.logical_and.reduce(
            (
                mask_img[:, :, 0] == obj["segment_color"]["r"],
                mask_img[:, :, 1] == obj["segment_color"]["g"],
                mask_img[:, :, 2] == obj["segment_color"]["b"],
            )
        )
        if np.sum(selected_mask) == 0:
            continue

        shape = obj["shape"]
        if shape == "structural":
            shape = "_".join(obj_name.split("_")[:-1])
            if shape == "":
                shape = "support"

        texture_color_list = obj["texture_color_list"]
        shape_str = (
            shape.replace(",", "---").replace("_", "--").replace(" ", "-")
        )

        # NOTE(Mazen): we don't care about tracking of placers when they change color
        # textures_str = "_".join(
        #     [
        #         texture_color.replace(",", "---")
        #         .replace("_", "--")
        #         .replace(" ", "-")
        #         for texture_color in texture_color_list
        #     ]
        # )
        # obj_unique_name = f"{obj_name}_{shape_str}_{textures_str}"
        obj_unique_name = f"{obj_name}_{shape_str}"

        if obj_unique_name not in tracking_results:
            role = "non-focused"
            if shape == "support":
                role = "focused"
            is_pole = False
            if "occluder" in obj_name:
                if "pole" in obj_name:
                    is_pole = True
                elif "wall" not in obj_name:
                    logger.error("unknown occluder")
                    exit()
            elif "placer" in obj_name:
                is_pole = True

            tracking_results[obj_unique_name] = {
                "content": {},
                "role": role,
                "is_pole": is_pole,
            }
            if provide_shape:
                tracking_results[obj_unique_name].update(
                    {"shape": shape, "texture": texture_color_list}
                )
        obj_data = prepare_obj_data(mask_img, obj)
        if obj_data == None:
            continue
        tracking_results[obj_unique_name]["content"][frame_num] = obj_data
    return tracking_results

def get_metadata_from_pipleine(tracking_results, scene_metadata, vid_len):

    seq = Sequence()
    for f in range(1, vid_len + 1):
        objs = data_adaptor(tracking_results, f)

        seq.add_mask_for_frame(f, scene_metadata["mask"][f - 1])
        seq.add_depth_for_frame(f, scene_metadata["depth"][f - 1])

        for obj in objs:
            assert f == obj.frame
            seq.add_obj_by_frame(obj.frame, obj)
            seq.add_obj_by_id(obj.obj_id, obj)

    return seq

def get_tracklets(
    scene_path,
    scene_name,
    store_path,
    load=False,
    save=False,
    provide_shape=False,
):
    """Returns ground truth tracklets and tracklets information
    Args:
        scene_path (str): Scene path
        scene_name (str): Scene name
        store_path (str): Where to store metadata in the pickle file
        load (bool, optional): Whether to load saved metadata from pickle file or not. Defaults to False.
        save (bool, optional): Whether to save the metadata or not. Defaults to False.
    Returns:
        new_tracking_results (dict):
            new_tracking_results = {
                obj_id (int): {
                    "content":{
                        frame_num (int): {
                            "2dbbox": [min_x, min_y, width, height],
                            "segment_color": [],
                        },
                        ...
                    }
                    "role" (str): # foucsed, occluder-wall, support, placer
                },
                ...
            }
        vid_len (int): Number of frames in a scene
    """
    step_output_dir = f"{scene_path}/Step_Output"
    vid_len = len(
        [
            json_file
            for json_file in os.listdir(step_output_dir)
            if "json" in json_file
        ]
    )
    new_tracking_results = None
    if not load:
        tracking_results = {}
        for step_num in range(1, vid_len + 1):
            json_file = f"{step_output_dir}/step_{step_num:06d}.json"
            mask_img = cv2.cvtColor(
                cv2.imread(f"{scene_path}/Mask/{step_num:06d}.png"),
                cv2.COLOR_BGR2RGB,
            )
            f = open(json_file)
            step_output = json.load(f)
            f.close()

            get_object_list(
                step_output,
                step_num,
                mask_img,
                tracking_results,
                provide_shape,
            )
            get_structural_object_list(
                step_output,
                step_num,
                mask_img,
                tracking_results,
                provide_shape,
            )
        new_tracking_results = {}
        for obj_id, (obj_name, obj) in enumerate(tracking_results.items()):
            new_tracking_results[obj_id + 1] = copy.deepcopy(obj)

        # add is_stationary key 
        bottom_points_objs_steps = {}
        for track_id, track in new_tracking_results.items():
            bottom_points_objs_steps[track_id] = {}
            for frame_num, content in track["content"].items():
                bottom_points_objs_steps[track_id][frame_num] = (
                    content["2dbbox"][0] + content["2dbbox"][2] / 2,
                    content["2dbbox"][1] + content["2dbbox"][3],
                )
        # get the difference between frames
        diff_bp = OrderedDict(
            {
                k: [
                    (
                        abs(v[frame_num][0] - v[frame_num - 1][0]),
                        abs(v[frame_num][1] - v[frame_num - 1][1]),
                    )
                    for frame_num in v.keys()
                    if frame_num - 1 in v.keys()
                ]
                for k, v in bottom_points_objs_steps.items()
            }
        )
        for track_id, track in new_tracking_results.items():
            diff_bp_x = [diff_bpp[0] for diff_bpp in diff_bp[track_id]]
            diff_bp_y = [diff_bpp[1] for diff_bpp in diff_bp[track_id]]
            avg_diff_bp = (
                sum(diff_bp_x) / len(diff_bp_x),
                sum(diff_bp_y) / len(diff_bp_y),
            )
            # NOTE(Mazen): Assumption - a chest is stationary based on the lower box position
            stationary = (
                True
                if (avg_diff_bp[0] < DELTA_STATIONARY and avg_diff_bp[1] < DELTA_STATIONARY)
                else False
            )
            new_tracking_results[track_id]["is_stationary"] = stationary

        if save:
            with open(
                f"{store_path}/{scene_name}_tracking_results.pkl", "wb"
            ) as f:
                pickle.dump(new_tracking_results, f)
    else:
        with open(
            f"{store_path}/{scene_name}_tracking_results.pkl", "rb"
        ) as f:
            new_tracking_results = pickle.load(f)
    return new_tracking_results, vid_len


def isPlacerAttached_v1(currObjId, curr_frame_no, states_dict):
    for objId, _ in states_dict.items():
        if objId != currObjId:

            if states_dict[objId]["bbox_width_height"][curr_frame_no] == None:
                continue

            if states_dict[objId]["is_pole"]:
                placer_objId = objId

                placer_top_y_coord = states_dict[placer_objId]["2dbbox"][
                    curr_frame_no
                ][1]
                placer_height = states_dict[placer_objId]["2dbbox"][
                    curr_frame_no
                ][3]

                placer_bottom_coord = placer_top_y_coord + placer_height
                fobj_top_coord = states_dict[currObjId]["2dbbox"][
                    curr_frame_no
                ][1]

                return abs(placer_bottom_coord - fobj_top_coord) <= 5



def data_adaptor(tracking_results, f):
    objs = []

    for unq_obj_id, unq_obj_details in tracking_results.items():
        for frameID, frameInfo in unq_obj_details["content"].items():
            if f == frameID:
                data = {
                    unq_obj_id: {
                        "2dbbox": frameInfo["2dbbox"],
                        # 'dimensions': frameInfo['center_3d_bbox'],
                        "segment_color": frameInfo["segment_color"],
                        "frame": f,
                        "obj_id": unq_obj_id,
                        "position": None,
                        "rotation": None,
                        "role": unq_obj_details["role"],
                        "is_pole": unq_obj_details["is_pole"],
                        "is_stationary": unq_obj_details["is_stationary"]
                    }
                }
                obj = Object()
                obj.set_object_info(data, f, unq_obj_id)

                objs.append(obj)
    return objs



def preprocessing(vid_len, seq, scene_metadata):

    states_dict = createObjStateDict(seq.obj_by_frame, seq.obj_by_id, vid_len)

    # com_dict = getCenterOfMassState(
    #     seq.obj_by_frame, seq.obj_by_id, vid_len, seq.mask_per_frame
    # )
    # for objID in states_dict.keys():
    #     states_dict[objID]["com"] = com_dict[objID]["com"]
    #     states_dict[objID]["bbox_width_height"] = com_dict[objID][
    #         "bbox_width_height"
    #     ]

    return states_dict


def fix(f):
    return lambda *args, **kwargs: f(fix(f), *args, **kwargs)

def createObjStateDict(obj_by_frame, obj_by_id, vid_len):
    state_dict = fix(defaultdict)()

    for unq_obj_id in obj_by_id.keys():
        state_dict[unq_obj_id]["obj_type"] = obj_by_id[unq_obj_id][0].obj_type
        state_dict[unq_obj_id]["is_pole"] = obj_by_id[unq_obj_id][0].is_pole
        state_dict[unq_obj_id]["is_stationary"] = obj_by_id[unq_obj_id][0].is_stationary
        obj_initial_frame = obj_by_id[unq_obj_id][0].frame
        obj_final_frame = obj_by_id[unq_obj_id][-1].frame

        for f in range(1, vid_len + 1):
            if f in obj_by_frame:
                currFrame = obj_by_frame[f]
                if (
                    any(unq_obj_id == obj.obj_id for obj in currFrame)
                    or f >= obj_initial_frame
                    and f <= obj_final_frame
                ):
                    state_dict[unq_obj_id]["obj_exists"][f] = True
                else:
                    state_dict[unq_obj_id]["obj_exists"][f] = False

                if (
                    any(unq_obj_id == obj.obj_id for obj in currFrame)
                    and f >= obj_initial_frame
                    and f <= obj_final_frame
                ):
                    state_dict[unq_obj_id]["is_occluded"][f] = False
                elif f <= obj_initial_frame or f >= obj_final_frame:
                    state_dict[unq_obj_id]["is_occluded"][f] = None
                else:
                    state_dict[unq_obj_id]["is_occluded"][f] = True

                if (
                    any(unq_obj_id == obj.obj_id for obj in currFrame)
                    and f >= obj_initial_frame
                    and f <= obj_final_frame
                ):
                    for frameObj in currFrame:
                        if frameObj.obj_id == unq_obj_id:

                            state_dict[unq_obj_id]["2dbbox"][
                                f
                            ] = frameObj.bbox_2d

                            break
                else:
                    state_dict[unq_obj_id]["2dbbox"][f] = None
            else:
                state_dict[unq_obj_id]["obj_exists"][f] = False
                state_dict[unq_obj_id]["is_occluded"][f] = None
                state_dict[unq_obj_id]["2dbbox"][f] = None

    return state_dict