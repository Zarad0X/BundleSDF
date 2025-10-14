#!/usr/bin/env python3
"""
Read a dataset folder produced by BundleSDF output and produce a JSON with frames and camera intrinsics.

Usage:
    python campose_postprocess.py --object_name usb_01

Assumptions:
 - `cam_K.txt` is a 3x3 matrix (whitespace-separated)
 - Each frame folder /<frame_id>/keyframes.yml contains a mapping with key `cam_in_ob` which is a 16-element list (row-major 4x4)
 - image files are under `images/` and masks under `masks/` with matching names; depth under `depth/` (if present)
"""
import argparse
import json
import os
import yaml
import numpy as np


def read_cam_K(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if len(lines) < 3:
        raise ValueError(f'cam_K.txt seems too short: {path}')
    K = []
    for i in range(3):
        parts = lines[i].split()
        row = [float(p) for p in parts]
        K.append(row)
    K = np.array(K, dtype=float)
    return K


def find_frame_dirs(scene_dir):
    # frame dirs appear to be numeric timestamps; collect directories with numeric names
    dirs = []
    if not os.path.exists(scene_dir):
        return dirs
    for p_entry in os.scandir(scene_dir):
        p = p_entry.path
        if os.path.isdir(p) and os.path.basename(p).isdigit():
            dirs.append(p)
    dirs.sort()
    return dirs


def get_last_frame_dir(scene_dir):
    dirs = find_frame_dirs(scene_dir)
    if not dirs:
        return None
    return dirs[-1]


def read_keyframe_yaml(yaml_path):
    # parse yaml and return a dict: {keyframe_name: 4x4 numpy array}
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    if not data:
        return {}

    result = {}
    # expected structure: keyframe_XXXXX: { cam_in_ob: [16 numbers] }
    for k, v in data.items():
        if isinstance(v, dict) and 'cam_in_ob' in v:
            arr = v['cam_in_ob']
            if len(arr) != 16:
                raise ValueError(f'cam_in_ob length != 16 in {yaml_path} for {k}')
            M = np.array(arr, dtype=float).reshape((4,4))
            result[k] = M
        elif isinstance(v, list) and len(v) == 16 and all(isinstance(x, (int, float)) for x in v):
            # fallback: the key maps directly to a 16-list
            result[k] = np.array(v, dtype=float).reshape((4,4))

    # If result still empty, try to search nested dicts for cam_in_ob entries
    if not result:
        def find_all(obj, prefix=''):
            if isinstance(obj, dict):
                for kk, vv in obj.items():
                    if isinstance(vv, dict) and 'cam_in_ob' in vv:
                        arr = vv['cam_in_ob']
                        if len(arr) == 16:
                            name = prefix + str(kk)
                            result[name] = np.array(arr, dtype=float).reshape((4,4))
                    else:
                        find_all(vv, prefix + str(kk) + '/')

        find_all(data)

    return result


def read_poses_txt(txt_path):
    """Read a text file containing row-major 4x4 matrices stacked (4 lines per matrix or whitespace separated)
    Return a list of 4x4 numpy arrays in order.
    """
    with open(txt_path, 'r') as f:
        txt = f.read()
    import re
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", txt)
    vals = [float(x) for x in nums]
    if len(vals) % 16 != 0:
        raise ValueError(f'poses_after_nerf.txt does not contain a multiple of 16 floats: {txt_path} (found {len(vals)})')
    mats = []
    for i in range(0, len(vals), 16):
        arr = np.array(vals[i:i+16], dtype=float).reshape((4,4))
        mats.append(arr)
    return mats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--object_name', type=str, required=True, help='name of the object/scene, e.g. usb_01')
    args = p.parse_args()

    # Dataset dir used for intrinsics and image/mask/depth lookup
    dataset_dir = os.path.join('datasets', args.object_name)
    camk_path = os.path.join(dataset_dir, 'cam_K.txt')

    # Choose output root: prefer 'datasets', fallback to 'outputs'
    root_out_dir = 'datasets'

    # Iterate two parts
    for part_idx in range(2):
        scene_root = os.path.join(root_out_dir, args.object_name, "BundleSDF", f'part_{part_idx:01d}')

        # Resolve intrinsics K
        K = None
        if os.path.exists(camk_path):
            K = read_cam_K(camk_path)
        fx = float(K[0,0]); fy = float(K[1,1]); cx = float(K[0,2]); cy = float(K[1,2])

        frames = []
        last_dir = get_last_frame_dir(scene_root)

        # Prefer poses_after_nerf.txt: check last numeric dir first, otherwise search other numeric dirs (from newest to oldest)
        poses_txt = os.path.join(last_dir, 'poses_after_nerf.txt')
        keyframes = {}
        used_poses_dir = None
        if not os.path.exists(poses_txt):
            # search other numeric dirs (newest first)
            for d in reversed(find_frame_dirs(scene_root)):
                ptxt = os.path.join(d, 'poses_after_nerf.txt')
                if os.path.exists(ptxt):
                    poses_txt = ptxt
                    used_poses_dir = d
                    break

        # if os.path.exists(poses_txt):
        #     mats = read_poses_txt(poses_txt)
        #     for i, M in enumerate(mats):
        #         name = f'keyframe_{i:05d}'
        #         keyframes[name] = M
        # else:
        # read keyframes.yml in the last directory (user said the YAML is inside that subfolder)
        kf_yaml = os.path.join(last_dir, 'keyframes.yml')
        if not os.path.exists(kf_yaml):
            # try parent-level keyframes.yml as fallback
            kf_yaml = os.path.join(scene_root, 'keyframes.yml')
            if not os.path.exists(kf_yaml):
                # nothing to do for this part
                continue
        keyframes = read_keyframe_yaml(kf_yaml)

        # iterate in sorted order of keyframe names, infer fid from key name (e.g., keyframe_00012 -> 00012)
        for k in sorted(keyframes.keys()):
            pose = keyframes[k]
            fid = ''.join([c for c in k if c.isdigit()]) or k

            T = pose.tolist()
            frame_entry = {
                'file_path': f"images/{fid}.png",
                'mask_path': f"mask_{part_idx:01d}/{fid}.png",
                'depth_path': f"depth/{fid}.npz",
                'transform_matrix': T,
                'fl_x': fx,
                'fl_y': fy,
                'cx': cx,
                'cy': cy,
                'w': int(2*cx),
                'h': int(2*cy),
                'k1': 0.0,
                'k2': 0.0,
                'p1': 0.0,
                'p2': 0.0,
            }
            frames.append(frame_entry)

        out = {
            'camera_model': 'PINHOLE',
            'frames': frames
        }

        # ensure part directory exists and write camera.json there
        out_path = os.path.join(dataset_dir, f'transforms_{part_idx}.json')
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=4)

        print(f'BundleSDF Post-process with {len(frames)} frames of part {part_idx}')


if __name__ == '__main__':
    main()
