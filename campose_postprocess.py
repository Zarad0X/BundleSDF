#!/usr/bin/env python3
"""
Read a dataset folder produced by BundleSDF output and produce a JSON with frames and camera intrinsics.

Usage:
    python campose_postprocess.py /path/to/output/2022-11-18-15-10-24_milk --out camera.json

Assumptions:
 - `cam_K.txt` is a 3x3 matrix (whitespace-separated)
 - Each frame folder /<frame_id>/keyframes.yml contains a mapping with key `cam_in_ob` which is a 16-element list (row-major 4x4)
 - image files are under `color/` and masks under `mask/` with matching names; depth under `depth/` (if present)
"""
import argparse
import json
import os
from pathlib import Path
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
    for p in scene_dir.iterdir():
        if p.is_dir() and p.name.isdigit():
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', help='path to scene folder e.g. output/*/')
    p.add_argument('--out', default='camera.json', help='output json file')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f'scene dir not found: {data_dir}')

    camk_path = data_dir / 'cam_K.txt'
    if not camk_path.exists():
        raise SystemExit(f'cam_K.txt not found in {data_dir}')

    K = read_cam_K(camk_path)
    fx = float(K[0,0]); fy = float(K[1,1]); cx = float(K[0,2]); cy = float(K[1,2])

    frames = []
    last_dir = get_last_frame_dir(data_dir)
    if last_dir is None:
        raise SystemExit('No numeric frame subdirectories found in scene dir')

    # read keyframes.yml in the last directory (user said the YAML is inside that subfolder)
    kf_yaml = last_dir / 'keyframes.yml'
    if not kf_yaml.exists():
        # try parent-level keyframes.yml as fallback
        kf_yaml = data_dir / 'keyframes.yml'
        if not kf_yaml.exists():
            raise SystemExit(f'No keyframes.yml found in {last_dir} or {data_dir}')

    keyframes = read_keyframe_yaml(kf_yaml)
    if not keyframes:
        raise SystemExit(f'No keyframe entries found in {kf_yaml}')

    # directories for assets
    color_dir = data_dir / 'color'
    mask_dir = data_dir / 'mask'
    depth_dir = data_dir / 'depth'
    # helper: find depth file preferring .png, then .npz, etc.
    def find_depth_file(scene_dir, fid):
        ddir = scene_dir / 'depth'
        exts = ['.png', '.npz', '.npy', '.exr', '.pfm', '.tif', '.tiff']
        for e in exts:
            p = ddir / (fid + e)
            if p.exists():
                return p
        # fallback: any file starting with fid
        if ddir.exists():
            gl = list(ddir.glob(f'{fid}*'))
            if gl:
                return gl[0]
        return None

    # iterate in sorted order of keyframe names
    for k in sorted(keyframes.keys()):
        pose = keyframes[k]
        # try to infer frame id from key name: keyframe_00012 -> 00012
        fid = ''.join([c for c in k if c.isdigit()]) or k

        img_name = f'{fid}.png'
        img_path = color_dir / img_name
        mask_path = mask_dir / img_name

        depth_file = find_depth_file(data_dir, fid)

        rel_img = str(img_path.relative_to(data_dir)) if img_path.exists() else str(img_path)
        rel_mask = str(mask_path.relative_to(data_dir)) if mask_path.exists() else (str(mask_path) if mask_path.exists() else '')
        rel_depth = str(depth_file.relative_to(data_dir)) if (depth_file is not None) else ''

        T = pose.tolist()
        frame_entry = {
            'file_path': rel_img,
            'mask_path': rel_mask,
            'depth_path': rel_depth,
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

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=4)

    print(f'Wrote {args.out} with {len(frames)} frames')


if __name__ == '__main__':
    main()
