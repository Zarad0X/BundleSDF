import yaml
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D

def load_keyframes(yml_path):
    """
    从 YAML 文件中读取所有 keyframe 的 cam_in_ob 矩阵（4×4）
    返回一个 dict：{ keyframe_name : numpy 4×4 矩阵 }
    """
    with open(yml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    keyframes = {}
    for k, v in data.items():
        if 'cam_in_ob' not in v:
            print(f"Warning: keyframe {k} 没有 cam_in_ob 字段，跳过")
            continue
        arr = v['cam_in_ob']
        mat = np.array(arr).reshape(4, 4)
        keyframes[k] = mat
    return keyframes


def load_poses_from_txt(txt_path):
    """
    从 .txt 文件中读取多个连续的 4x4 pose 矩阵
    文件格式：
    每 4 行构成一个矩阵，中间无空行
    返回：dict { "pose_000": mat0, "pose_001": mat1, ... }
    """
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(lines) % 4 != 0:
        raise ValueError(f"行数 {len(lines)} 不是 4 的倍数，文件格式可能不正确")
    
    keyframes = {}
    num_poses = len(lines) // 4
    for i in range(num_poses):
        mat_lines = lines[i*4:(i+1)*4]
        mat = np.array([[float(x) for x in line.split()] for line in mat_lines])
        keyframes[f"pose_{i:03d}"] = mat
    print(f"Loaded {num_poses} poses from {txt_path}")
    return keyframes


def load_transforms_json(json_path):
    """
    从 transforms_*.json 读取 frames 中的 transform_matrix，生成 {name: 4x4}。
    name 优先使用 file_path 提取的帧号，否则使用顺序编号。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    frames = data.get('frames', [])
    keyframes = {}
    for i, fr in enumerate(frames):
        mat = np.array(fr.get('transform_matrix'), dtype=float)
        # 提取名字：优先 file_path 的数字部分
        name = f"frame_{i:05d}"
        fp = fr.get('file_path')
        if isinstance(fp, str):
            base = os.path.basename(fp)
            stem = os.path.splitext(base)[0]
            if stem.isdigit():
                name = stem
            else:
                # 尝试抽取数字子串
                digits = ''.join([c for c in stem if c.isdigit()])
                if digits:
                    name = digits
                else:
                    name = stem
        keyframes[name] = mat
    print(f"Loaded {len(keyframes)} frames from {json_path}")
    return keyframes


def set_axes_equal(ax):
    """
    使 3D 坐标系的 XYZ 坐标轴在图中看起来单位尺度一致（等比例显示）。
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])


def visualize_cameras(keyframes, output_path):
    """
    可视化所有相机位姿。
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for name, mat in keyframes.items():
        R = mat[:3, :3]
        t = mat[:3, 3]
        ax.scatter(t[0], t[1], t[2], color='r', s=30)
        ax.text(t[0], t[1], t[2], name, fontsize=6)
        scale = 0.05
        ax.quiver(
            t[0], t[1], t[2],
            R[0, 2], R[1, 2], R[2, 2],
            length=scale, color='b', arrow_length_ratio=0.3
        )

    centers = np.array([mat[:3, 3] for mat in keyframes.values()])
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
            color='gray', linestyle='--', linewidth=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Poses Visualization')
    ax.view_init(elev=20, azim=60)
    ax.grid(True)
    plt.tight_layout()
    set_axes_equal(ax)
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize camera poses from BundleSDF keyframes or transforms")
    parser.add_argument("--path", required=True, help="Path to keyframes.yml, poses_after_nerf.txt, or transforms_*.json")
    parser.add_argument("--out", default="camera_poses.png", help="Output image path (default: camera_poses.png)")
    args = parser.parse_args()

    if args.path.endswith(".yml") or args.path.endswith(".yaml"):
        keyframes = load_keyframes(args.path)
    elif args.path.endswith(".txt"):
        keyframes = load_poses_from_txt(args.path)
    elif args.path.endswith(".json"):
        keyframes = load_transforms_json(args.path)
    else:
        raise ValueError("Unsupported file type, must be .yml/.yaml, .txt or .json")

    if not keyframes:
        print("No pose data found.")
        return

    visualize_cameras(keyframes, args.out)


if __name__ == "__main__":
    main()
