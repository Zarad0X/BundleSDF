# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,zmq,pdb,sys,time,torchvision
code_dir = os.path.dirname(os.path.realpath(__file__))
import argparse
import cv2
import torch,imageio
from BundleTrack.LoFTR.src.loftr import *
from Utils import *
import torch.nn.functional as F

# Add tracking submodule to path for AllTracker imports
tracking_path = os.path.dirname(os.path.dirname(code_dir))
print("Tracking path:", tracking_path)
if tracking_path not in sys.path:
    sys.path.append(tracking_path)
from tracking.tracker_toolkit import Alltracker
class AlltrackerRunner:
  def __init__(self):
    print("Initialize AllTrackerRunner")
    self.is_video_tracker = True
    

    class Config:
        def __init__(self):
            self.window_len = 16
            self.query_frame = 0
            self.inference_iters = 4
            self.verbose = False
            self.rate = 8
            
    self.cfg = Config()
    self.model = Alltracker(self.cfg)

  @torch.no_grad()
  def predict_sequence(self, rgb_list):
    '''
    @rgb_list: list of (H,W,C) numpy arrays, length T
    Matches frame 0 to frame T-1
    '''
    T_in = len(rgb_list)
    H, W, _ = rgb_list[0].shape
    
    # Prepare tensor
    imgs_torch = [torch.from_numpy(img).permute(2,0,1).float() for img in rgb_list]
    rgbs = torch.stack(imgs_torch, dim=0).unsqueeze(0).cuda() # (1, T, 3, H, W)
    
    images_dict = {"rgbs_tensor": rgbs}
    traj_maps, visconf_maps = self.model.get_track_maps(images_dict)
    # traj_maps: 1, T, 2, H, W
    
    # We want matches from frame 0 to frame T-1
    # traj_maps[0, T-1] contains coordinates in frame T-1 for each pixel in frame 0
    all_res = []
    for t in range(traj_maps.size(1)-T_in+1, traj_maps.size(1)):
      if traj_maps.dim() == 5:
        coords_end = traj_maps[0, t].permute(1, 2, 0).cpu().numpy() # (H,W,2)
      else:
        coords_end = traj_maps[0].permute(1, 2, 0).cpu().numpy() # (H,W,2) 
      if visconf_maps.dim() == 4:
        vis = visconf_maps[0, 0].cpu().numpy() # (B, 2, H, W) -> (H, W)
        conf = visconf_maps[0, 1].cpu().numpy() # (B, 2, H, W) -> (H, W)
      else:
        vis = visconf_maps[0, t, 0].cpu().numpy() # (H,W)
        conf = visconf_maps[0, t, 1].cpu().numpy() # (H,W)
      
      ys, xs = np.indices((H, W))
      pts0 = np.stack([xs, ys], axis=-1).reshape(-1, 2)
      pts1 = coords_end.reshape(-1, 2)
      conf_flat = conf.reshape(-1, 1)

      mask = (conf_flat[:, 0] > 0.5) & (vis.reshape(-1) > 0.5)
      pts0_sel = pts0[mask]
      pts1_sel = pts1[mask]
      conf_sel = conf_flat[mask]
      res = np.concatenate([pts0_sel, pts1_sel], axis=1).astype(np.float32)
      all_res.append(res)
    return all_res

  @torch.no_grad()
  def predict(self, rgbAs: np.ndarray, rgbBs: np.ndarray):
    '''
    Use AllTracker to find dense correspondences between two frames.
    Currently treating (rgbAs[i], rgbBs[i]) as a 2-frame video clip.
    @rgbAs: (N,H,W,C)
    @rgbBs: (N,H,W,C) 
    '''
    corres_all = []
    N, H, W, _ = rgbAs.shape

    for i in range(N):
      img0 = rgbAs[i]
      img1 = rgbBs[i]

      # (H,W,C) -> (C,H,W) -> (1,2,C,H,W)
      t0 = torch.from_numpy(img0).permute(2,0,1).float()
      t1 = torch.from_numpy(img1).permute(2,0,1).float()
      rgbs = torch.stack([t0, t1], dim=0).unsqueeze(0).cuda()

      images_dict = {"rgbs_tensor": rgbs}
      traj_maps, visconf_maps = self.model.get_track_maps(images_dict)
      # traj_maps: 1, T, 2, H, W

      # Get matches from frame 0 to frame 1
      # traj_maps[0,1] contains coordinates in frame 1 for each pixel in frame 0
      coords1 = traj_maps[0, 1].permute(1, 2, 0).cpu().numpy() # (H,W,2)
      if visconf_maps.dim() == 4:
         conf = visconf_maps[0, 1].cpu().numpy()
      else:
         conf = visconf_maps[0, 1, 1].cpu().numpy() # (H,W)

      ys, xs = np.indices((H, W))
      pts0 = np.stack([xs, ys], axis=-1).reshape(-1, 2)
      pts1 = coords1.reshape(-1, 2)
      conf_flat = conf.reshape(-1, 1)

      # High confidence filter
      mask = conf_flat[:, 0] > 0.5
      
      pts0_sel = pts0[mask]
      pts1_sel = pts1[mask]
      conf_sel = conf_flat[mask]

      # [x0, y0, x1, y1, conf]
      res = np.concatenate([pts0_sel, pts1_sel, conf_sel], axis=1).astype(np.float32)
      corres_all.append(res)

    return corres_all