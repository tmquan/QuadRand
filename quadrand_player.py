# -*- coding: utf-8 -*-
# File: atari.py
# Author: Yuxin Wu

import numpy as np
import os
import cv2
import threading
import six
from six.moves import range
from tensorpack.utils import logger
from tensorpack.utils.utils import get_rng, execute_only_once
from tensorpack.utils.fs import get_dataset_path
import heapq
from collections import deque
import gym
from gym import spaces
from gym.envs.atari.atari_env import ACTION_MEANING
from skimage.morphology import skeletonize
from ale_python_interface import ALEInterface
from PIL import Image, ImageDraw
import math, cv2
__all__ = ['ImagePlayer']

ROM_URL = "https://github.com/openai/atari-py/tree/master/atari_py/atari_roms"
_SAFE_LOCK = threading.Lock()

DIMZ = 1
DIMY = 256
DIMX = 256
LEAF_SIZE = 8
LINE_COLOR = (0, 255, 0)
FILL_COLOR = (0, 100, 0)
class Quad(object):
    def __init__(self, model, box, depth):
        self.model = model
        self.box = box
        self.depth = depth
        
        self.leaf = self.is_leaf()
        self.area = self.compute_area()
        self.children = []
        self.val = None
    def is_leaf(self, leaf_size=LEAF_SIZE):
        l, t, r, b = self.box
        return int(r - l <= leaf_size or b - t <= leaf_size)
    def is_last(self, leaf_size=LEAF_SIZE):
        l, t, r, b = self.box
        return int(r - l == leaf_size or b - t == leaf_size)
    def compute_area(self):
        l, t, r, b = self.box
        return (r - l) * (b - t)
    def split(self):
        l, t, r, b = self.box
        lr = l + (r - l) / 2
        tb = t + (b - t) / 2
        depth = self.depth + 1
        tl = Quad(self.model, (l, t, lr, tb), depth)
        tr = Quad(self.model, (lr, t, r, tb), depth)
        bl = Quad(self.model, (l, tb, lr, b), depth)
        br = Quad(self.model, (lr, tb, r, b), depth)
        self.children = (tl, tr, bl, br)
        # ct = Quad(self.model, (l+lr/2, t+tb/2, lr+lr/2, tb+tb/2), depth)
        # self.children = (tl, tr, bl, br, ct)
        return self.children
    def get_leaf_nodes(self, max_depth=None):
        if not self.children:
            return [self]
        if max_depth is not None and self.depth >= max_depth:
            return [self]
        result = []
        for child in self.children:
            result.extend(child.get_leaf_nodes(max_depth))
        return result
    # def get_curr_nodes(self, max_depth=None):
    #     if not self.children:
    #         return [self]
    #     else:
    #         return self.children[0]
class ImagePlayer(gym.Env):
    """
    A wrapper for ALE emulator, with configurations to mimic DeepMind DQN settings.

    Info:
        score: the accumulated reward in the current game
        gameOver: True when the current game is Over
    """

    def __init__(self, 
                 data_dir='/home/Pearl/quantm/QuadRand/data/SNEMI/', 
                 viz=0,
                 frame_skip=8, 
                 nullop_start=30,
                 max_num_frames=0):
        """
        Args:
            data_dir: path to the image 
            frame_skip: skip every k frames and repeat the action
            viz: visualization to be done.
                Set to 0 to disable.
                Set to a positive number to be the delay between frames to show.
                Set to a string to be a directory to store frames.
            nullop_start: start with random number of null ops.
            live_losts_as_eoe: consider lost of lives as end of episode. Useful for training.
            max_num_frames: maximum number of frames per episode.
        """
        super(ImagePlayer, self).__init__()

        # Read the image here
        import glob, skimage.io, cv2
        from natsort import natsorted
        self.imageFiles = natsorted (glob.glob(os.path.join(data_dir, 'images/*.tif')))
        self.labelFiles = natsorted (glob.glob(os.path.join(data_dir, 'labels/*.tif')))

        # self.images = cv2.imread(self.imageFiles[idx], cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        # self.labels = cv2.imread(self.labelFiles[idx], cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        print(self.imageFiles)
        print(self.labelFiles)
        self.images = []
        self.labels = []
        for imageFile in self.imageFiles:
            self.images.append( skimage.io.imread(imageFile))
        for labelFile in self.labelFiles:
            self.labels.append( skimage.io.imread(labelFile))        # 3x100x1024x1024
        # self.images = np.array(self.images)        
        # self.labels = np.array(self.labels)        
        self.image  = None
        self.label  = None
        self.estim  = None
        self.heap   = None
        self.root   = None    

        # avoid simulator bugs: https://github.com/mgbellemare/Arcade-Learning-Environment/issues/86
        with _SAFE_LOCK:
            self.rng = get_rng(self)
            
            # viz setup
            if isinstance(viz, six.string_types):
                assert os.path.isdir(viz), viz
                viz = 0
            if isinstance(viz, int):
                viz = float(viz)
            self.viz = viz
            if self.viz and isinstance(self.viz, float):
                self.windowname = os.path.basename(data_dir)
                cv2.namedWindow(self.windowname)

            
        self.width  = DIMX
        self.height = DIMY
        self.actions = [0, 1, 2, 3, 4, 5, 6] # Modify here {0, 1, 2, 3, 4, 5, 6: bg, fg, split}
        self.frame_skip = frame_skip
        self.nullop_start = nullop_start

        

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
        self._restart_episode()

    
    # def get_action_meanings(self):
    #     return [ACTION_MEANING[i] for i in self.actions]

    def _grab_raw_image(self):
        """
        :returns: the current 3-channel image
        """
        return self.image
    def _grab_raw_label(self):
        """
        :returns: the current 3-channel image
        """
        return self.label
    def _grab_raw_estim(self):
        """
        :returns: the current 3-channel image
        """
        return self.estim
    
    @property
    def quads(self):
        # return [x[-1] for x in self.heap]
        return [item for item in self.heap]
    
    def push(self, quad):
        # heapq.heappush(self.heap, (quad.leaf, quad))
        self.heap.append((quad.leaf, quad))

    def pop(self, act):
        if self.heap:
            # quad =  heapq.heappop(self.heap)[-1]
            quad =  self.heap.popleft()[-1]
            # print(quad)
            quad.val = act
            return quad
        else:
            return None

    def split(self, act):
        quad = self.pop(act)
        if quad.is_last():
            # self.push(quad)
            quad.val = self.actions[-1] # No need to push it again
        else:
            children = quad.split()
            for child in children:
                self.push(child)

    def _observation(self):
        img = self._grab_raw_image()    
        lbl = self._grab_raw_label()
        
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Overlay the mask
        bImg = np.zeros(vis.shape, vis.dtype)
        bImg[:,:] = (255, 0, 0)
        bMask = cv2.bitwise_and(bImg, bImg, mask=lbl)
        cv2.addWeighted(bMask, 1, vis, 1, 0, vis)

        # Overlay the current segmentation
        est  = np.zeros_like(vis)
        msk  = np.zeros_like(vis) #For visualization

        # Draw the grid
        for quad in self.root.get_leaf_nodes(max_depth=None):
            l, t, r, b = quad.box
            box = (l, t, r - 1, b - 1)
            cv2.rectangle(msk, (l, t),(r - 1, b - 1), (0, 250, 0), 2)
            cv2.rectangle(est, (l, t),(r - 1, b - 1), (0, 250, 0), -1) # Fill the estimation
        for quad in self.root.get_leaf_nodes(max_depth=None):
            l, t, r, b = quad.box
            box = (l, t, r - 1, b - 1)
            if quad.val < 5 and quad.val is not None: # BG
                cv2.rectangle(msk, (l, t),(r - 1, b - 1), tuple(quad.val*c for c in (0, 40, 0)), -1)
                cv2.rectangle(est, (l, t),(r - 1, b - 1), tuple(quad.val*c for c in (0, 40, 0)), -1)

        
        # Color the next active quad
        if self.heap:
            quad = self.heap[0][-1]
            l, t, r, b = quad.box
            box = (l, t, r - 1, b - 1)
            cv2.rectangle(msk, (l, t),(r - 1, b - 1), (0, 200, 0), 2)
            cv2.rectangle(est, (l, t),(r - 1, b - 1), (0, 200, 0), -1) # Fill the estimation


        cv2.addWeighted(msk, 1, vis, 1, 0, vis)
        
        self.estim = est[...,1] # Take the green channel

        if self.viz:
            if isinstance(self.viz, float):
                cv2.imshow(self.windowname, vis)
                cv2.imshow("estim", self.estim)
                cv2.waitKey(int(self.viz * 1000))
        # # Calculate the observation
        obs = np.stack([self.image, self.estim], axis=2)
        return vis, obs

    def render(self):
        vis, obs = self._observation()
        return vis

    def _current_state(self):
        vis, obs = self._observation()
        return obs
    

    def _restart_episode(self):
        with _SAFE_LOCK:
            idx = self.rng.randint(0, len(self.imageFiles))
            # print(idx)
            # self.image = self.images[idx].copy()
            # self.label = self.labels[idx].copy()
            self.image = self.images[idx].copy ()
            self.label = self.labels[idx].copy ()



           
            self.image = self.image.astype(np.uint8)
            self.label = self.label.astype(np.uint8)

            dimz, dimy, dimx = self.image.shape
            # The same for pair
            randz = np.random.randint(0, dimz-DIMZ+1)
            randy = np.random.randint(0, dimy-DIMY+1)
            randx = np.random.randint(0, dimx-DIMX+1)

            self.image = np.squeeze(self.image[randz:randz+DIMZ,randy:randy+DIMY,randx:randx+DIMX])
            self.label = np.squeeze(self.label[randz:randz+DIMZ,randy:randy+DIMY,randx:randx+DIMX])

            # print(self.image, self.image.shape, self.image.dtype)
            # print(self.label, self.label.shape, self.label.dtype)
            self.estim = np.zeros_like(self.image)
            ############################################################################################
            self.heap = deque([])
            self.root = Quad(self, (0, 0, self.width, self.height), 0)
            # self.push(self.root)
            self.push(self.root)
            # for k in range(1):
            # self.split(act=2)
            ############################################################################################
           

    def reset(self):
        self._restart_episode()
        return self._current_state()

    def step(self, act=None):
        r = 0
        isOver = False
        info = {'ale.lives' : 0}

        if act==self.actions[-1]:
            self.split(act)
        else:
            self.pop(act)
        # print(self.heap)
        # print(not self.heap)
        isOver = not self.heap
        if isOver:
            #import scipy.spatial.distance # Similar = 1, different = 0
            
            #r = 1.0 - scipy.spatial.distance.dice(self._grab_raw_label().flatten() / 255.0, 
            #                                          self._grab_raw_estim().flatten() / 255.0) 
            from sklearn.metrics import mean_absolute_error
            lbl = self._grab_raw_label()
            est = self._grab_raw_estim()
            # # est[est==128] = 0 # Threshold
            # mae = 1.0 - mean_absolute_error(lbl.flatten()/255.0, 
            #                                 est.flatten()/255.0)

            # def dice_metrics(gt, seg):
            #     dice = 0
            #     for k in np.unique(gt)[1:]: # No background
            #         dice += np.sum(seg[gt==k]==k)*2.0 / (np.sum(seg[seg==k]==k) + np.sum(gt[gt==k]==k))
            #     return dice
            # dice = dice_metrics(self._grab_raw_label().flatten()/255.0, 
            #                     self._grab_raw_estim().flatten()/255.0)
            # r = dice #(mae + dice) / 2.0
            import skimage.measure
            lbl = skimage.measure.label(lbl)
            est = skimage.measure.label(est)
            from sklearn.metrics.cluster import adjusted_rand_score
            rand_idx = adjusted_rand_score(lbl.flatten(), est.flatten())
            r = rand_idx
        return self._current_state(), r, isOver, info

if __name__ == '__main__':
    a = ImagePlayer(viz=-1)
    num = a.action_space.n
    print(num)

    # for act in (0,0,1,1,1   ,1,1,1,1,       0,0,1,1,1   ,1,1,1,1):
    # for act in (0,1,1,1,0,0,1,1,1,1,1,1,1,  0,0,1,1,1,0,1,1,1,1,1,1,1):
    #     print(act)
    #     state, reward, isOver, info = a.step(act)
    #     if isOver:
    #         print("Reward:", reward)
    #         a.reset()

    rng = get_rng(num)
    isFirst = True
    while True:
        if isFirst:
            act = 6
            isFirst = False
        else:
            act = int(cv2.waitKey()-ord('0'))
            
        
        state, reward, isOver, info = a.step(act)
        if isOver:
            print("Reward:", reward)
            print("isOver, press any key to continue!")
            cv2.waitKey()
            a.reset()
        print(act, isOver)
