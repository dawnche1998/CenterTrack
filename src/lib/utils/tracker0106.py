#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from numba import jit
import copy


class Tracker(object):
    def __init__(self, opt):
        self.opt = opt
        self.reset()

    def reset(self):
        self.id_count = 0
        self.tracks = []

    def init_track(self, results, resdet, cnt): # initialize the first frame of the video
        for item in results:
            if item['score'] > self.opt.new_thresh:
                self.id_count += 1 #score>0.3
                # active and age are never used in the paper
                item['active'] = 1 # activte : how long the item is visible
                item['age'] = 1 # age : how long the item is invisible
                item['tracking_id'] = self.id_count
                if not ('ct' in item):
                    bbox = item['bbox']
                    item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                self.tracks.append(item)
        self.resdet = resdet
        self.cnt = cnt
        # self.keyQueue = []

    def step(self, results, resdet, cur_cnt, public_det=None):
        N = len(results)  # N : nums of targets in cur_img
        M = len(self.tracks) # M : nums of tracklets in pre_frames

        # use center dist
        dets = np.array(
            [det['ct'] + det['tracking'] for det in results], np.float32)  # N x 2 #预测上一帧的中心点：本帧+偏移量
        track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * \
                                (track['bbox'][3] - track['bbox'][1])) \
                               for track in self.tracks], np.float32)  # M 上一帧的面积
        track_cat = np.array([track['class'] for track in self.tracks], np.int32)  # M
        
        # pose similarity end 

        # offset 5-frames 预测5帧之前的中心点位置 start
        # if cur_cnt >= 6:
        #     dets_back5_pred = np.array([det['ct'] + det['tracking'] * 5 for det in results], np.float32) # back 5 frames 中心点位置
        #     dets_back5 = np.array([det_back5['ct'] for det_back5 in resdet[cur_cnt -5]], np.float32)
        #     dist_pre5 = (((dets_back5.reshape(1,-1,2) - dets_back5_pred.reshape(-1, 1, 2)) ** 2).sum(axis=2))
        #     N x LEN(CUR_CNT - 5))
            # track_size_back_5 = np.array([((candidate['bbox'][2] - candidate['bbox'][0]) * \
            #                                (candidate['bbox'][3] - candidate['bbox'][1])) for candidate in resdet[cur_cnt - 5]])
                                           # len(resdet[cur_cnt - 5])
        # offset 5-frames 预测5帧之前的中心点位置 end

        item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * \
                               (item['bbox'][3] - item['bbox'][1])) \
                              for item in results], np.float32)  # N 这一帧的面积
        item_cat = np.array([item['class'] for item in results], np.int32)  # N
        tracks = np.array(
            [pre_det['ct'] for pre_det in self.tracks], np.float32)  # M x 2 上一帧的中心点
        dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M  两帧之间的距离矩阵
        
        invalid = ((dist > track_size.reshape(1, M)) + \
               (dist > item_size.reshape(N, 1)) + \
               (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
        dist = dist + invalid * 1e18
        
        # use iou only
            # compute iou mat 
        iou_mat = []
        for cur_person in results:# N 这一帧的人
            tmp = []
            for pre_person in self.tracks: # M 上一帧的人
                iou = compute_iou(cur_person['bbox'],pre_person['bbox'])
                tmp.append(iou)
            iou_mat.append(tmp)
        
        # use pose similarity + centerdist
        # compute pose similarity 
        pose_sim = 1.0      
        if cur_cnt >= 2:
            pre_hps = np.array([track['hps'] for track in self.tracks], np.float) #M
            cur_hps = np.array([det['hps'] for det in results],np.float) #N
            pos_mat = []
            for i in range(len(cur_hps)):
                row = []
                for j in range(len(pre_hps)):
                    x = cur_hps[i].tolist()
                    y = pre_hps[j].tolist()
                    row.append(1.0 - pose_similarity(x,y))
                pos_mat.append(row)
            pos_mat = np.array(pos_mat)
        
        # normalize 中心点之间的距离
        # normdist = maxminnorm(dist)
        # if cur_cnt >= 2:
        #     finaldist = normdist + pos_mat
        # else:
        #     finaldist = normdist
        
        # finaldist = finaldist + invalid * 1e18
        
        if self.opt.hungarian:
            item_score = np.array([item['score'] for item in results], np.float32)  # N
            dist[dist > 1e18] = 1e18
            matched_indices = linear_assignment(dist)
        else:
            matched_indices = greedy_assignment(copy.deepcopy(dist))
        unmatched_dets = [d for d in range(dets.shape[0]) \
                          if not (d in matched_indices[:, 0])]
        unmatched_tracks = [d for d in range(tracks.shape[0]) \
                            if not (d in matched_indices[:, 1])]

        if self.opt.hungarian:
            matches = []
            for m in matched_indices:
                if dist[m[0], m[1]] > 1e16:
                    unmatched_dets.append(m[0])
                    unmatched_tracks.append(m[1])
                else:
                    matches.append(m)
            matches = np.array(matches).reshape(-1, 2)
        else:
            matches = matched_indices

        ret = []
        for m in matches: #匹配到的tracking_id 继承id
            track = results[m[0]]
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']
            track['age'] = 1
            track['active'] = self.tracks[m[1]]['active'] + 1
            ret.append(track)


        if self.opt.public_det and len(unmatched_dets) > 0:
            # Public detection: only create tracks from provided detections
            pub_dets = np.array([d['ct'] for d in public_det], np.float32)
            dist3 = ((dets.reshape(-1, 1, 2) - pub_dets.reshape(1, -1, 2)) ** 2).sum(
                axis=2)
            matched_dets = [d for d in range(dets.shape[0]) \
                            if not (d in unmatched_dets)]
            dist3[matched_dets] = 1e18
            for j in range(len(pub_dets)):
                i = dist3[:, j].argmin()
                if dist3[i, j] < item_size[i]:
                    dist3[i, :] = 1e18
                    track = results[i]
                    if track['score'] > self.opt.new_thresh:
                        self.id_count += 1
                        track['tracking_id'] = self.id_count
                        track['age'] = 1
                        track['active'] = 1
                        ret.append(track)

        else:
            #增加一个新的tracklet
            # Private detection: create tracks for all un-matched detections
            for i in unmatched_dets:
                track = results[i]
                if track['score'] > self.opt.new_thresh:
                    self.id_count += 1
                    track['tracking_id'] = self.id_count
                    track['age'] = 1
                    track['active'] = 1
                    ret.append(track)

        for i in unmatched_tracks:
            track = self.tracks[i]
            if track['age'] < self.opt.max_age:
                track['age'] += 1
                track['active'] = 0
                bbox = track['bbox']
                ct = track['ct']
                v = [0, 0]
                track['bbox'] = [
                    bbox[0] + v[0], bbox[1] + v[1],
                    bbox[2] + v[0], bbox[3] + v[1]]
                track['ct'] = [ct[0] + v[0], ct[1] + v[1]]
                ret.append(track)
        self.tracks = ret
        return ret


def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)


# keyframe Queue
def keymatch(self,unmatched_tracks,resdet,unmateched_dets):
    lost_ids = {'lost_ids':unmatched_tracks}
    resdet[cur_cnt - 1].append(lost_ids)
    keyQueue.append(resdet[cur_cnt - 1])
    keylen = 5
    if len(keyQueue)>keylen: #关键帧列表长度为5帧
        keyQueue = keyQueue[-5:]
    for i in umatched_dets: 
        track = results[i]
        if track['score'] > self.opt.new_thresh:
            for keyframe in keyQueue:
                if not keyframe[-1].has_key('lost_ids'):
                    break
                for result in keyframe:
                    if result['tracking_id'] in keyframe[-1]['lost_ids']:
                        if pose_similarity(track,result) > opt.pose_threshold:
                            track['tracking_id'] = result['tracking_id']
                            track['active'] = 1
                            track['age'] = 1
                            ret.append(track)
                            keyframe[-1]['lost_ids'].remove(track['tracking_id'])
                        else:  # 新的 tracklet
                            self.id_count += 1
                            track['tracking_id'] = self.id_count
                            track['age'] = 1
                            track['active'] = 1
                            ret.append(track)

# pose similarity
def pose_similarity(cur_hps, pre_hps):
    x,y = cur_hps,pre_hps
    assert len(x) == len(y),"len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x==y else float(0)
    res = np.array([[x[i]*y[i],x[i]*x[i],y[i]*y[i]]for i in range(len(x))])
    cos = sum(res[:,0]) / (np.sqrt(sum(res[:,1])) * np.sqrt(sum(res[:,2])))
    return 0.5*cos + 0.5

# normalize center dist min-max 
def maxminnorm(array):
    maxarr = array.max(axis=0) if array.any() else 1
    minarr = array.min(axis=0) if array.any() else 0
    norDataset = np.zeros(array.shape)
    rangestmp = maxarr - minarr
    mtmp = array.shape[0]
    norDataset = array - np.tile(minarr,(mtmp,1))
    norDataset = norDataset/np.tile(rangestmp,(mtmp,1))
    return norDataset

# compute iou 
def compute_iou(box1,box2):

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  max(b1_x1, b2_x1)
    inter_rect_y1 =  max(b1_y1, b2_y1)
    inter_rect_x2 =  min(b1_x2, b2_x2)
    inter_rect_y2 =  min(b1_y2, b2_y2)

    # Intersection area
    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1,torch.zeros(inter_rect_x2.shape))*max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou
