import numpy as np
import torch
from sklearn.utils.linear_assignment_ import linear_assignment
from numba import jit
import copy
from scipy.optimize import linear_sum_assignment
class Tracker(object):
  def __init__(self, opt):
    self.opt = opt
    self.reset() 

  def reset(self):
    self.id_count = 0
    self.tracks = [] # all tracks
    self.visible_tracks = [] # visible tracks
    self.invisible_tracks = [] # invisible tracks
    self.delete_track_list = [] # delete tracklets

  def init_track(self, results, resdet, cnt): # initialize the first frame
    for item in results:
      if item['score'] > self.opt.new_thresh:
        self.id_count += 1
        item['active'] = 1 
        item['age'] = 1  # how long the item exits
        item['consecutiveInvisibleCount'] = 0 # how long the item is invisible
        item['totalVisibleCount'] = 1 # how long the item is visible        
        item['tracking_id'] = self.id_count
        if not ('ct' in item):
          bbox = item['bbox']
          item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        self.tracks.append(item)
        self.visible_tracks.append(item)
    self.resdet = resdet # all matched tracks in pre frames
    self.cnt = cnt 


  def step(self, results, resdet, cur_cnt, public_det=None): 
    N = len(results) # N : nums of targets in cur_img
    M = len(self.tracks) # M : nums of tracklets in pre_frames

#          **************************************** track det match ****************************************
    # use matching mertric : center dist
    dets = np.array([det['ct'] + det['tracking'] for det in results], np.float32) # N x 2
    track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * \
      (track['bbox'][3] - track['bbox'][1])) \
      for track in self.tracks], np.float32) # M
    track_cat = np.array([track['class'] for track in self.tracks], np.int32) # M
    item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * \
      (item['bbox'][3] - item['bbox'][1])) \
      for item in results], np.float32) # N
    item_cat = np.array([item['class'] for item in results], np.int32) # N
    tracks = np.array([pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2
    dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # N x M

    invalid = ((dist > track_size.reshape(1, M)) + \
      (dist > item_size.reshape(N, 1)) + \
      (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
    dist = dist + invalid * 1e18
    
    # use matching mertric : iou
    iou_mat = np.zeros((len(results),len(self.tracks)))
    
    for i in range(N):# N ???????????????
        for j in range(M): # M ???????????????
            #NOTE predict bbox
            predict_box = [x + results[i]['tracking'] for x in results[i]['bbox']]
            iou_mat[i][j] = compute_iou(predict_box,self.tracks[j]['bbox'])
    # cal_matching targets
    if self.opt.hungarian:
      item_score = np.array([item['score'] for item in results], np.float32) # N
      dist[dist > 1e18] = 1e18
      matched_indices = linear_assignment(iou_mat)
    else:
      matched_indices = greedy_assignment_center_dist(copy.deepcopy(dist))

    unmatched_dets = [d for d in range(dets.shape[0]) \
      if not (d in matched_indices[:, 0])] # unmatched targets in cur_detections , unmatched_2
    unmatched_tracks = [d for d in range(tracks.shape[0]) \
      if not (d in matched_indices[:, 1])] # unmatched targets in pre_tracks , unmatched_1
    
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

    #      **************************************** ??? update matched tracks ****************************************
    ret = []
    tracklets_all = []
    for m in matches: # id - matching
      track = results[m[0]]
      track['tracking_id'] = self.tracks[m[1]]['tracking_id'] # inherit id
      track['consecutiveInvisibleCount']  = 0
      track['totalVisibleCount'] = self.tracks[m[1]]['totalVisibleCount'] + 1 
      track['age'] = self.tracks[m[1]]['age'] + 1 
      # track['active'] = self.tracks[m[1]]['active'] + 1
      ret.append(track)
      

    #         **************************************** update unmatched tracks****************************************

    for i in unmatched_tracks: # unmatch targets in pre_tracks , unmatched_1
      track = self.tracks[i]
      track['age'] += 1
      track['consecutiveInvisibleCount'] += 1
      # track['active'] = 0 
      # bbox = track['bbox']
      # ct = track['ct']
      # v = [0, 0]
      # track['bbox'] = [
      #   bbox[0] + v[0], bbox[1] + v[1],
      #   bbox[2] + v[0], bbox[3] + v[1]]
      # track['ct'] = [ct[0] + v[0], ct[1] + v[1]]
      # Compute the fraction of the track's age for which it was visible.
      #NOTE ????????????
      v = track['tracking']
      bbox = track['bbox']
      track['bbox'] = [
        bbox[0] - v[0], bbox[1] - v[1],
        bbox[2] - v[0], bbox[3] - v[1]] # ???????????????
      ct = track['ct']
      track['ct'] = [ct[0] - v[0], ct[1] - v[1]]
      ages = track['age']
      totalVisibleCounts = track['totalVisibleCount']
      visibility = totalVisibleCounts / ages

      # find the indices of 'lost' tracks
      if (ages < self.opt.max_ages and visibility < 0.6) or \
      (track['consecutiveInvisibleCount'] >= self.opt.max_invisible_frames): # ??????????????????
        continue
      else:
        tracklets_all.append(track)

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
      # Private detection: create tracks for all un-matched detections
      
    #         **************************************** create new tracks ****************************************

      for i in unmatched_dets: # cur_image unmatched targets, unmatched_2
        track = results[i]
        if track['score'] > self.opt.new_thresh:
          self.id_count += 1
          track['tracking_id'] = self.id_count
          track['age'] = 1
          track['consecutiveInvisibleCount'] = 0
          track['totalVisibleCount'] = 1
          # track['active'] = 1
          ret.append(track)

    # return new tracks' results   
    tracklets_all.extend(ret)
    self.tracks = tracklets_all
    return ret

def greedy_assignment_iou(self,iou_mat):
  matched_indices = []
  if iou_mat.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1,2)
  for i in range(iou_mat.shape[0]):
    j = iou_mat[i].argmax()
    if iou_mat[i][j] > self.opt.min_iou: # default min_iou = 0.5
      iou_mat[:,j] = -1
      matched_indices.append([i,j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)

def greedy_assignment_center_dist(dist):
  matched_indices = []
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    j = dist[i].argmin()
    if dist[i][j] < 1e16:
      dist[:, j] = 1e18
      matched_indices.append([i, j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)

# compute iou 
def compute_iou(box1,box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    i_w = min(x2, x4) - max(x1, x3)
    i_h = min(y2, y4) - max(y1, y3)
    if(i_w <= 0 or i_h <= 0):
      return 0
    i_s = i_w * i_h
    s_1 = (x2 - x1) * (y2 - y1)
    s_2 = (x4 - x3) * (y4 - y3)
    return float(i_s) / (s_1 + s_2 - i_s) 