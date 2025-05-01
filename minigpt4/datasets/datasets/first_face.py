import glob
import os
import json
import pickle
import random
import time
import itertools
import pandas as pd
import json

import torch.nn.functional as F

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import torch
from torch.utils.data import Dataset
import webdataset as wds
import cv2

from minigpt4.datasets.datasets.base_dataset import BaseDataset

class FeatureFaceDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):

        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.caption_instruction_pool = [
            "Please describe the narrative and tone in the video, particularly any potentially biased or harmful portrayals of migrants or refugees.",
            "Can you identify and describe how migrants or refugees are portrayed in the video, including any language or imagery that might reinforce stereotypes?",
            "Describe how the tone and expressions in the video could influence viewers’ perceptions of displaced individuals.",
            "Analyze the visual and auditory cues in the video. Are there elements that may unintentionally spread negative stereotypes?",
            "Provide a detailed description of how the video frames human mobility and migration—are the portrayals neutral, sympathetic, or hostile?",
            "Examine the mood and language used in the video to determine if it reflects bias or harmful generalizations about migration.",
        ]

        self.aggressive_instruction_pool = [
            "Please assess whether the video contains xenophobic, racist, or discriminatory language or implications.",
            # "Does the video include language or visuals that portray migrants or refugees in a negative or dehumanizing way?",
            # "Determine whether the statements in the video reflect hostility, exclusion, or fear toward migrants or displaced populations.",
            # "Identify any instances of hate speech, negative stereotypes, or generalizations about migration or mobility.",
            # "Evaluate the aggressiveal tone of the video. Does it incite fear, distrust, or resentment toward migrant communities?",
            # "Is the video promoting an inclusive and respectful portrayal of migration, or does it reinforce harmful attitudes?",
        ]

        self.aggressive_instruction_pool = [
            "Please assess whether the video contains xenophobic, racist, or discriminatory language or implications.",
            "Does the video include language or visuals that portray migrants or refugees in a negative or dehumanizing way?",
            "Determine whether the statements in the video reflect hostility, exclusion, or fear toward migrants or displaced populations.",
            "Identify any instances of hate speech, negative stereotypes, or generalizations about migration or mobility.",
            "Evaluate the aggressiveal tone of the video. Does it incite fear, distrust, or resentment toward migrant communities?",
            "Is the video promoting an inclusive and respectful portrayal of migration, or does it reinforce harmful attitudes?",
        ]




        self.task_pool = [
           "aggressive",
        #    "reason",
        #    "reason_v2",
        ]

        print("ann_path: ", ann_path)
        self.ann_path = ann_path
        self.file_path = os.path.dirname(ann_path)
        self.tmp = [x.strip().split(' ') for x in open(ann_path)]
        print(('video number:%d' % (len(self.tmp))))

        emos = ['agressive', 'non']

        self.emo2idx, self.idx2emo = {}, {}
        for ii, emo in enumerate(emos): self.emo2idx[emo] = ii
        for ii, emo in enumerate(emos): self.idx2emo[ii] = emo

        json_file_path = "/data/user/dataset/unicc/MERR_coarse_grained.json" 
        with open(json_file_path, 'r') as json_file:
            self.MERR_coarse_grained_dict = json.load(json_file)

        reason_json_file_path = "/data/user/dataset/unicc/MERR_fine_grained.json"
        with open(reason_json_file_path, 'r') as json_file:
            self.MERR_fine_grained_dict = json.load(json_file)

        self.character_lines = pd.read_csv('/data/user/dataset/unicc/transcription_en_all.csv')


    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, index):
        t = self.tmp[index]
        video_name = t[0]

        video_path = os.path.join(self.vis_root, video_name + ".mp4")
        if os.path.exists(video_path):
            image = self.extract_frame(video_path)
        else:
            video_path = os.path.join(self.vis_root, video_name + ".avi")
            image = self.extract_frame(video_path)

        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)




        FaceMAE_feats, VideoMAE_feats, Audio_feats = self.get(video_name)
        if len(VideoMAE_feats.shape) == 1:
            VideoMAE_feats = VideoMAE_feats.unsqueeze(0)
        if len(Audio_feats.shape) == 1:
            Audio_feats = Audio_feats.unsqueeze(0)
        if len(FaceMAE_feats.shape) == 1:
            FaceMAE_feats = FaceMAE_feats.unsqueeze(0)
        video_features = torch.cat((FaceMAE_feats, VideoMAE_feats, Audio_feats), dim=0)


        # random task
        task = random.choice(self.task_pool)
        if task == "aggressive":
            caption = t[2] # llama2 putput only aggressive class
            caption = self.text_processor(caption)
            instruction_pool = self.aggressive_instruction_pool
        elif task == "reason":
            caption = self.MERR_coarse_grained_dict[video_name]['caption']

            caption = self.text_processor(caption)
            instruction_pool = self.reason_instruction_pool

        elif task == "reason_v2":
            caption = self.MERR_fine_grained_dict[video_name]['smp_reason_caption']

            # caption = "" # for test reasoning

            caption = self.text_processor(caption)
            instruction_pool = self.reason_instruction_pool


        aggressive = self.emo2idx[t[2]]
        sentence = self.character_lines.loc[self.character_lines['name'] == video_name, 'sentence'].values[0]
        character_line = "The person in video says: {}. ".format(sentence)
        
        instruction = "<video><VideoHere></video> <feature><FeatureHere></feature> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))
        # instruction = "<video><VideoHere></video> <feature><FeatureHere></feature> [{}] {} ".format(task, random.choice(instruction_pool))

        return {
            "image": image,
            "video_features": video_features,
            "instruction_input": instruction,
            "answer": caption,
            "aggressive": aggressive,
            "image_id": video_name
        }
    
    def extract_frame(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        success, frame = video_capture.read()
        if not success:
            raise ValueError("Failed to read video file:", video_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_capture.release()

        return frame_rgb


    def get(self, video_name):
        # FaceMAE feature
        FaceMAE_feats_path = os.path.join(self.file_path, 'mae_340_UTT', video_name + '.npy')
        FaceMAE_feats = torch.tensor(np.load(FaceMAE_feats_path))

        # VideoMAE feature
        VideoMAE_feats_path = os.path.join(self.file_path, 'maeV_399_UTT', video_name + '.npy')
        VideoMAE_feats = torch.tensor(np.load(VideoMAE_feats_path))

        # Audio feature
        Audio_feats_path = os.path.join(self.file_path, 'HL-UTT', video_name + '.npy')
        Audio_feats = torch.tensor(np.load(Audio_feats_path))

        return FaceMAE_feats, VideoMAE_feats, Audio_feats
