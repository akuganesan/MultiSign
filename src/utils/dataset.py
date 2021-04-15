import re
import os
import glob
import torch
import numpy as np
import pandas as pd
import deepdish as dd
import utils.skel as skel
import utils.constants as constants

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# If we want batches then we have to pad the sequence data so that it has the same lengths
def collate_noPose_fn(data_tuple):
    """
       data: is a list of tuples [img seq, annot eng, annot deu, transl eng, transl deu, seq len] with where 'img_seq' is a tensor of arbitrary shape
             and label/length are scalars
    """
    img_seq, annot_eng, annot_deu, transl_eng, transl_deu, seq_len = zip(*data_tuple)
    padded_img_seqs = pad_sequence(img_seq, batch_first=True)
    return  { 'img_seq': padded_img_seqs, 
              'annot_eng': annot_eng,
              'annot_deu': annot_deu,
              'transl_eng': transl_eng,
              'transl_deu': transl_deu,
              'seq_len': seq_len
            }

def collate_all_fn(data_tuple):
    img_seq, pose_seq, label_seq, annot_eng, annot_deu, transl_eng, transl_deu, seq_len = zip(*data_tuple)
    padded_img_seqs = pad_sequence(img_seq, batch_first=True)
    padded_pose_seqs = pad_sequence(pose_seq, batch_first=True)
    padded_label_seq = pad_sequence(label_seq, batch_first=True)
    return  { 'img_seq': padded_img_seqs,
              'pose_seq': padded_pose_seqs,
              'label_seq': padded_label_seq,
              'annot_eng': annot_eng,
              'annot_deu': annot_deu,
              'transl_eng': transl_eng,
              'transl_deu': transl_deu,
              'seq_len': seq_len
            }

def collate_noImg_fn(data_tuple):
    pose_seq, label_seq, annot_eng, annot_deu, transl_eng, transl_deu, seq_len = zip(*data_tuple)
    padded_pose_seqs = pad_sequence(pose_seq, batch_first=True)
    padded_label_seq = pad_sequence(label_seq, batch_first=True)
    return  {
              'pose_seq': padded_pose_seqs,
              'label_seq': padded_label_seq,
              'annot_eng': annot_eng,
              'annot_deu': annot_deu,
              'transl_eng': transl_eng,
              'transl_deu': transl_deu,
              'seq_len': seq_len
            }


# This function "unpads" the sequences based on the respective input sequence lenghts
def unpad_sequence(packed):
    return pad_packed_sequence(packed, batch_first=True)

def pack_sequence(tensor, lengths):
    return pack_padded_sequence(tensor, lengths, batch_first=True, enforce_sorted=False)

class SIGNUMDataset(Dataset):
    def __init__(self, dataset_dir, img_size=256, use_pose=False, include_word=False, \
                 use_image=True, subsample=10, normalize_poses=True, gen_constants=False, root_joint=1, body="BODY_25", \
                 training=False, validation=False, testing=False, gaussian_frames=False, speaker_id=None):
        """
        Args:
            dataset_dir (string): Path to SIGNUM dataset.
            img_size (int): Size to crop images to
            pose_sequence (list, optional): path to corresponding pose sequences for each sentence.
            include_word (boolean, default=False): whether or not to include individual words
            use_image (boolean, default=True): whether or not to load up images
            subsample (int, default=10): take every n frames from the poses and images
            normalize_poses (boolean, default=True): whether or not to normalize the poses
            gen_constants (boolean, default = False): if the dataset is being used to generate pose mean and 
                                                      std (no images and use "whole pose")
            root_joint (int, default=1): which joint index to center based on
            body (string, default="BODY_25"): which body model to use
            training (boolean, default=False): fixed training split
            validation (boolean, default=False): fixed validation split
            testing (boolean, default=False): fixed test split
            gaussian_frames (boolean, default=False): whether or not to sample the frames from a gaussian distribution
        """
        valtest = [265, 691, 727, 181, 584, 225, 473, 561, 135, 241, 284, 14, 64, 187, 547, 118, 457, 436, 612, 27, 301, 48, 348, 93, 777, 576, 710, 100, 453, 150, 298, 482, 489, 95, 202, 329, 88, 4, 673, 706, 382, 190, 739, 619, 229, 11, 18, 179, 677, 446, 429, 297, 309, 318, 380, 687, 511, 557, 542, 543, 487, 139, 109, 775, 704, 264, 694, 196, 768, 703, 395, 111, 488, 683, 522, 349, 653, 652, 505, 384, 735, 253, 252, 130, 594, 426, 655, 175, 365, 285, 537, 497, 424, 751, 711, 663, 7, 302, 737, 102, 47, 1, 107, 400, 9, 292, 236, 249, 682, 112, 86, 288, 437, 145, 646, 636, 199, 606, 671, 300, 161, 240, 510, 598, 445, 144, 168, 478, 570, 476, 590, 140, 52, 391, 746, 116, 566, 114, 338, 60, 638, 779, 521, 314, 423, 535, 693, 174, 217, 486, 381, 695, 607, 454, 439, 388]
        
        
        val = valtest[len(valtest)//2:]
        test = valtest[:len(valtest)//2]
        
        self.dataset_dir = dataset_dir
        self.include_word = include_word
        self.subsample=subsample
        self.use_image = use_image
        self.use_pose = use_pose
        self.normalize_poses = normalize_poses
        self.gen_constants = gen_constants
        
        if gen_constants:
            print("USING DATASET TO GENERATE MEAN AND STD")
            self.use_image = False
            self.normalize_poses = False
            
        self.root_joint = root_joint
        self.body = body
        
        self.mean = None
        self.std = None
        
        if self.body == "BODY_25":
            self.mean = constants.SAMPLE_MEAN_BODY_25
            self.std = constants.SAMPLE_STD_BODY_25
        
        
        if not self.use_image:
            self.collate = collate_noImg_fn
        elif not self.use_pose:
            self.collate = collate_noPose_fn
        else:
            self.collate = collate_all_fn
        
        
        self.transform = transforms.Compose([
            transforms.Resize([img_size,img_size]),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]) 
        
        if speaker_id:
            self.speakers = glob.glob(os.path.join(self.dataset_dir, "s"+speaker_id+"*"))
        else:
            self.speakers = glob.glob(os.path.join(self.dataset_dir, "s*"))

        self.sentence_folders = []
        self.text_files = []
        self.pose_folders = []
        
        for speaker in self.speakers:
        # For filtering the files/folders (iso = word, con = sentence)
            if self.include_word:
                folder_sep = "*/"
                file_sep = "*.txt"
            else:
                folder_sep = "con*[!_h?][!_json]/"
                file_sep = "con*.txt"

            sorted_sf = sorted(glob.glob(os.path.join(speaker, folder_sep)))
            sorted_tf = sorted(glob.glob(os.path.join(speaker, file_sep)))

            if training:
                for i in range(780):
                    if i not in valtest:
                        self.sentence_folders.append(sorted_sf[i])
                        self.text_files.append(sorted_tf[i])
                        
            elif validation:
                for i in range(780):
                    if i in val:
                        self.sentence_folders.append(sorted_sf[i])
                        self.text_files.append(sorted_tf[i])
                        print(sorted_sf[i], i)
            elif testing:
                for i in range(780):
                    if i in test:
                        self.sentence_folders.append(sorted_sf[i])
                        self.text_files.append(sorted_tf[i])
        
            # TODO: will be added after running the dataset through OpenPose
            folder_sep = "con*_h5/"
            pose_folders = sorted(glob.glob(os.path.join(speaker, folder_sep)))
            
            if self.use_pose:
                if training:
                    for i in range(780):
                        if i not in valtest:
                            self.pose_folders.append(pose_folders[i])
                elif validation:
                    for i in range(780):
                        if i in val:
                            self.pose_folders.append(pose_folders[i])
                elif testing:
                    for i in range(780):
                        if i in test:
                            self.pose_folders.append(pose_folders[i])
                self.pose_paths = []
                for folder in self.pose_folders:
                    self.pose_paths.append(sorted(glob.glob(os.path.join(folder, '*'))))
            
        
        self.sequence_paths = []
        self.text_annotation = []
        
        # Gets the image files for each folder/sequence
        for folder in self.sentence_folders:
            self.sequence_paths.append(sorted(glob.glob(os.path.join(folder, '*.jpg'))))
            
        # Read in all annotations from text files
        regex = re.compile(r'[\n\r\t]')
        for file in self.text_files:
            lines = []
            txt = open(file)
            for line in txt.readlines():
                lines.append(line)
            temp = {}
            df = pd.read_csv(file, sep='\t', lineterminator='\r', header=None)
            temp['annot_eng'] = regex.sub("", lines[0].split('annot_eng')[1])
            temp['annot_deu'] = regex.sub("", lines[1].split('annot_deu')[1])
            temp['transl_eng'] = regex.sub("", lines[2].split('transl_eng')[1])
            temp['transl_deu'] = regex.sub("", lines[3].split('transl_deu')[1])
            self.text_annotation.append(temp)

    def _load_image_sequence(self, sequence_path, frames=None):
        """
            Load in a sequence of data; NOTE WE PROBABLY WANT TO SUBSAMPLE THIS!!    
        """
        sequence = []
        for i, file_path in enumerate(sequence_path):
            if i % self.subsample == 0:
                image = Image.open(file_path).convert('RGB')
                image_tensor = self.transform(image).float()
                sequence.append(image_tensor)
        return torch.stack(sequence, dim=0), len(sequence)
    
    def _load_pose_sequence(self, pose_path, frames=None):
        sequence = []
        for i, file_path in enumerate(pose_path):
            if i % self.subsample == 0:
                pose_dict = dd.io.load(file_path)
                
                #TODO: TEMP FIX, CHANGE LATER SO IT'S NOT A DICT
                # Ignore the confidence score and only extract the 2D pose
                pose = torch.FloatTensor(pose_dict['people'][0]['pose_keypoints_2d']).view(-1, 3)[:,:-1]
                lhand = torch.FloatTensor(pose_dict['people'][0]['hand_left_keypoints_2d']).view(-1, 3)[:,:-1]
                rhand = torch.FloatTensor(pose_dict['people'][0]['hand_right_keypoints_2d']).view(-1, 3)[:,:-1]
                pose = torch.cat((pose, lhand, rhand), dim=0)
                
                # Remove unnecessary joints from the pose and root center it
                pose, gt_root = skel.root_center_pose(pose, root_joint=self.root_joint)
                pose = skel.remove_excess_joints(pose)
                
                if self.normalize_poses:
                    # normalize the pose
                    pose = skel.normalize_pose(pose, self.mean, self.std, root_joint=self.root_joint)
                    sequence.append(pose)
                else:
                    pose = skel.pose2pytorch(pose)
                    sequence.append(pose)
                
        return torch.stack(sequence, dim=0), len(sequence)
    
    def __len__(self):
        return len(self.sentence_folders)

    def __getitem__(self, idx):
        image_folder = self.sentence_folders[idx]
        sentence_annotation = self.text_annotation[idx]
        sequence_paths = self.sequence_paths[idx]
        pose_paths = self.pose_paths[idx]
        pose_folder = self.pose_folders[idx]   
        
        print(len(self.pose_folders))
        print(len(self.pose_paths))
        print(len(self.sequence_paths))
        print(len(self.sentence_folders))
        
        # make sure images and poses are coming from the same sentence
        assert image_folder.split('/')[-1] == pose_folder.split('/')[-1][:-3], 'pose seq: {}  img seq: {}'.format(image_folder, pose_folder) 
        
        if self.use_image:
            image_sequence, sequence_length = self._load_image_sequence(sequence_paths)
        
        if self.use_pose:
            pose_sequence, pose_length = self._load_pose_sequence(pose_paths)
            
            # if we are generating constants then we want to use the whole pose (not just the input ones)
            if not self.gen_constants:
                input_pose = pose_sequence[:-1,...]
                label_pose = pose_sequence[1:,...]
            else:
                input_pose = pose_sequence
                label_pose = pose_sequence

            if self.use_image:
#                 assert pose_length == sequence_length, 'pose seq: {}  img seq: {}'.format(image_folder, pose_folder)
                if pose_length != sequence_length:
                    print('pose seq: {}  img seq: {}'.format(image_folder, pose_folder))
                return [image_sequence, 
                        input_pose,
                        label_pose,
                        sentence_annotation['annot_eng'],
                        sentence_annotation['annot_deu'],
                        sentence_annotation['transl_eng'],
                        sentence_annotation['transl_deu'],
                        len(input_pose) # because the inputs/labels will be 1 less than the number of images 
                    ]
            else:
                return [
                        input_pose,
                        label_pose,
                        sentence_annotation['annot_eng'],
                        sentence_annotation['annot_deu'],
                        sentence_annotation['transl_eng'],
                        sentence_annotation['transl_deu'],
                        len(input_pose) # because the inputs/labels will be 1 less than the number of images 
                    ]
            
        else:
            return [image_sequence, 
                    sentence_annotation['annot_eng'],
                    sentence_annotation['annot_deu'],
                    sentence_annotation['transl_eng'],
                    sentence_annotation['transl_deu'],
                    sequence_length
                ]
