import torch
import glob
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# If we want batches then we have to pad the sequence data so that it has the same lengths
def collate_fn(data_tuple):
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

# This function "unpads" the sequences based on the respective input sequence lenghts
def unpad_sequence(img_sequence, sequence_lens):
    return pack_padded_sequence(img_sequence, sequence_lens, batch_first=True, enforce_sorted=False)

class SIGNUMDataset(Dataset):
    def __init__(self, dataset_dir, img_size=256, pose_sequence=None, include_word=False):
        """
        Args:
            dataset_dir (string): Path to SIGNUM dataset.
            img_size (int): Size to crop images to
            pose_sequence (list, optional): path to corresponding pose sequences for each sentence.
            include_word (boolean, default=False): whether or not to include individual words
        """
        self.dataset_dir = dataset_dir
        self.include_word = include_word
        
        self.transform = transforms.Compose([
            transforms.Resize([img_size,img_size]),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]) 
        
        # For filtering the files/folders (iso = word, con = sentence)
        if self.include_word:
            folder_sep = "*/*/"
            file_sep = "*/*.txt"
        else:
            folder_sep = "*/con*/"
            file_sep = "*/con*.txt"
                    
        self.sentence_folders = sorted(glob.glob(os.path.join(self.dataset_dir, folder_sep)))
        self.text_files = sorted(glob.glob(os.path.join(self.dataset_dir, file_sep)))
        
        # TODO: will be added after running the dataset through OpenPose
        self.pose_sequence = pose_sequence
        
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

    def _load_image_sequence(self, sequence_path):
        """
            Load in a sequence of data; NOTE WE PROBABLY WANT TO SUBSAMPLE THIS!!    
        """
        sequence = []
        for file_path in sequence_path:
            image = Image.open(file_path).convert('RGB')
            image_tensor = self.transform(image).float()
            sequence.append(image_tensor)
        return torch.stack(sequence, dim=0), len(sequence_path)
    
    def __len__(self):
        return len(self.sentence_folders)

    def __getitem__(self, idx):
        image_folder = self.sentence_folders[idx]
        sentence_annotation = self.text_annotation[idx]
        sequence_paths = self.sequence_paths[idx]

        image_sequence, sequence_length = self._load_image_sequence(sequence_paths)
            
        if self.pose_sequence is not None:
            pose_sequence = self.pose_sequence[idx]
            
        return [image_sequence, 
                sentence_annotation['annot_eng'],
                sentence_annotation['annot_deu'],
                sentence_annotation['transl_eng'],
                sentence_annotation['transl_deu'],
                sequence_length
            ]