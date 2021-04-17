import os
import torch
import matplotlib
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import utils.model as model
import utils.dataset as dataset
import utils.skel as skel
from utils.utils import create_folder, save_args, save_configs, generate_unique_run_name
import torch.optim as optim

os.environ['TRANSFORMERS_CACHE'] = 'transformer_cache/'
from utils.dataset import SIGNUMDataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

from runner import basic_test

import configargparse



def config_parser():
    parser = configargparse.ArgumentParser()
    
    # Which type of model to evaluate
    parser.add_argument('--encoder_type', type=str,
                        help='type of language encoder')  

    # Hyperparameters and Dataset
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers for the dataloaders')     
    parser.add_argument('--subsample', type=int, default=10,
                        help='number of frames to subsample from the raw sequences')
    parser.add_argument('--dataset_root', type=str,
                        help='path to the dataset', default='/scratch/datasets/SIGNUM')
    parser.add_argument('--speaker_id', type=int, default=11,
                        help='which speaker to run testing for')
    
    
    # GPU Allocation
    parser.add_argument('--cuda_num', type=int, default=0,
                        help='which GPU to use if there are multiple available')

    # Logging/Monitoring
    parser.add_argument('--run_num', type=int,
                        help='folder to store run files', default=0)
    parser.add_argument('--model_path', type=str,
                        help='where to load model checkpoints', default='model-best')
    
    return parser

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    batch_size = 1 #Fixed to 1
    dataset_root = args.dataset_root
    num_workers = args.num_workers
    subsample = args.subsample
    encoder_type = args.encoder_type
    speaker_id = str(args.speaker_id)
    print('Running Evaluation for {} on Speaker {}'.format(encoder_type, speaker_id))
    
    run_num = args.run_num
    model_path = args.model_path   
    
    device = device = torch.device("cuda:{}".format(args.cuda_num) if torch.cuda.is_available() else "cpu")

    """LOAD THE DATASET"""
    test_dataset = SIGNUMDataset('/scratch/datasets/SIGNUM', use_pose=True, subsample=subsample,\
                                  testing=True, normalize_poses=True, speaker_id=speaker_id)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, \
                                                    num_workers=num_workers, collate_fn=test_dataset.collate)

    print('Testing Examples: {}'.format(len(test_dataset)))

    # Initialize Models
    print('INITIALIZING MODELS')
    encoder = model.language_encoder(model_type=encoder_type)
    for param in encoder.parameters():
        param.requires_grad = False
        
    encoder.eval()
    
    """SET THE MODEL NAME"""
    if encoder_type == "multi":
        model_name = "multi_run={}".format(run_num)
    elif encoder_type == "en":
        model_name = "english_run={}".format(run_num)
    else: # encoder_type == "de"
        model_name = "german_run={}".format(run_num)
        
    decoder = model.Decoder(hidden_size=768, pose_size=57*2, trajectory_size=0,
                               use_h=False, start_zero=False, use_tp=False,
                               use_lang=False, use_attn=False).to(device)

    model_file = os.path.join(model_path, model_name, "lowest_validation_model.tar")
    checkpoint = torch.load(model_file, map_location=device)
    print("LOADING: {} from training epoch {}".format(model_file, checkpoint["epoch"]))
    decoder.load_state_dict(checkpoint["model_state_dict"])
    decoder.eval()
    
   

    print('Starting Testing')
    if encoder_type == 'multi':
        test_dict = basic_test(test_loader, encoder, decoder, device, encoder_type='en')
        test_loss = test_dict['loss']

        print("ENGLISH Loss on Testing Dataset after Epoch {} of Training = {}".format(checkpoint["epoch"], test_loss))
        
        test_dict = basic_test(test_loader, encoder, decoder, device, encoder_type='de')
        test_loss = test_dict['loss']

        print("GERMAN Loss on Testing Dataset after Epoch {} of Training = {}".format(checkpoint["epoch"], test_loss))
    else:
        test_dict = basic_test(test_loader, encoder, decoder, device, encoder_type=encoder_type)
        test_loss = test_dict['loss']

        print("Loss on Testing Dataset after Epoch {} of Training = {}".format(checkpoint["epoch"], test_loss))

    print('Finished Testing')
    
    