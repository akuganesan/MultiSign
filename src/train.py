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

from utils.dataset import SIGNUMDataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

from runner import basic_train

import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    
    # Load config
    parser.add_argument('--config', is_config_file=True,
                        help='path to config file')

    # Hyperparameters and Dataset
    parser.add_argument('--epoch', type=int, default=5,
                        help='total number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate during training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for training/evaluation')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers for the dataloaders')     
    parser.add_argument('--seed', type=int, default=1,
                        help='seed to use')
    parser.add_argument('--subsample', type=int, default=10,
                        help='number of frames to subsample from the raw sequences')
    parser.add_argument('--dataset_root', type=str,
                        help='path to the dataset', default='/scratch/datasets/SIGNUM')
    
    # Training
    parser.add_argument('--attn', dest='attn', action='store_true')
    parser.add_argument('--no-attn', dest='attn', action='store_false')
    parser.set_defaults(attn=False)
    
    parser.add_argument('--denorm', dest='denorm', action='store_true')
    parser.add_argument('--no-denorm', dest='denorm', action='store_false')
    parser.set_defaults(denorm=False)
    
    parser.add_argument('--norm_poses', dest='normalize', action='store_true')
    parser.add_argument('--no_norm_poses', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)
    
    parser.add_argument('--lr_sched', dest='lr_scheduler', action='store_true')
    parser.add_argument('--no_lr_sched', dest='lr_scheduler', action='store_false')
    parser.set_defaults(lr_scheduler=False)
    
    
    # GPU Allocation
    parser.add_argument('--cuda_num', type=int, default=0,
                        help='which GPU to use if there are multiple available')

    # Logging/Monitoring
    parser.add_argument('--run_name', type=str,
                        help='name of the current experiment')
    parser.add_argument('--run_folder', type=str,
                        help='folder to store run files', default='runs')
    parser.add_argument('--model_path', type=str,
                        help='where to store model checkpoints', default='model')
    
    return parser

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    total_epochs = args.epoch
    learning_rate = args.lr
    batch_size = args.batch_size
    dataset_root = args.dataset_root
    num_workers = args.num_workers
    subsample = args.subsample
    denorm = args.denorm
    attn = args.attn
    normalize_poses = args.normalize
    lr_scheduler = args.lr_scheduler
    print('using normalized poses: ', normalize_poses)
    
    run_name = args.run_name
    run_folder = args.run_folder
    model_path = args.model_path
    
    torch.manual_seed(args.seed)

    device = device = torch.device("cuda:{}".format(args.cuda_num) if torch.cuda.is_available() else "cpu")

    run_name = generate_unique_run_name(run_name, model_path, run_folder)
    
    # Create folders for logging
    model_folder = create_folder(model_path, run_name)
    run_folder = create_folder(run_folder, run_name)
    
    save_args(os.path.join(run_folder, "args.txt"), args)
    save_configs(os.path.join(run_folder, "config.txt"), args.config)
    writer = SummaryWriter(os.path.join(run_folder, "total_epoch={}-bs={}-lr={}-attn={}-normalize={}-subsample={}".format(total_epochs, \
                                                                                                                          batch_size, 
                                                                                                                         learning_rate,
                                                                                                                         attn,
                                                                                                                         normalize_poses,
                                                                                                                         subsample)))

    train_dataset = SIGNUMDataset('/scratch/datasets/SIGNUM', use_pose=True, subsample=subsample,\
                                  training=True, normalize_poses=normalize_poses)
    validation_dataset = SIGNUMDataset('/scratch/datasets/SIGNUM', use_pose=True, subsample=subsample,\
                                       training = False, normalize_poses=normalize_poses)
    
    print('Training Examples: {}'.format(len(train_dataset)))
    print('Validation Examples: {}'.format(len(validation_dataset)))
            
    # Initialize Models
    print('INITIALIZING MODELS')
    encoder = model.language_encoder()
    
    decoder = model.Decoder(hidden_size=768, pose_size=57*2, trajectory_size=57*2,
                               use_h=False, start_zero=False, use_tp=True,
                               use_lang=False, use_attn=False).to(device)

    for param in encoder.parameters():
        param.requires_grad = False
        
    encoder.eval() 

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, \
                                                    num_workers=num_workers, collate_fn=train_dataset.collate)
    
    print('Starting Validation')
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, \
                                                    num_workers=num_workers, collate_fn=train_dataset.collate)

    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    if lr_scheduler:
        steps = 10
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    
    lowest_validation_loss = 1e7

    loss_fn = nn.L1Loss()

    print('Starting Training')
    for epoch in range(total_epochs):
        
        train_dict = basic_train(epoch, train_loader, encoder, decoder, optimizer, loss_fn, \
                                 device, training=True, writer=writer, denorm=denorm, use_attn=attn, \
                                 normalize_poses=normalize_poses)
        model = train_dict['model']
        optimizer = train_dict['optimizer']
        training_loss = train_dict['loss']

        val_dict = basic_train(epoch, validation_loader, encoder, decoder, None, loss_fn, \
                               device, training=False, writer=writer, denorm=denorm, use_attn=attn,\
                               normalize_poses=normalize_poses)
        validation_loss = val_dict['loss']
        
        print("Loss on Validation Dataset after Epoch {} = {}".format(epoch, validation_loss))

        """SAVE MODEL AND OPTIMIZER"""
        training_file = os.path.join(model_folder, "latest_validation.tar")
        torch.save({
                    'epoch': epoch,
                    'batch_size': batch_size,
                    'validation_loss': validation_loss,
                    'lowest_validation_loss':lowest_validation_loss,
                    'model_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
        }, training_file)

        if validation_loss < lowest_validation_loss:
            lowest_validation_loss = validation_loss
            validation_model = os.path.join(model_folder, "lowest_validation_model.tar")
            torch.save({
                        'epoch': epoch,
                        'batch_size': batch_size,
                        'validation_loss': lowest_validation_loss,
                        'model_state_dict': decoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
            }, validation_model)

    print('Finished Training')