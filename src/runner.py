import torch
import matplotlib
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import utils.dataset as dataset

def basic_train(epoch, dataloader, encoder, decoder, optimizer, loss_fn, device, training=True, num_joints=57, joint_dim=2):
    
    if training == True:
        decoder.train()
    else:
        decoder.eval()
        
    all_loss = 0
    for i, data in enumerate(dataloader):
        if training:
            optimizer.zero_grad()

        img_seq = torch.FloatTensor(data['img_seq'])
        pose_seq = data['pose_seq'].cuda()
        label_seq = data['label_seq'].cuda()
        transl_eng = data['transl_eng']
        transl_deu = data['transl_deu']
        img_seq_len = data['seq_len']

        total_sequence = sum(np.array(img_seq_len))
        delta = label_seq - pose_seq

        initial_delta = torch.zeros_like(delta)
        combined = torch.cat((pose_seq, initial_delta), dim=-2)

        lang_embed = torch.FloatTensor(encoder(transl_eng)).cuda()

        # For use TP!
        output = decoder(lang_embed, max(img_seq_len), combined.view(combined.shape[0], combined.shape[1], -1),\
                              epoch=epoch)
        pred_pose = dataset.unpad_sequence(output, img_seq_len).data


        combined_label = torch.cat((label_seq, delta), dim=-2)

        gt_label = dataset.unpad_sequence(combined_label, np.array(img_seq_len)).data.view(-1, num_joints*joint_dim*2)

        # MAKESHIFT ATTENTION (TODO: REPLACE THIS LATER W/ PROPER ATTENTION)
        attention = torch.ones_like(gt_label)
        attention[1:7,...] *= 2.5
        attention[:,15:] *= 1.5

        loss = loss_fn(pred_pose*attention, gt_label*attention)
        
        if training:
            loss.backward()
            optimizer.step()
            if i % 1 == 0:
                print('Epoch: {} | Iteration: {} | Loss: {}'.format(epoch, i, loss.cpu().item()))
        
        all_loss += loss.detach().cpu()

        # TODO: set up things w/ tensorboard
    
    if training:
        return {
                'model': decoder,
                'optimizer': optimizer,
                'loss': all_loss / len(dataloader)
        }
    else:
        return {
                'loss': all_loss / len(dataloader) 
        }

