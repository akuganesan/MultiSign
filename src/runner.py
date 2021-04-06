import torch
import matplotlib
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import utils.constants as constants


from utils.skel import TB_vis_pose2D, prep_poses

import utils.dataset as dataset

def basic_train(epoch, dataloader, encoder, decoder, optimizer, loss_fn, device, training=True, \
                num_joints=57, joint_dim=2, writer=None, update=10):
    
    if training == True:
        decoder.train()
    else:
        decoder.eval()
        
    all_loss = 0
    for i, data in enumerate(dataloader):
#         if i == 1:
#             break
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
        packed = dataset.pack_sequence(output, np.array(img_seq_len))
        pred_pose = packed.data.view(-1,num_joints*joint_dim*2)

        combined_label = torch.cat((label_seq, delta), dim=-2)

        packed_gt = dataset.pack_sequence(combined_label, np.array(img_seq_len))
        gt_label = packed_gt.data.view(-1, num_joints*joint_dim*2)

        # MAKESHIFT ATTENTION (TODO: REPLACE THIS LATER W/ PROPER ATTENTION)
        attention = torch.ones_like(gt_label)
        attention[1:7,...] *= 2.5
        attention[:,15:] *= 1.5

        loss = loss_fn(pred_pose*attention, gt_label*attention)
        
        if training:
            loss.backward()
            optimizer.step()
            if i % update == 0:
                iterations = epoch*len(dataloader) + i        
                print('Epoch: {} | Iteration: {} | Loss: {}'.format(epoch, i, loss.cpu().item()))
                writer.add_scalar('Training Loss', loss, iterations)
                writer.add_figure('Training Predicted Pose', TB_vis_pose2D(packed, packed_gt), global_step=iterations)
                                
        all_loss += loss.detach().cpu()
    
    if training:
        return {
                'model': decoder,
                'optimizer': optimizer,
                'loss': all_loss / len(dataloader)
        }
    else:
        writer.add_scalar('Validation Loss', loss, epoch)
        writer.add_figure('Validation Predicted Pose', TB_vis_pose2D(packed, packed_gt), global_step=epoch)
        return {
                'loss': all_loss / len(dataloader) 
        }

