import torch
import matplotlib
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import utils.constants as constants
import utils.skel as skel


from utils.skel import TB_vis_pose2D, prep_poses

import utils.dataset as dataset

def mean_prior(pred_pose, mean_pose):
    return

def basic_train(epoch, dataloader, encoder, decoder, optimizer, loss_fn, device, training=True, \
                num_joints=57, joint_dim=2, writer=None, update=15, denorm=False, use_attn=False,\
                normalize_poses=True, attention_value=1, encoder_type='multi'):
    print('normalize: ', normalize_poses)
    
    if training == True:
        decoder.train()
    else:
        decoder.eval()
        
    all_loss = 0
    for i, data in enumerate(dataloader):
#         if i == 2:
#             break
        
        if training:
            optimizer.zero_grad()

        img_seq = torch.FloatTensor(data['img_seq'])
        pose_seq = data['pose_seq'].to(device)
        label_seq = data['label_seq'].to(device)
        transl_eng = data['transl_eng']
        transl_deu = data['transl_deu']
        multilingual = data['multi']
        img_seq_len = data['seq_len']

        total_sequence = sum(np.array(img_seq_len))
        delta = label_seq - pose_seq
        
        if encoder_type == 'multi':
#             print('multi')
            encoder_input = multilingual
        elif encoder_type == 'en':
#             print('en')
            encoder_input = transl_eng
        else:
#             print('de')
            encoder_input = transl_deu

        lang_embed = torch.FloatTensor(encoder(encoder_input)).to(device)

        if training:
            output = decoder(lang_embed, max(img_seq_len), pose_seq.view(pose_seq.shape[0], pose_seq.shape[1], -1),\
                                  epoch=epoch)
        else:
            output = decoder.sample(lang_embed.to(device), max(img_seq_len), \
                             pose_seq.view(pose_seq.shape[0], pose_seq.shape[1], -1)[:,0,...].to(device),\
                             attn=None)
            
        packed = dataset.pack_sequence(output, np.array(img_seq_len))
        pred_pose = packed.data.view(-1,num_joints*joint_dim)

        #test
#         final_delta = torch.zeros_like(delta)
#         final_delta[:, :-1, ...] = delta[:, 1:, ...]
#         combined_label = torch.cat((label_seq, final_delta), dim=-2)
#         combined_label = torch.cat((final_delta, label_seq), dim=-2)

        packed_gt = dataset.pack_sequence(label_seq, np.array(img_seq_len))
        gt_label = packed_gt.data.view(-1, num_joints*joint_dim)

        # MAKESHIFT ATTENTION (TODO: REPLACE THIS LATER W/ PROPER ATTENTION)
        if training:
            pred_pose = packed.data.view(-1,num_joints, joint_dim)
            gt_label = packed_gt.data.view(-1, num_joints, joint_dim)
            
            attention = torch.ones_like(gt_label)
            if use_attn:
                attention_value = 5
                attention[:,15:57,:] *= attention_value
                attention[:,15:36,:] *= attention_value # attention for the left hand
                attention[:,4,:] *= attention_value
                attention[:,7,:] *= attention_value
                
                # attention for the deltas
#                 attention[:,15+57:,:] *= attention_value
#                 attention[:,15+57:36+57,:] *= attention_value # attention for the left hand
#                 attention[:,57+4,:] *= attention_value
#                 attention[:,57+7,:] *= attention_value

            loss = loss_fn(pred_pose*attention, gt_label*attention)
        else:
            if normalize_poses:
                pred_pose = skel.denormalize_pose(pred_pose.view(-1, num_joints, joint_dim).detach().cpu())
                gt_label = skel.denormalize_pose(gt_label.view(-1, num_joints, joint_dim).detach().cpu())
            else:
                pred_pose = skel.pytorch2pose(pred_pose.view(-1, num_joints, joint_dim).detach().cpu())
                gt_label = skel.pytorch2pose(gt_label.view(-1, num_joints, joint_dim).detach().cpu())

            loss = skel.calculate_batch_mpjpe(pred_pose, gt_label)
        
        if training:
            loss.backward()
            optimizer.step()
            if i % update == 0:
                iterations = epoch*len(dataloader) + i        
                print('Epoch: {} | Iteration: {} | Loss: {}'.format(epoch, i, loss.cpu().item()))
                writer.add_scalar('Training Loss', loss, iterations)
                writer.add_figure('Training Predicted Pose', TB_vis_pose2D(packed, packed_gt, normalize_poses), global_step=iterations)
                                
        all_loss += loss.detach().cpu()
    
    if training:
        return {
                'model': decoder,
                'optimizer': optimizer,
                'loss': all_loss / len(dataloader)
        }
    else:
        writer.add_scalar('Validation Loss', all_loss / len(dataloader), epoch)
        writer.add_figure('Validation Predicted Pose', TB_vis_pose2D(packed, packed_gt, normalize_poses), global_step=epoch)
        return {
                'loss': all_loss / len(dataloader) 
        }


    
def basic_test(dataloader, encoder, decoder, device, num_joints=57, joint_dim=2, encoder_type='multi'):
    
    encoder.eval()
    decoder.eval()
        
    all_loss = 0
    for i, data in enumerate(dataloader):
#         if i == 2:
#             break

        img_seq = torch.FloatTensor(data['img_seq'])
        pose_seq = data['pose_seq'].to(device)
        label_seq = data['label_seq'].to(device)
        transl_eng = data['transl_eng']
        transl_deu = data['transl_deu']
        multilingual = data['multi']
        img_seq_len = data['seq_len']

        total_sequence = sum(np.array(img_seq_len))
        delta = label_seq - pose_seq
        
        if encoder_type == 'multi':
#             print('multi')
            encoder_input = multilingual
        elif encoder_type == 'en':
#             print('en')
            encoder_input = transl_eng
        else:
#             print('de')
            encoder_input = transl_deu

        lang_embed = torch.FloatTensor(encoder(encoder_input)).to(device)

        output = decoder.sample(lang_embed.to(device), max(img_seq_len), \
                                 pose_seq.view(pose_seq.shape[0], pose_seq.shape[1], -1)[:,0,...].to(device),\
                                 attn=None)
            
        packed = dataset.pack_sequence(output, np.array(img_seq_len))
        pred_pose = packed.data.view(-1,num_joints*joint_dim)

        packed_gt = dataset.pack_sequence(label_seq, np.array(img_seq_len))
        gt_label = packed_gt.data.view(-1, num_joints*joint_dim)

        pred_pose = skel.denormalize_pose(pred_pose.view(-1, num_joints, joint_dim).detach().cpu())
        gt_label = skel.denormalize_pose(gt_label.view(-1, num_joints, joint_dim).detach().cpu())
        loss = skel.calculate_batch_mpjpe(pred_pose, gt_label)
                                
        all_loss += loss.detach().cpu()
          
    return {
            'loss': all_loss / len(dataloader) 
    }