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
                num_joints=57, joint_dim=2, writer=None, update=20, denorm=False, use_attn=False, normalize_poses=True, \
                attention_value=1, pose_to_video=False, generator=None, discriminator=None):
    
    print('normalize: ', normalize_poses)
    
    if training == True:
        decoder.train()
        if pose_to_video:
            generator.train()
            discriminator.train()
    else:
        decoder.eval()
        if pose_to_video:
            generator.eval()
            
    # setup for pix2pix
    if pose_to_video:
        # loss
        D_bce_loss = nn.BCELoss()
        G_l1_loss = nn.L1Loss()
        G_l1_lambda = 100

        # learning rate
        lrG = 0.0002
        lrD = 0.0002
        beta1 = 0.5
        beta2 = 0.999

        # optimizers
        G_optimizer = torch.optim.Adam(generator.parameters(), lr=lrG, betas=(beta1, beta2))
        D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lrD, betas=(beta1, beta2))
        
    all_loss, g_losses, d_losses = 0
    l2_loss = nn.MSELoss(reduction='sum')
    for i, data in enumerate(dataloader):      
        if training:
            optimizer.zero_grad()

        img_seq = torch.FloatTensor(data['img_seq'])
        pose_seq = data['pose_seq'].to(device)
        label_seq = data['label_seq'].to(device)
        transl_eng = data['transl_eng']
        transl_deu = data['transl_deu']
        img_seq_len = data['seq_len']

        total_sequence = sum(np.array(img_seq_len))
        delta = label_seq - pose_seq

#         initial_delta = torch.zeros_like(delta)
        
        #test
#         initial_delta[:, :, ...] = delta[:, :, ...]
#         combined = torch.cat((pose_seq, initial_delta), dim=-2)
#         combined = torch.cat((initial_delta, pose_seq), dim=-2)

        lang_embed = torch.FloatTensor(encoder(transl_eng)).to(device)

        if training:
            output = decoder(lang_embed, max(img_seq_len), pose_seq.view(pose_seq.shape[0], pose_seq.shape[1], -1),\
                                  epoch=epoch)
        else:
            output = decoder.sample(lang_embed.to(device), cd 
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
            
        # pix2pix
        # preprocess batched joints to images

        # input & target image data
        x_ = Variable(input.cuda())
        y_ = Variable(target.cuda())
            
        if training:
            # Train discriminator with real data
            d_real_decision = generator(x_, y_).squeeze()
            real_ = Variable(torch.ones(d_real_decision.size()).cuda())
            d_real_loss = d_bce_loss(d_real_decision, real_)

            # Train discriminator with fake data
            gen_image = generator(x_)
            d_fake_decision = discriminator(x_, gen_image).squeeze()
            fake_ = Variable(torch.zeros(d_fake_decision.size()).cuda())
            d_fake_loss = d_bce_loss(d_fake_decision, fake_)

            # Back propagation
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            discriminator.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            gen_image = generator(x_)
            d_fake_decision = discriminator(x_, gen_image).squeeze()
            g_fake_loss = d_bce_loss(d_fake_decision, real_)

            # L1 loss
            g_l1_loss = g_l1_lambda * g_l1_loss(gen_image, y_)

            # Back propagation
            g_loss = g_fake_loss + g_l1_loss
            generator.zero_grad()
            g_loss.backward()
            g_optimizer.step()
        else:
            gen_images = generator(x_)
            gen_images = gen_image.cpu().data
            g_loss = l2_loss(gen_images[:img_seq.shape[0]], img_seq)
        
        if training:
            loss.backward()
            optimizer.step()
            if i % update == 0:
                iterations = epoch*len(dataloader) + i        
                print('Epoch: {} | Iteration: {} | Pose Loss: {} | G Loss: {} | D Loss: {} '.format(epoch, i, \
                                                                                                    loss.cpu().item() \
                                                                                                    G_loss.cpu().item(), \
                                                                                                    D_loss.cpu().item()))
                writer.add_scalar('Training Loss', loss, iterations)
                writer.add_figure('Training Predicted Pose', TB_vis_pose2D(packed, packed_gt, normalize_poses), global_step=iterations)
                writer.add_figure('Training Predicted Frame', global_step=iterations)
                                
        all_loss += loss.detach().cpu()
        g_losses += g_loss.detach().cpu()
        d_losses += d_loss.detach().cpu()
    
    if training:
        return {
                'model': decoder,
                'optimizer': optimizer,
                'pose-loss': all_loss / len(dataloader),
                'g-model': generator,
                'g-optimizer': g_optimizer,
                'd-model': discriminator,
                'd-optimizer': d_optimizer,
                'g-loss': g_losses / len(dataloader),
                'd-loss': d_losses / len(dataloader)
        }
    else:
        writer.add_scalar('Validation Loss', all_loss / len(dataloader), epoch)
        writer.add_figure('Validation Predicted Pose', TB_vis_pose2D(packed, packed_gt, normalize_poses), global_step=epoch)
        writer.add_figure('Validation Predicted Frame', )
        return {
                'pose-loss': all_loss / len(dataloader),
                'gan-loss': g_losses / len(dataloader)
        }

