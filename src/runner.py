import torch
import random
import matplotlib
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import utils.constants as constants
import utils.skel as skel
import utils.dataset as dataset
import utils.visualize as vis

from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from utils.skel import TB_vis_pose2D, prep_poses

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
            encoder_input = transl_deu
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

def endToEnd_test(dataloader, encoder, decoder, generator, device, num_joints=57, joint_dim=2, encoder_type='multi'):
    
    encoder.eval()
    decoder.eval()
    generator.eval()
    
    # preprocessing
    TensortoPIL = transforms.ToPILImage()
    PILtoTensor = transforms.ToTensor()
        
    all_loss = 0
    MSE = torch.nn.MSELoss()
    L1 = torch.nn.L1Loss()
    l1 = 0
    psnr = 0
    count = 0
    for i, data in enumerate(dataloader):
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
            encoder_input = transl_eng
        elif encoder_type == 'en':
            encoder_input = transl_eng
        else:
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
        
        for image, pred_pose, gt_pose in zip(img_seq.squeeze(0)[1:], pred_pose, gt_label):
            # pred pose image
            ax = plt.subplot()
            plt.axis('off')
            skel.plot_pose2D(ax, pred_pose)
            ax.get_figure().canvas.draw()
            a = Image.frombytes('RGB', ax.get_figure().canvas.get_width_height(),ax.get_figure().canvas.tostring_rgb())
            ax.clear()
            
            # gt pose image
            ax = plt.subplot()
            plt.axis('off')
            skel.plot_pose2D(ax, gt_pose)
            ax.get_figure().canvas.draw()
            gt_pose_img = Image.frombytes('RGB', ax.get_figure().canvas.get_width_height(), \
                                          ax.get_figure().canvas.tostring_rgb())
            ax.clear()            
            
            # target image
            b = TensortoPIL(image)

            # Preprocess pose image and target image
            a = a.resize((286, 286), Image.BICUBIC)
            b = b.resize((286, 286), Image.BICUBIC)
            gt_pose_img = gt_pose_img.resize((286, 286), Image.BICUBIC)
            a = transforms.ToTensor()(a)
            b = transforms.ToTensor()(b)
            gt_pose_img = transforms.ToTensor()(gt_pose_img)
            w_offset = random.randint(0, max(0, 286 - 256 - 1))
            h_offset = random.randint(0, max(0, 286 - 256 - 1))

            a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
            b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]

            a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
            b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)
            gt_pose_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(gt_pose_img)

            if random.random() < 0.5:
                idx = [i for i in range(a.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                a = a.index_select(2, idx)
                b = b.index_select(2, idx)
                gt_pose_img = gt_pose_img.index_select(2, idx)
                
            a = a.unsqueeze(0)
            b = b.unsqueeze(0)
            gt_pose_img = gt_pose_img.unsqueeze(0)
                
            # input & target image data
            x_ = Variable(a.cuda())
            y_ = Variable(b.cuda())

            gen_image = generator(x_)
            gen_image = gen_image.cpu().data

            l1 = l1 + L1(b, gen_image)
            psnr = psnr + 20 * torch.log10(255.0 / torch.sqrt(MSE(b, gen_image)))
            
            # print gen_image and target image to file
            data_dir = '/scratch/abi/MultiSign/evaluations/mBERT/'
            pred = gen_image.clone()
            gt = b.clone()
            pred = (((pred[0] - pred[0].min()) * 255) / (pred[0].max() - pred[0].min())).numpy().transpose(1, 2, 0).astype(np.uint8)
            gt = (((gt[0] - gt[0].min()) * 255) / (gt[0].max() - gt[0].min())).numpy().transpose(1, 2, 0).astype(np.uint8)
            pred = Image.fromarray(pred)
            gt = Image.fromarray(gt)
            pred.save(data_dir + 'pred/{:d}.jpg'.format(count))
            gt.save(data_dir + 'gt/{:d}.jpg'.format(count))

            
#             if i == 5:
#                 # Show result for test data
#                 vis.pose2video(encoder_input[0], gt_pose_img, b, a, gen_image, count, training=False, save=True, \
#                                save_dir='/scratch/abi/MultiSign/eval/')
#                 print('%d images are generated.' % (count + 1))
            count +=1
       
    return {
            's2p loss': all_loss / len(dataloader),
            'p2v l1 loss': l1/count,
            'p2v psnr loss': psnr/count
    }