import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import utils.constants as constants
matplotlib.use('Agg')

BODY_25_EXTRA_JOINTS = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]
BODY_25_KEEP_SLICES = [[0,10], [12,13], [15,19], [25]]

BODY_BONES = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [8,9], [8,10], 
         [0,11],[0,12], [11,13], [12,14]]

HAND_BONES = [[0,1], [1,2], [2,3], [3,4], 
              [0,5], [5,6], [6,7], [7,8],
              [0,9], [9,10], [10,11], [11,12],
              [0,13], [13,14], [14,15], [15,16],
              [0,17], [17,18], [18,19], [19,20]
             ]

GLOBAL_BONES = [*BODY_BONES, *(np.array(HAND_BONES)+15).tolist(), *(np.array(HAND_BONES)+36).tolist()]

def root_center_pose(pose, poseType='BODY_25', root_joint=1):
    pose_clone = pose.clone()
    if poseType == 'BODY_25':
        pose_clone = pose_clone - pose_clone[root_joint,...] # center the pose around the "bottom of the neck"?
        return pose_clone, pose[root_joint,...]

def remove_excess_joints(pose, poseType='BODY_25'):
    # Input {pose} should be a tensor
    pose_trimmed = None
    if poseType=='BODY_25':
        pose_trimmed = torch.cat((pose[BODY_25_KEEP_SLICES[0][0]:BODY_25_KEEP_SLICES[0][1],...],
                                  pose[BODY_25_KEEP_SLICES[1][0]:BODY_25_KEEP_SLICES[1][1],...],
                                  pose[BODY_25_KEEP_SLICES[2][0]:BODY_25_KEEP_SLICES[2][1],...],
                                  pose[BODY_25_KEEP_SLICES[3][0]:,...]))
        
    return pose_trimmed

def normalize_pose(pose, mean, std, root_joint=1):
    normalized_joints = pose.clone()
    normalized_joints[:root_joint,:] = torch.div(pose[:root_joint,:] - mean[:root_joint,:], std[:root_joint,:])
    normalized_joints[root_joint,:] = pose[root_joint,:] - mean[root_joint,:]
    normalized_joints[root_joint+1:,:] = torch.div(pose[root_joint+1:,:] - mean[root_joint+1:,:], std[root_joint+1:,:])
    return normalized_joints

def denormalize_pose(pose, mean=constants.SAMPLE_MEAN_BODY_25, std=constants.SAMPLE_STD_BODY_25):
    return torch.mul(pose, std) + mean 

def pose2pytorch(pose, dim1=776, dim2=578):
    p = torch.clone(pose)
    p[:,0] = (p[:,0] / dim1 * 2) 
    p[:,1] = (p[:,1] / dim2 * 2) 
    return p

def pytorch2pose(pose, dim1=776, dim2=578):
    p = torch.clone(pose)
    p[:,:,0] = (p[:,:,0]) / 2 * dim1  
    p[:,:,1] = (p[:,:,1]) / 2 * dim2
    return p

# VISUALIZATION CODE
def plot_pose2D(ax, pose_2d_1, bones=GLOBAL_BONES, linewidth=1, alpha=0.95, colormap='gist_rainbow', autoAxisRange=True):
    cmap = plt.get_cmap(colormap)
    pose_2d = pose_2d_1.clone()
    pose_2d = np.reshape(pose_2d.numpy().transpose(), (2, -1))

    X, Y = np.squeeze(np.array(pose_2d[0, :])), np.squeeze(np.array(pose_2d[1, :]))
    XY = np.vstack([X, Y])

    maximum = len(bones)
    
    for i, bone in enumerate(bones):
        colorIndex = cmap.N - cmap.N * i/float(maximum) # cmap.N - to start from back (nicer color)
        color = cmap(int(colorIndex))
        ax.plot(XY[0, bone], XY[1, bone], color=color, linewidth=linewidth, alpha=alpha, solid_capstyle='round')
        
    # HARDCODED TO MATCH THE ORIGINAL IMAGE SIZE
    ax.set_xlim([-388,388])
    ax.set_ylim([378,-200]) # backwards because image and plot coordinates are opposite in y axis

def prep_poses(packed, packed_gt, num_joints=57, joint_dim=2, normalize=True):
    import utils.dataset as dataset
    pose, len_pose  = dataset.unpad_sequence(packed)
    pose_gt, len_gt = dataset.unpad_sequence(packed_gt)
    
    vis_pose = pose[0,:len_pose[0],:].view(-1, num_joints, 2)[:,:num_joints, :].detach().cpu()
    vis_gt = pose_gt[0,:len_gt[0], :num_joints, ...].detach().cpu()
    
    if normalize:
        vis_pose = denormalize_pose(vis_pose, constants.SAMPLE_MEAN_BODY_25, constants.SAMPLE_STD_BODY_25)
        vis_gt = denormalize_pose(vis_gt, constants.SAMPLE_MEAN_BODY_25, constants.SAMPLE_STD_BODY_25)
    else:
        vis_pose = pytorch2pose(vis_pose)
        vis_gt = pytorch2pose(vis_gt)
    return vis_pose, vis_gt, len_pose[0].item(), len_gt[0].item()

def TB_vis_pose2D(packed, packed_gt, normalize):
    pose_2d, gt_2d, len_pose, len_gt = prep_poses(packed, packed_gt, normalize=normalize)
    
    if len_pose > 1 or len_gt > 1:
        plot_len = min(len_gt, 20)
        fig, ax = plt.subplots(1,plot_len)
        fig.set_size_inches(plot_len*2, 2)
        
        for i in range(plot_len):
            plot_pose2D(ax[i], pose_2d[i], colormap='gist_rainbow')
            plot_pose2D(ax[i], gt_2d[i], colormap='copper')
    else:
        plot_len = min(len_gt, 20)
        fig, ax = plt.subplots(1,plot_len)
        for i in range(plot_len):
            plot_pose2D(ax, pose_2d[i,...], colormap='gist_rainbow')
            plot_pose2D(ax, gt_2d[i,...], colormap='copper')
    return fig
    
    
def calculate_batch_mpjpe(output, label):
    difference =  output - label 
    square_difference = torch.square(difference) 
    sum_square_difference_per_point = torch.sum(square_difference, dim=2) 
    euclidean_distance_per_point = torch.sqrt(sum_square_difference_per_point) 
    mpjpe = torch.mean(euclidean_distance_per_point)
    return mpjpe