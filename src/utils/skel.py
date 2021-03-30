import torch
import numpy as np
import matplotlib.pyplot as plt

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

def denormalize_pose(pose, mean, std):
    return torch.mul(pose, std) + mean 

# VISUALIZATION CODE
def plot_pose2D(ax, pose_2d_1, bones=GLOBAL_BONES, linewidth=2, alpha=0.95, colormap='gist_rainbow', autoAxisRange=True):
    cmap = plt.get_cmap(colormap)
    pose_2d = pose_2d_1.clone()
    pose_2d = np.reshape(pose_2d.numpy().transpose(), (2, -1))
#     print(pose_2d.shape)
#     pose_2d[1,:] *= -1

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