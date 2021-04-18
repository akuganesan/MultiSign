import os
import cv2
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

def pose2video(sentence, gt_pose, gt_image, pred_pose, gen_image, epoch, training=True, save=False, \
               save_dir='results/', show=False, fig_size=(5, 5)):
    '''
    Visualize Pose2Video results
    input:     joint image
    target:    ground truth video frame
    gen_image: predicted image from the generator
    sentence:  sentence to translate
    training:
        Set training == True if visualizing validation or training results
        Set training == False if visualizing test results
    '''
    if not training:
        fig_size = (gt_pose.size(2) * 5 / 100, gt_pose.size(3)/100)
    
    img = Image.new('RGB', (gt_pose.size(2), gt_pose.size(3)), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, gt_pose.size(3)//2), sentence, fill=(0,0,0))
    PILtoTensor = transforms.ToTensor()
    sentence_image = PILtoTensor(img).expand(3, -1, -1).unsqueeze(0)
        
    fig, axes = plt.subplots(1, 5, figsize=fig_size)
    imgs = [sentence_image, gt_pose, gt_image, pred_pose, gen_image]
    for ax, img in zip(axes.flatten(), imgs):
        ax.axis('off')
        ax.set_adjustable('box')
        # Scale to 0-255
        img = (((img[0] - img[0].min()) * 255) / (img[0].max() - img[0].min())).numpy().transpose(1, 2, 0).astype(np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)

    if training:
        title = 'Epoch {0}'.format(epoch + 1)
        fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if training:
            save_fn = save_dir + 'Result_epoch_{:d}'.format(epoch+1) + '.png'
        else:
            save_fn = save_dir + 'Test_result_{:d}'.format(epoch+1) + '.png'
            fig.subplots_adjust(bottom=0)
            fig.subplots_adjust(top=1)
            fig.subplots_adjust(right=1)
            fig.subplots_adjust(left=0)
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()