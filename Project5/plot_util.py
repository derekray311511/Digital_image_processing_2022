import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def plot_and_save(demo_name, imgs, filenames, type='gray', dpi=200, plot=True, save=True, col_size=4, show_axis=1):
    if type == 'rgb':
        type = None
    size = len(imgs)
    if save:
        root = 'results'
        result_path = os.path.join(root, demo_name)
        imgs_path = os.path.join(result_path, 'imgs')
        if not os.path.exists(root):
            os.mkdir(root)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        if not os.path.exists(imgs_path):
            os.mkdir(imgs_path)
        for i in range(size):
            img_path = imgs_path + '/' + filenames[i] + '.png'
            plt.figure()
            plt.title(filenames[i])
            if not show_axis: plt.axis('off')
            plt.tight_layout()
            plt.imshow(imgs[i], cmap=type)
            plt.savefig(img_path, bbox_inches='tight', dpi=dpi)
        plt.close('all')

    if plot:
        plt.figure(figsize=(16, 10))
        img_in_row = col_size
        r = size // img_in_row + (size % img_in_row > 0)
        c = img_in_row
        for i in range(size):
            plt.subplot(r, c, i+1)
            plt.title(filenames[i])
            if not show_axis: plt.axis('off')
            plt.imshow(imgs[i], cmap=type)
        plt.suptitle(demo_name + ' images')
        plt.tight_layout()
        plt.show()
        plt.close('all')

def show_hist(demo_name, gray_imgs, filenames, plot=True, save=True, density=True):
    size = len(gray_imgs)
    if save:
        root = 'results'
        result_path = os.path.join(root, demo_name)
        hist_path = os.path.join(result_path, 'hist')
        if not os.path.exists(root):
            os.mkdir(root)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        if not os.path.exists(hist_path):
            os.mkdir(hist_path)
        for i in range(size):
            img_path = hist_path + '/' + filenames[i] + '.png'
            plt.figure()
            plt.title(filenames[i])
            plt.hist(gray_imgs[i].ravel(), 256, [0,256])
            plt.tight_layout()
            plt.savefig(img_path, bbox_inches='tight')
        plt.close('all')

    if plot:
        plt.figure(figsize=(16, 9))
        r = size // 4 + (size % 4 > 0)
        c = 4
        for i in range(size):
            plt.subplot(r, c, i+1)
            plt.title(filenames[i])
            plt.hist(gray_imgs[i].ravel(), 256, [0,256])
        plt.suptitle(demo_name + ' histograms')
        plt.tight_layout()
        plt.show()
        plt.close('all')

    hists = []
    for i in range(size):
        hists.append(np.histogram(gray_imgs[i].ravel(), 256, [0, 256], density=density))
    return hists