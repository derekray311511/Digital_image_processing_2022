import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def plot_and_save(demo_name, gray_imgs, filenames, dpi=200, plot=True, save=True, cv_plot=False):
    size = len(gray_imgs)
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
            plt.tight_layout()
            plt.imshow(gray_imgs[i], cmap='gray')
            plt.savefig(img_path, bbox_inches='tight', dpi=dpi)
        plt.close('all')

    if plot and not cv_plot:
        plt.figure(figsize=(16, 8))
        img_in_row = 4
        r = size // img_in_row + (size % img_in_row > 0)
        c = img_in_row
        for i in range(size):
            plt.subplot(r, c, i+1)
            plt.title(filenames[i])
            plt.imshow(gray_imgs[i], cmap='gray')
        plt.suptitle(demo_name + ' images')
        plt.tight_layout()
        plt.show()
        plt.close('all')
    
    if cv_plot:
        c = size
        p = 0
        concated_img = []
        for i in range(size):
            cv2.putText(gray_imgs[i], filenames[i], (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 1, cv2.LINE_AA)
        blanks = 4 - size % 4
        for _ in range(blanks):
            gray_imgs.append(np.zeros(gray_imgs[0].shape, dtype='uint8'))
            cv2.putText(gray_imgs[-1], 'BLANK IMG', (gray_imgs[0].shape[1]//2-80, gray_imgs[0].shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        while c > 0:
            concated_img.append(cv2.hconcat(gray_imgs[p:p+4]))
            c -= 4
            p += 4
        final_img = cv2.vconcat(concated_img)
        final_img = cv2.resize(final_img, (1280, 720))
        cv2.imshow('imgs', final_img)
        cv2.moveWindow('imgs', 0, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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