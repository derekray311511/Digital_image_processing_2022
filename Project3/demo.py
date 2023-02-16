import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from plot_util import plot_and_save, show_hist

def Gaussian_Filter(img_size, lowpass=True, D0=100):
    '''
    Return filtered img and filter H
    '''
    # Create Gaussin Filter: Low Pass Filter in frequenct domain
    M, N = img_size[0], img_size[1]
    H = np.zeros((M,N), dtype=np.float32)
    D0 = D0 # Cutoff frequency
    
    xv, yv = np.meshgrid(range(M), range(N), sparse=False)
    H = np.exp(-((xv - M/2)**2 + (yv - N/2)**2) / (2. * D0 * D0))

    if lowpass:
        pass
    else:
        H = 1 - H
    return H

class alpha_trimmed_mean_filter:
    def __init__(self, mask_size=5, alpha=16):
        self.mask_size = mask_size
        self.alpha = alpha

    def padding(self, img):
        pad_size = self.mask_size // 2
        pad_img = np.pad(img, pad_width=pad_size)
        return pad_img

    def filter(self, img):
        img = np.array(img)
        img_size = img.shape
        pad_img = self.padding(img)
        result_img = np.empty(img_size)
        kernel_size = self.mask_size

        f_i = []
        for i in range(img_size[0]):
            f_j = []
            for j in range(img_size[1]):
                startx = i
                starty = j
                f = pad_img[startx : startx + kernel_size, starty : starty + kernel_size]
                f_j.append(f.flatten())
            f_i.append(f_j)
        f = np.array(f_i).reshape(img_size[0], img_size[1], kernel_size**2)
        f = np.sort(f, axis=-1)
        result_img = np.mean(f[:, :, (self.alpha // 2): -(self.alpha // 2)], axis=-1)

        return result_img

class inverse_filter:
    def __init__(self, degradation_model):
        self.H = degradation_model

    def inverse(self, s):
        s = np.array(s)
        assert s.shape == self.H.shape
        S = np.fft.fft2(s)
        S = np.fft.fftshift(S)
        F = S / self.H
        F = np.fft.ifftshift(F)
        f = np.fft.ifft2(F)
        return np.real(f)
    
def demo():
    '''
    Assume system: \n
    Assume h is Gaussian blur \n
        f * h = s \n
        s + n = g \n
    Try to find f from g \n
    '''
    img_file = 'Kid2 degraded.tiff'
    img = cv2.imread(img_file, 0)
    g = img
    
    D0s = np.linspace(100, 250, 4, dtype=int)
    for D0 in D0s:
        # Denoise (g -> s)
        mask_size = 5
        alpha = 16
        filter = alpha_trimmed_mean_filter(mask_size, alpha)
        s = filter.filter(g)

        # Inverse filter (s -> f)
        H = Gaussian_Filter(s.shape, lowpass=True, D0=D0)
        inv_filter = inverse_filter(degradation_model=H)
        f = inv_filter.inverse(s)

        # Plot results
        name_list = []
        img_list = []
        name_list.append('origin_img'), img_list.append(g)
        name_list.append('denoise_img'), img_list.append(s)
        name_list.append('inv_filter_img'), img_list.append(f)
        plot_and_save(f'Kid_restoration_D0={str(D0)}', img_list, name_list, dpi=200, plot=False, save=True)
        show_hist(f'Kid_restoration_D0={str(D0)}', img_list, name_list, plot=False, save=True, density=False)

    # Anaylize noise
    n_hat = g - s
    n_hist = show_hist('Kid_restoration', [n_hat], filenames=['noise_hist'], plot=False, save=True)[0]
    n_mean = np.sum(n_hist[0] * n_hist[1][:-1])
    n_var = np.sum(n_hist[0] * (n_hist[1][:-1] - n_mean)**2)
    print(f'noise mean: {n_mean}')
    print(f'noise variance: {n_var}')

if __name__ == "__main__":
    demo()