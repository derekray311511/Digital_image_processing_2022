import numpy as np
import cv2
import os, argparse
from matplotlib import pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser(description='Filters demo')
    parser.add_argument('--no_save_img', dest='save_img', action='store_false', default=True)
    parser.add_argument('--no_show_img', dest='show_img', action='store_false', default=True)
    return parser

# Function to map each intensity level to output intensity level.
def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        if r1 == 0:
            return s1
        else:
            return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        if r2 == 255:
            return s2
        else:
            return ((255 - s2)/(255 - r2)) * (pix - r2) + s2

# Map (rmin, rmax) to (0, 255)
def prep_plot(img):
    # Find min and max value of img pixels
    rmin = np.amin(img)
    rmax = np.amax(img)
    # Define parameters.
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = 255
    # Vectorize the function to apply it to each value in the Numpy array.
    pixelVal_vec = np.vectorize(pixelVal)
    # Apply contrast stretching.
    contrast_stretched_img = pixelVal_vec(img, r1, s1, r2, s2)
    return np.array(contrast_stretched_img, dtype = 'uint8')

def plot_and_save(demo_name, gray_imgs, filenames, dpi=150, plot=True, save=True, cv_plot=False):
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

def Gaussian_Filter(img, lowpass=True):
    '''
    Return filtered img and filter H
    '''
    # Create Gaussin Filter: Low Pass Filter in frequenct domain
    M, N = img.shape
    img = abs(img)
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 100 # Cutoff frequency
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2)**2 + (v - N / 2)**2)
            H[u, v] = np.exp(-D**2 / (2 * D0 * D0))

    if lowpass:
        pass
    else:
        H = 1 - H
    return np.multiply(img, H), H

def padding_img(img, times):
    M, N = img.shape
    zeros_array = np.zeros((M, N * (times-1)), dtype=np.float32)
    img = np.hstack([img, zeros_array])
    zeros_array = np.zeros((M * (times-1), N * (times)), dtype=np.float32)
    img = np.vstack([img, zeros_array])
    return img

def create_minus_xy_array(img):
    M, N = img.shape
    ret = np.ones((M, N), dtype=np.float32)
    for x in range(M):
        for y in range(N):
            if (x + y) % 2:
                ret[x, y] = -1
    return ret

def main(parser):
    args = parser.parse_args()
    img_list = []
    img_name = []
    demo_names = ['kid']#, 'fruit']
    files_type = ['.tif']#, '.tif']

    for demo_name, file_type in zip(demo_names, files_type):
        file = demo_name + file_type
        print(f'Running {file}')

        f = cv2.imread(file, 0)
        f = padding_img(f, times=2)
        M,N = f.shape
        print(f.shape)
        F = np.fft.fft2(f)
        Fshift = np.fft.fftshift(F)
        H = np.zeros((M,N), dtype=np.float32)
        D0 = 10
        for u in range(M):
            for v in range(N):
                D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
                H[u,v] = np.exp(-D**2/(2*D0*D0))
        
        Gshift = Fshift * H
        G = np.fft.ifftshift(Gshift)
        g = np.abs(np.fft.ifft2(G))

        plt.figure(figsize=(5,5))
        plt.imshow(g, cmap='gray')
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(5,5))
        plt.imshow(np.log1p(np.abs(Gshift)), cmap='gray')
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(5,5))
        plt.imshow(np.log1p(np.abs(G)), cmap='gray')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main(get_parser())