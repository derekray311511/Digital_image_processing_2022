import numpy as np
import cv2
import os, argparse
import pandas as pd
import openpyxl
from PIL import Image
from matplotlib import pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser(description='Filters demo')
    parser.add_argument('--demo_name', type=str, default='kid', help='kid/fruit or others.type')
    parser.add_argument('--hist_file', type=str, default='Histograms.xlsx', help='Path of the histogram file')
    parser.add_argument('--no_save_img', dest='save_img', action='store_false', default=True)
    parser.add_argument('--no_show_img', dest='show_img', action='store_false', default=True)
    return parser

def save_PIL_image(demo_name, gray_img, filename, dpi=(200, 200)):
    root = 'results'
    result_path = os.path.join(root, demo_name)
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    result_path = result_path + '/' + filename + '.png'
    RGBimage = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    PILimage = Image.fromarray(RGBimage)
    PILimage.save(result_path, dpi=dpi)

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
        r = size // 4 + (size % 4 > 0)
        c = 4
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

def show_hist(demo_name, gray_imgs, filenames, plot=True, save=True):
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
            plt.hist(gray_imgs[i].ravel(), 256, [0,255])
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
            plt.hist(gray_imgs[i].ravel(), 256, [0,255])
        plt.suptitle(demo_name + ' histograms')
        plt.tight_layout()
        plt.show()
        plt.close('all')

def get_image_pdf(demo_name, gray_imgs, filenames, hist_file='Histograms.xlsx'):
    # Only save origin and final hist to excel
    size = len(gray_imgs)
    save_idx = [0, size-1]

    # Read excel file
    if not os.path.exists(hist_file):
        return
    df = pd.read_excel(hist_file)

    for i in range(size):
        count = np.zeros((256), dtype=np.int)
        if i not in save_idx:
            continue
        img = gray_imgs[i].ravel()
        for j in range(len(img)):
            count[int(img[j])] += 1

        if demo_name == 'kid':
            if i == 0:
                df.iloc[:256, 1] = count
            elif i == size-1:
                df.iloc[:256, 2] = count
        elif demo_name == 'fruit':
            if i == 0:
                df.iloc[:256, 3] = count
            elif i == size-1:
                df.iloc[:256, 4] = count
        else:
            return
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter('Histograms.xlsx', engine='openpyxl')
        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        print(f"Excel file updated! ({demo_name})")

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

def main(parser):
    args = parser.parse_args()
    demo_name = args.demo_name
    print(f"demo category \'{demo_name}\'")
    demo_name_list = ['kid', 'fruit']
    img_list = []
    img_name = []

    # origin img (a)
    if demo_name == demo_name_list[0]:
        img = cv2.imread('kid blurred-noisy.tif', 0)
    elif demo_name == demo_name_list[1]:
        img = cv2.imread('fruit blurred-noisy.tif', 0)
    else:
        img = cv2.imread(demo_name, 0)
    
    img_name.append('(a)origin')
    img_list.append(img)

    # Smoothed gradient (0)
    kernel_size = 5
    avg_kernel = (1/kernel_size**2) * np.ones((kernel_size, kernel_size), dtype='float32')
    img = cv2.filter2D(img, -1, kernel=avg_kernel)

    # Laplacian (b)(c)
    # img_Laplacian = cv2.Laplacian(img, -1, ksize=3)
    # img_Laplacian = cv2.convertScaleAbs(img_Laplacian)
    Laplacian_kernel = np.array([[-1, -1, -1],
                                 [-1,  8, -1],
                                 [-1, -1, -1]], dtype='float32')
    img_Laplacian = cv2.filter2D(img, -1, kernel=Laplacian_kernel)
    img_name.append('(b)Laplacian')
    img_list.append(abs(img_Laplacian))
    img_name.append('(c)Laplacian-sharpened')
    img_list.append(img_Laplacian + img)

    # Sobel gradient (d)
    gx = np.array([[ 1,  0, -1],
                   [ 2,  0, -2],
                   [ 1,  0, -1]], dtype='float32')
    gy = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype='float32')
    img_Sobel_gradient = cv2.filter2D(img, -1, kernel=gx) + cv2.filter2D(img, -1, kernel=gy)
    img_name.append('(d)Sobel-gradient')
    img_list.append(abs(img_Sobel_gradient))

    # Smoothed gradient (e)
    kernel_size = 5
    avg_kernel = (1/kernel_size**2) * np.ones((kernel_size, kernel_size), dtype='float32')
    img_smoothed = cv2.filter2D(img_Sobel_gradient, -1, kernel=avg_kernel)
    img_name.append('(e)Smoothed-gradient')
    img_list.append(img_smoothed)

    # Extract feature (f) = (e) * (b)
    img_feature = img_smoothed * img_Laplacian
    img_name.append('(f)Extracted-feature')
    img_list.append(img_feature)

    # stage 1 result (g) = (f) + (a)
    img_g = img + img_feature
    img_name.append('(g)Stage1-result')
    img_list.append(img_g)

    # Apply power-law transformation(gamma=0.5, c=1) to (g)
    # power-law s = c * r^gamma
    c = 1
    gamma = 0.5
    img_s = np.array((255 * c * (img_g / 255)**gamma), dtype='uint8')   # Prevent overflow
    img_name.append('(h)Stage2-result')
    img_list.append(img_s)

    # Preprocessing of plotting img (Map (rmin, rmax) to (0, 255))
    img_list_plot = []
    for img in img_list:
        img_list_plot.append(prep_plot(img))

    plot_and_save(demo_name, img_list_plot, img_name, dpi=200, plot=args.show_img, save=args.save_img, cv_plot=True)
    show_hist(demo_name, img_list, img_name, plot=args.show_img, save=args.save_img)
    get_image_pdf(demo_name, img_list, img_name, hist_file=args.hist_file)
    
if __name__ == "__main__":
    main(get_parser())