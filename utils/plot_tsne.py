#coding=utf-8
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math
import pickle
import os

text_64p64_subject = {1:'s', 2:'sh', 3:'e', 4:'u/hu', 5:'C', 6:'sh', 7:'s', 8:'s', 9:'s',
                      10:'C', 11:'a', 12:'i', 13:'s', 14:'a ', 15:'e', 16:'e', 17:'u', 18:'a', 19:'e',
                      20:'i', 21:'s', 22:'a', 23:'C', 24:'e', 25:'o', 26:'C', 27:'a', 28:'C', 29:'C',
                      30:'s', 31:'x', 32:'e', 33:'C', 34:'C', 35:'sil', 36:'s', 37:'C', 38:'u',
                      39:'e',
                      40:'C', 41:'a', 42:'C', 43:'a', 44:'C', 45:'C', 46:'C', 47:'a', 48:'e', 49:'C',
                      50:'s', 51:'a', 52:'e', 53:'i', 54:'e', 55:'e', 56:'C', 57:'s', 58:'a',
                      59:'e',
                      60:'e', 61:'i', 62:'s', 63:'a'}


# text_64l64_subject = {0:'nontar', 1:'s', 2:'hu', 3:'e', 4:'e', 5:'s', 6:'e', 7:'he', 8:'s/sh', 9:'x',
#                       10:'C', 11:'e/u', 12:'i', 13:'z', 14:'a', 15:'e', 16:'i', 17:'u', 18:'a/3', 19:'a/e',
#                       20:'-', 21:'u', 22:'e', 23:'u/hu', 24:'i', 25:'z/s', 26:'u/e', 27:'a', 28:'u/hu', 29:'r',
#                       30:'3/u', 31:'-', 32:'V', 33:'C', 34:'u', 35:'a', 36:'C', 37:'e/u', 38:'o',
#                       39:'u/o',
#                       40:'C', 41:'e', 42:'hu', 43:'e/a', 44:'C', 45:'-', 46:'-', 47:'u', 48:'hu', 49:'-',
#                       50:'s', 51:'a ', 52:'u/e', 53:'e', 54:'C', 55:'o', 56:'-', 57:'sh', 58:'e',
#                       59:'V',
#                       60:'e/i', 61:'o/u', 62:'sh', 63:'e'}

def plot_2d_dots(tsne_samples_fp):
    with open(tsne_samples_fp, 'rb') as fr:
        samples = pickle.load(fr)
    plt.scatter(samples[:, 0], samples[:, 1], s=2, c='b')
    plt.show()

def plot_2d_dots_multi(tsne_samples_fps):
    num_fp = len(tsne_samples_fps)
    fig = plt.figure(figsize=(20, 20))
    colors = ['Purples', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlGn', 'PuBuGn']
    for i, fp in enumerate(tsne_samples_fps):
        if i == 3:
            # plt.subplot(2, (num_fp + 1)//2, i + 1)
            with open(fp, 'rb') as fr:
                samples = pickle.load(fr)

            plt.scatter(samples[:, 0], samples[:, 1], s=1, c=range(samples.shape[0]), cmap='hsv')
            plt.xticks([])
            plt.yticks([])
            for s in range(1, samples.shape[0]):
                plt.text(samples[s, 0]*1.01, samples[s, 1]*1.01, text_64p64_subject[s], fontsize=50,
                         color="r", style="italic", weight="light", verticalalignment='center',
                         horizontalalignment='right',
                         rotation=0)
            # plt.xlabel(fp.strip('.pkl').split('/')[-1])
    # plt.scatter(track_samples[:, 0], track_samples[:, 1], s=2, c=color[:], alpha=0.5)
    plt.show()

def plot_2d_dots_kde(x, y, c='Blues'):
    # Meshgrid
    xmin, xmax = -40, 40
    ymin, ymax = -40, 40
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # Peform the kernel density estimate
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    # Kernel density estimate plot
    plt.imshow(np.rot90(f), cmap=c, extent=[xmin, xmax, ymin, ymax])
    # Contour plot
    cset = plt.contour(xx, yy, f, colors='k')
    # Label plot
    plt.clabel(cset, inline=1, fontsize=8)

def plot_2d_track_split(tsne_samples_fp):
    with open(tsne_samples_fp, 'rb') as fr:
        samples = pickle.load(fr)
    # colors = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    # colors = ['Purples', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlGn', 'PuBuGn']
    plot_2d_dots_kde(samples[:, 0], samples[:, 1])
    # plt.scatter(track_samples[:, 0], track_samples[:, 1], s=2, c=color[:], alpha=0.5)
    plt.show()


if __name__ == "__main__":
    middle_save_dir = '/home/huangqinlong/Workspace/PhonemeExtraction/target_files/'
    tsne_fps = ['{}_transformed_targets=64_interpolation=linear_hidden=64.pkl', '{}_transformed_targets=64_interpolation=linear_hidden=256.pkl', 
                '{}_transformed_targets=64_interpolation=pchip_hidden=256.pkl','{}_transformed_targets=128_interpolation=linear_hidden=64.pkl', 
                '{}_transformed_targets=500_interpolation=linear_hidden=512.pkl', '{}_transformed_targets=500_interpolation=pchip_hidden=512.pkl', 
                '{}_transformed_targets=500_interpolation=pchip_hidden=512_l1loss.pkl']
    tsne_samples_fp = os.path.join(middle_save_dir, 'tsne_transformed_targets=500_interpolation=linear_hidden=512.pkl')
    # plot_2d_dots(tsne_samples_fp)
    # plot_2d_track_split(tsne_samples_fp)
    # method_list = ['tsne_per50_lr10_iter10000', 'tsne', 'tsne_per10_lr200_iter10000', 'tsne_per30_lr200_iter10000', 'tsne_per50_lr10_iter100000', 
    #                 'tsne_per50_lr10_iter10000', 'tsne_per50_lr200_iter100000', 'tsne_per50_lr500_iter10000', 'tsne_per5_lr200_iter10000']
    method_list = ['tsne_per10_lr50_iter20000', 'tsne_per10_lr50_iter100000','tsne_per10_lr200_iter10000', 'tsne_per10_lr200_iter100000', 'tsne_per10_lr500_iter10000', 'tsne_per5_lr200_iter10000']
    # method_list = ['PCA', 'FastICA', 'KernelPCA', 'LLE', 'LTSA', 'Hessian LLE', 'Modified LLE', 'SE', 'Isomap', 'MDS']
    # for method in method_list:
    #     print('showing results of {}'.format(method))
    #     plot_2d_dots_multi([os.path.join(middle_save_dir, fp.format(method)) for fp in tsne_fps])

    tsne_fps = ['{}_transformed_targets=64_interpolation=pchip_hidden=256.pkl']
    for tsne_fp in tsne_fps:
        print('showing results of {}'.format(tsne_fp))
        plot_2d_dots_multi([os.path.join(middle_save_dir, tsne_fp.format(method)) for method in method_list])