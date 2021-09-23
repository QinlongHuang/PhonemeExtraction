#coding-utf-8
import time
import numpy as np
from collections import OrderedDict
# from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn.decomposition import PCA, FastICA, KernelPCA
from tsnecuda import TSNE
from functools import partial
import glob
import os
import pickle
import multiprocessing


def tsne_transform(ori_samples_fp, tsne_samples_fp):
    '''
    Input:
        ori_samples_fp: pkl file contains a list of tvs samples(np.2darray), each with same label 
        tsne_samples_fp: pkl contains transformed tvs samples along with their labels
    '''
    print('Loading from {}...'.format(ori_samples_fp))
    samples = np.load(ori_samples_fp) # [num_samples, sample_dim]
    print('shape of track samples: {}'.format(samples.shape))

    #do t-SNE analysis
    print('Doing t-SNE transforming...')
    # tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=10, init='random', verbose=1)
    tsne = TSNE(n_components=2, perplexity=10.0, learning_rate=50, n_iter=20000, verbose=1)
    samples_transformed = tsne.fit_transform(samples)

    with open(tsne_samples_fp, 'wb') as fw:
        # pickle.dump([Y_tvs, tvs_labels, Y_track, track_labels], fw)
        pickle.dump(samples_transformed, fw)
    print('Transformed track samples saved in {}'.format(tsne_samples_fp))

def manifold_transform(ori_samples_fp, middle_samples_fp_format, n_components=2, n_neighbors=10):
    print('Loading from {}...'.format(ori_samples_fp))
    samples = np.load(ori_samples_fp) # [num_samples, sample_dim]
    
    # Set-up manifold methods
    LLE = partial(manifold.LocallyLinearEmbedding, n_neighbors, n_components, eigen_solver='dense')
    methods = OrderedDict()
    methods['PCA'] = PCA(n_components=n_components) #
    methods['FastICA'] = FastICA(n_components=n_components)
    methods['KernelPCA'] = KernelPCA(n_components=n_components)
    methods['LLE'] = LLE(method='standard') #50000,50min
    methods['LTSA'] = LLE(method='ltsa') #10000, 260s
    methods['Hessian LLE'] = LLE(method='hessian') # may 30000
    methods['Modified LLE'] = LLE(method='modified')
    methods['SE'] = manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors) #10000, 180s
    methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
    methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
    
    print('Doing manifold transfroming...')
    for label, method in methods.items():
        start_ = time.time()
        Y = method.fit_transform(samples)
        end_ = time.time()
        print("{}: {:.2f} sec".format(label, end_ - start_))
        lle_samples_fp = middle_samples_fp_format.format(label)
        with open(lle_samples_fp, 'wb') as fw:
            pickle.dump(Y, fw)
        print('Transformed tvs samples saved in {}'.format(lle_samples_fp))
    print('Done!') 


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    middle_save_dir = '/home/sunyifan/workplace/EMSSL_light/phoneme_extract/target_files/'
    target_fps = ['targets=128_interpolation=linear_hidden=64.npy', 'targets=500_interpolation=linear_hidden=512.npy',
                  'targets=500_interpolation=pchip_hidden=512_l1loss.npy', 'targets=500_interpolation=pchip_hidden=512.npy',
                  'targets=64_interpolation=linear_hidden=256.npy', 'targets=64_interpolation=linear_hidden=64.npy',
                  'targets=64_interpolation=pchip_hidden=256.npy']
    for fp in target_fps:
        ori_samples_fp = os.path.join(middle_save_dir, fp)
        tsne_samples_fp = os.path.join(middle_save_dir, 'tsne_per10_lr50_iter20000_transformed_{}.pkl'.format(fp.strip('.npy')))
        tsne_transform(ori_samples_fp, tsne_samples_fp)
        # middle_samples_fp_format = os.path.join(middle_save_dir, '{}_transformed_' + fp.replace('.npy', '.pkl'))
        # manifold_transform(ori_samples_fp, middle_samples_fp_format, n_components=2, n_neighbors=10)
    