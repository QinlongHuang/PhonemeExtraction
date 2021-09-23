# -*- coding: utf-8 -*-
# @File: PhonemeExtraction/plot_curves.py
# @Author: Qinlong Huang
# @Create Date: 2021/03/31 15:03
# @Contact: qinlonghuang@gmail.com
# @Description:

import scipy.interpolate as sci
import numpy as np
import matplotlib.pyplot as plt
import torch
from src import utils
from src.config import cfg


# find the a & b points
def get_bezier_coef(points):
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B


# returns the general Bezier cubic formula given 4 control points
def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d


# return one cubic curve for each consecutive points
def get_bezier_cubic(points):
    A, B = get_bezier_coef(points)
    return [
        get_cubic(points[i], A[i], B[i], points[i + 1])
        for i in range(len(points) - 1)
    ]


def get_sub_two_interpolation_func(x_: np.ndarray, y_: np.ndarray):
    def sub_two_interpolation_func(Lx):
        result = 0
        for index in range(len(x_) - 2):
            if x_[index] <= Lx <= x_[index + 2]:
                result = y_[index] * (Lx - x_[index + 1]) * (Lx - x_[index + 2]) / (x_[index] - x_[index + 1]) / (
                            x_[index] - x_[index + 2]) + \
                         y_[index + 1] * (Lx - x_[index]) * (Lx - x_[index + 2]) / (x_[index + 1] - x_[index]) / (
                                     x_[index + 1] - x_[index + 2]) + \
                         y_[index + 2] * (Lx - x_[index]) * (Lx - x_[index + 1]) / (x_[index + 2] - x_[index]) / (
                                     x_[index + 2] - x_[index + 1])
        return result

    return sub_two_interpolation_func


def plot_single_ori_rec_tvs(tvs_list):

    color_list = ['red', 'blue']
    tvs_dim_name = ['microInt', 'glotVol', 'aspVol', 'fricVol', 'fricPos', 'fricCF', 'fricBW',
                    'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'velum']


    # for tvs_idx in range(tvs_num):
    #     for tvs_dim in range(15):
    #         ax = fig.add_subplot(15, 3, 3 * tvs_dim + tvs_idx + 1)
    #         ax.plot(range(len(tvs_list[tvs_idx])), tvs_list[tvs_idx][:, tvs_dim], color=color_list[tvs_idx])

    fig, axs = plt.subplots(15, 1, figsize=(8, 8), sharex=True)
    for tvs_dim, ax in enumerate(axs):
        ax.plot(range(len(tvs_list[0])), tvs_list[0][:, tvs_dim], color=color_list[0], linestyle='solid')
        ax.plot(range(len(tvs_list[1])), tvs_list[1][:, tvs_dim], color=color_list[1], linestyle='dashed')
        ax.set_ylabel(tvs_dim_name[tvs_dim])
        ax.set_xlim(xmin=0)

    plt.figlegend(loc='upper right', labels=['original trajectory', 'reconstructed trajectory'])

    return fig


if __name__ == '__main__':

    # tvs = torch.randn(2, 200, 15)
    # fig = plot_single_ori_rec_tvs(tvs)
    # fig.show()

    # X = [0, 4, 5, 6, 10, 12, 18, 19, 22, 24]
    # Y = [1, 2, 4, 2, 6, 1, 9, 7, 4, 5]

    # x = torch.tensor(X, dtype=torch.float)
    # y = torch.randn(10, 5)
    #
    # new_x = torch.linspace(X[0], X[-1], 100, dtype=torch.float)
    #
    # f_pchi = TPchip(x, y)
    # y_pchi = f_pchi(new_x)
    #
    # fig, axes = plt.subplots(figsize=(6, 10), nrows=5, ncols=1, sharex=True)
    # for i in range(5):
    #     axes[i].scatter(x, y[:, i])
    #     axes[i].plot(new_x, y_pchi[:, i])
    #
    # fig.show()

    # x = np.array(X, dtype=np.float32)
    # y = np.array(Y, dtype=np.float32)
    # new_x = np.linspace(X[0], X[-1], 100, dtype=np.float32)
    #
    # plt.scatter(x, y)
    # 1,2,3无法外插，而B样条可以
    # 1. Cubic spline
    # f_cs = sci.interp1d(x, y, kind='cubic')
    # y_cs = f_cs(new_x)
    # plt.plot(new_x, y_cs, label='Cubic spline')

    # # 2. linear spline
    # f_ls = sci.interp1d(x, y, kind='slinear')
    # y_ls = f_ls(new_x)
    # plt.plot(new_x, y_ls, label='Linear spline')

    # 3. quadratic
    # f_qs = sci.interp1d(x, y, kind='quadratic')
    # y_qs = f_qs(new_x)
    # plt.plot(new_x, y_qs, label='Quadratic')

    # 4. Bezier interpolation
    # points = np.array([X, Y]).T
    # curves = get_bezier_cubic(points)
    # path = np.array([fun(t) for fun in curves for t in (x / 25)])
    # plt.plot(path[:, 0], path[:, 1], label='Bezier cubic interpolation')

    # 5. Cubic Hermite
    # Pchip stands for Piecewise Cubic Hermite Interpolation Polynomial
    # TODO: 看一下是否能保证要target是极值点
    # PCHIP的性质是保单调，所以确实会是极值点
    # Cubic Hermite本身是不用保单调的，但是需要给定一阶导信息
    # 如果有一阶导信息，则可以使用 sci.CubicHermiteSpline(x, y, dydx)
    # 否则使用PchipInterpolator
    # f_ch = sci.PchipInterpolator(x, y)
    # y_ch = f_ch(new_x)
    # plt.plot(new_x, y_ch, label='Cubic Hermite (Scipy)')

    # f_ch2 = utils.Pchip(x, y)
    # y_ch2 = f_ch2(new_x)
    # plt.plot(new_x, y_ch2, label='Cubic Hermite (Numpy)')

    # f_ch3 = utils.TPchip(torch.from_numpy(x), torch.from_numpy(y[:, None]))
    # y_ch3 = f_ch3(torch.from_numpy(new_x)).squeeze(-1)
    # plt.plot(new_x, y_ch3, label='Cubic Hermite (PyTorch)')

    # 5*. Cubic Hermite (PyTorch)
    # xt = torch.from_numpy(x).float()
    # yt = torch.from_numpy(y).float()
    # new_xt = torch.from_numpy(new_x).float()
    # ft_ch = TPchip(xt, yt)
    # yt_ch = ft_ch(new_xt)
    # plt.plot(new_xt, yt_ch, label='Cubic Hermite (PyTorch)')
    #
    # print((torch.from_numpy(y_ch) - yt_ch).sum())

    # 6. 分段二次插值
    # f_qx = get_sub_two_interpolation_func(x, y)
    # y_qx = [f_qx(i) for i in new_x]
    # plt.plot(new_x, y_qx, label='Quadratic spline')

    # 7. B-Spline(Basis Spline)
    # Any spline function of given degree can be expressed as a linear combination of B-splines of that degree.
    # splrep returns a tuple (t,c,k) containing
    # the vector of knots, the B-spline coefficients, and the degree of the spline.
    # t, c, k = sci.splrep(x, y, s=0, k=2)
    # bspl = sci.BSpline(t, c, k, extrapolate=True)
    # y_bspl = bspl(new_x)
    # plt.plot(new_x, y_bspl, label='b-spline, k={}'.format(k))

    # plt.legend()
    # plt.show()

    from src.models import GenerativeModelConditionDistribution
    import os
    import seaborn as sns
    generative_condition_model = GenerativeModelConditionDistribution(num_targets=cfg.MODEL.NUM_TARGETS,
                                                                      tvs_dim=cfg.MODEL.TVS_DIM,
                                                                      use_stds=cfg.TRAIN.USE_STDS,
                                                                      num_processes=cfg.TRAIN.NUM_PROCESSES,
                                                                      mode=cfg.MODEL.INTERPOLATION_MODE)

    checkpoint = torch.load(os.path.join(cfg.TRAIN.WS_MODEL_PATH, cfg.TRAIN.WS_MODEL_PKL))
    generative_condition_model.load_state_dict(checkpoint['generative_condition_model_state_dict'])
    means = None
    for name, param in generative_condition_model.named_parameters():
        means = param.detach().numpy()
        break
    fig, ax = plt.subplots()
    ax.set_xlim(-3, 3)
    y = ['microInt', 'glotVol', 'aspVol', 'fricVol', 'fricPos', 'fricCF', 'fricBW',
         'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'velum']
    sns.set(style="whitegrid")
    for i in range(cfg.MODEL.NUM_TARGETS):
        ax.errorbar(x=means[i], y=y)
    fig.show()

