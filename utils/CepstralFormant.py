# -*- coding: utf-8 -*-
# @File: PhonemeExtraction/CepstralFormant.py
# @Author: Qinlong Huang
# @Create Date: 2021/06/12 11:50
# @Contact: qinlonghuang@gmail.com
# @Description:

from scipy.signal import lfilter
import librosa
import numpy as np
import matplotlib.pyplot as plt


def local_maxium(x):
    """
    求序列的极大值
    :param x:
    :return:
    """
    d = np.diff(x)
    l_d = len(d)
    maxium = []
    loc = []
    for i in range(l_d - 1):
        if d[i] > 0 and d[i + 1] <= 0:
            maxium.append(x[i + 1])
            loc.append(i + 1)
    return maxium, loc


def lpc_coeff(s, p):
    """
    :param s: 一帧数据
    :param p: 线性预测的阶数
    :return:
    """
    n = len(s)
    # 计算自相关函数
    Rp = np.zeros(p)
    for i in range(p):
        Rp[i] = np.sum(np.multiply(s[i + 1:n], s[:n - i - 1]))
    Rp0 = np.matmul(s, s.T)
    Ep = np.zeros((p, 1))
    k = np.zeros((p, 1))
    a = np.zeros((p, p))
    # 处理i=0的情况
    Ep0 = Rp0
    k[0] = Rp[0] / Rp0
    a[0, 0] = k[0]
    Ep[0] = (1 - k[0] * k[0]) * Ep0
    # i=1开始，递归计算
    if p > 1:
        for i in range(1, p):
            k[i] = (Rp[i] - np.sum(np.multiply(a[:i, i - 1], Rp[i - 1::-1]))) / Ep[i - 1]
            a[i, i] = k[i]
            Ep[i] = (1 - k[i] * k[i]) * Ep[i - 1]
            for j in range(i - 1, -1, -1):
                a[j, i] = a[j, i - 1] - k[i] * a[i - j - 1, i - 1]
    ar = np.zeros(p + 1)
    ar[0] = 1
    ar[1:] = -a[:, p - 1]
    G = np.sqrt(Ep[p - 1])
    return ar, G


def Formant_Cepst(u, cepstL):
    """
    倒谱法共振峰估计函数
    :param u:输入信号
    :param cepstL: 频率上窗函数的宽度
    :return: val共振峰幅值
    :return: loc共振峰位置
    :return: spec包络线
    """
    wlen2 = len(u) // 2
    u_fft=np.fft.fft(u)                         #按式（2-1）计算
    U = np.log(np.abs( u_fft[:wlen2]))
    Cepst = np.fft.ifft(U)                      #按式（2-2）计算
    cepst = np.zeros(wlen2, dtype=np.complex)
    cepst[:cepstL] = Cepst[:cepstL]             #按式（2-3）计算
    cepst[-cepstL + 1:] = Cepst[-cepstL + 1:]	#取第二个式子的相反
    spec = np.real(np.fft.fft(cepst))
    val, loc = local_maxium(spec)               #在包络线上寻找极大值
    return val, loc, spec


def Formant_Interpolation(u, p, fs):
    """
    插值法估计共振峰函数
    :param u:
    :param p:
    :param fs:
    :return:
    """
    ar, _ = lpc_coeff(u, p)
    U = np.power(np.abs(np.fft.rfft(ar, 2 * 255)), -2)
    df = fs / 512
    val, loc = local_maxium(U)
    ll = len(loc)
    pp = np.zeros(ll)
    F = np.zeros(ll)
    Bw = np.zeros(ll)
    for k in range(ll):
        m = loc[k]
        m1, m2 = m - 1, m + 1
        p = val[k]
        p1, p2 = U[m1], U[m2]
        aa = (p1 + p2) / 2 - p
        bb = (p2 - p1) / 2
        cc = p
        dm = -bb / 2 / aa
        pp[k] = -bb * bb / 4 / aa + cc
        m_new = m + dm
        bf = -np.sqrt(bb * bb - 4 * aa * (cc - pp[k] / 2)) / aa
        F[k] = (m_new - 1) * df
        Bw[k] = bf * df
    return F, Bw, pp, U, loc


def Formant_Root(u, p, fs, n_frmnt):
    """
    LPC求根法的共振峰估计函数
    :param u:
    :param p:
    :param fs:
    :param n_frmnt:
    :return:
    """
    ar, _ = lpc_coeff(u, p)
    U = np.power(np.abs(np.fft.rfft(ar, 2 * 255)), -2)
    const = fs / (2 * np.pi)
    rts = np.roots(ar)
    yf = []
    Bw = []
    for i in range(len(ar) - 1):
        re = np.real(rts[i])
        im = np.imag(rts[i])
        fromn = const * np.arctan2(im, re)
        bw = -2 * const * np.log(np.abs(rts[i]))
        if fromn > 150 and bw < 700 and fromn < fs / 2:
            yf.append(fromn)
            Bw.append(bw)
    return yf[:min(len(yf), n_frmnt)], Bw[:min(len(Bw), n_frmnt)], U


if __name__ == '__main__':

    plt.figure(figsize=(14, 12))
    path = "/data1/huangqinlong/PhonemeExtraction/target_wavs/targets=64_interpolation=linear_hidden=256/38.wav"

    data, fs = librosa.load(path)
    # 预处理-预加重
    u = lfilter([1, -0.99], [1], data)

    cepstL = 6
    wlen = len(u)
    wlen2 = wlen // 2
    # 预处理-加窗
    u2 = np.multiply(u, np.hamming(wlen))
    # 预处理-FFT,取对数
    U_abs = np.log(np.abs(np.fft.fft(u2))[:wlen2])
    # 4.3.1
    freq = [i * fs / wlen for i in range(wlen2)]
    val, loc, spec = Formant_Cepst(u, cepstL)
    plt.subplot(4, 1, 1)
    plt.plot(freq, U_abs, 'k')
    plt.title('频谱')
    plt.subplot(4, 1, 2)
    plt.plot(freq, spec, 'k')
    plt.title('倒谱法共振峰估计')
    for i in range(len(loc)):
        plt.subplot(4, 1, 2)
        plt.plot([freq[loc[i]], freq[loc[i]]], [np.min(spec), spec[loc[i]]], '-.k')
        plt.text(freq[loc[i]], spec[loc[i]], 'Freq={}'.format(int(freq[loc[i]])))
    # 4.3.2
    p = 12
    freq = [i * fs / 512 for i in range(256)]
    F, Bw, pp, U, loc = Formant_Interpolation(u, p, fs)

    plt.subplot(4, 1, 3)
    plt.plot(freq, U)
    plt.title('LPC内插法的共振峰估计')

    for i in range(len(Bw)):
        plt.subplot(4, 1, 3)
        plt.plot([freq[loc[i]], freq[loc[i]]], [np.min(U), U[loc[i]]], '-.k')
        plt.text(freq[loc[i]], U[loc[i]], 'Freq={:.0f}\nHp={:.2f}\nBw={:.2f}'.format(F[i], pp[i], Bw[i]))

    # 4.3.3

    p = 12
    freq = [i * fs / 512 for i in range(256)]

    n_frmnt = 4
    F, Bw, U = Formant_Root(u, p, fs, n_frmnt)

    plt.subplot(4, 1, 4)
    plt.plot(freq, U)
    plt.title('LPC求根法的共振峰估计')

    for i in range(len(Bw)):
        plt.subplot(4, 1, 4)
        plt.plot([freq[loc[i]], freq[loc[i]]], [np.min(U), U[loc[i]]], '-.k')
        plt.text(freq[loc[i]], U[loc[i]], 'Freq={:.0f}\nBw={:.2f}'.format(F[i], Bw[i]))

    plt.show()
