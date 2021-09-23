# -*- coding: utf-8 -*-
# @File: PhonemeExtraction/models.py
# @Author: Qinlong Huang
# @Create Date: 2021/03/31 14:34
# @Contact: qinlonghuang@gmail.com
# @Description:

import multiprocessing

import numpy as np
from functools import wraps
import time
import ray
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import grad
from scipy.interpolate import PchipInterpolator, interp1d
from src.utils import TPchip, Tinterp1d, diff


import warnings
warnings.simplefilter('ignore')

SILENCE = [0.27659574, -1., -1., -1., 0.57142857, -0.38461538, -1.,
           -0.39999999, -0.31304347, -0.46956521, -0.392, -0.16, -0.10434781, -0.99333333, -0.8]


def compute_running_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        running_time = time.time() - t1
        return res, running_time
    return wrapper


# @ray.remote
# def interpolate_for_a_z_sleep_linear(z, num_frame, target_means, target_stds):
#     """
#     重参数化：value = mu + n * std
#     :param z: np.ndarray: [max_num_frame]
#     :param num_frame: [1, ]
#     :param target_means: np.ndarray: [num_targets, tvs_dim]
#     :param target_stds: np.ndarray: [num_targets, tvs_dim] or None
#     :return:  tensor: [num_frame, tvs_dim]
#     """
#     last_target = 0
#     max_num_frame = z.shape[0]
#     tvs_dim = target_means.shape[1]
#
#     result = np.zeros((max_num_frame, tvs_dim))
#     for i in range(num_frame):  # We only need to process first num_frame steps
#         if z[i] == 0:  # 0 for non-target element
#             continue
#         else:
#             values_i = (target_means[z[i]] + np.random.normal() * target_stds[z[i]]).clip(-1, 1) \
#                 if target_stds is not None \
#                 else (target_means[z[last_target]] + np.random.normal()).clip(-1, 1)
#             points_to_interpolate = i - last_target - 1
#             if points_to_interpolate > 0:
#                 left_end = (target_means[z[last_target]] + np.random.normal() * target_stds[z[last_target]]) \
#                     .clip(-1, 1) if target_stds is not None \
#                     else (target_means[z[last_target]] + np.random.normal()).clip(-1, 1)
#                 right_end = values_i
#
#                 xs = np.linspace(0, points_to_interpolate + 1, 2)
#                 ys = np.array([left_end, right_end]).T  # [obs_embedding_dim, 2]
#                 f = interp1d(xs, ys, kind='slinear')
#                 x_new = np.linspace(0, points_to_interpolate + 1, points_to_interpolate + 2)
#                 y_new = f(x_new).T  # [points_to_interpolate + 2, obs_embedding_dim]
#                 for p in range(points_to_interpolate):
#                     assert result[i - p - 1].sum() == 0
#                     result[i - p - 1] = y_new[points_to_interpolate - p]
#
#             last_target = i
#             result[i] = values_i  # 当前位置是target，就直接赋值
#     if last_target != num_frame - 1:
#         left_end = (target_means[z[last_target]] + np.random.normal() * target_stds[z[last_target]]).clip(-1, 1) \
#             if target_stds is not None \
#             else (target_means[z[last_target]] + np.random.normal()).clip(-1, 1)
#         right_end = np.zeros((tvs_dim, ))
#         points_to_interpolate = num_frame - 1 - last_target - 1
#
#         xs = np.linspace(0, points_to_interpolate + 1, 2)
#         ys = np.array([left_end, right_end]).T  # [obs_embedding_dim, 2]
#         f = interp1d(xs, ys, kind='slinear')
#         x_new = np.linspace(0, points_to_interpolate + 1, points_to_interpolate + 2)
#         y_new = f(x_new).T  # [points_to_interpolate + 2, obs_embedding_dim]
#         for p in range(points_to_interpolate):
#             assert result[num_frame - 1 - p - 1].sum() == 0
#             result[num_frame - 1 - p - 1] = y_new[points_to_interpolate - p]
#
#     return torch.from_numpy(result)


@ray.remote
def interpolate_for_a_z_sleep(z, num_frame, target_means, target_stds, mode='pchip'):
    """
    重参数化：value = mu + n * std
    :param z: np.ndarray: [max_num_frame]
    :param num_frame: [1, ]
    :param target_means: np.ndarray: [num_targets, tvs_dim]
    :param target_stds: np.ndarray: [num_targets, tvs_dim] or None
    :param mode: linear or pchip
    :return:  tensor: [max_num_frame, tvs_dim]
    """
    max_num_frame = z.shape[0]
    tvs_dim = target_means.shape[1]
    silence = np.zeros((1, tvs_dim))

    target_idxs = z[:num_frame].nonzero()[0]  # 只包含target的序列，这里实际上是一个index！！！而不是target本身
    if len(target_idxs) == 0:  # 如果全不是target
        return torch.zeros(max_num_frame, tvs_dim)

    res = np.zeros((len(target_idxs), tvs_dim))
    # 由于每一帧的重参数化都需要使用不同的n，故这里无法做并行
    for i, target_idx in enumerate(target_idxs):
        res[i] = (target_means[z[target_idx]] + np.random.normal() * target_stds[z[target_idx]]).clip(-1, 1)

    # 边界条件
    # 1.如果全是target，则直接返回
    if len(target_idxs) == num_frame:
        rest = np.zeros((max_num_frame - num_frame, tvs_dim))
        res = np.concatenate([res, rest], axis=0)
        return torch.from_numpy(res)
    # 2. 首帧不是target，不管中间还有多少帧不是target，都认为从第一帧开始，发音器官就从静止态开始运动了
    if target_idxs[0] != 0:
        res = np.concatenate([silence, res], axis=0)
        target_idxs = np.concatenate([np.zeros(1), target_idxs])
    # 3. 尾帧不是target，不管中间还有多少帧不是target，都认为在最后一帧正好停止
    # 这两个起始不算很强的假设，如果预先做一下静音检测，去掉静音帧，问题应该不大
    if target_idxs[-1] != (num_frame - 1):  # 如果最后一个target后面还有non-target，这理论上是最常见的
        res = np.concatenate([res, silence], axis=0)
        target_idxs = np.concatenate([target_idxs, np.array([num_frame-1])])
    x_new = np.linspace(0, num_frame - 1, num_frame)

    if mode == 'pchip':
        f = PchipInterpolator(target_idxs, res)
    elif mode == 'linear':
        f = interp1d(target_idxs, res, kind='linear', axis=0)
    else:
        raise NotImplementedError('Please check your sleep interpolation mode !')

    res = f(x_new).clip(-1, 1)

    rest = np.zeros((max_num_frame - num_frame, tvs_dim))
    res = np.concatenate([res, rest], axis=0)

    return torch.from_numpy(res)


def sample_for_sleep(zs, num_frames, target_means, target_stds, mode):
    """

    :param zs: tensor: [num_samples, max_num_frame]
    :param num_frames: [num_samples]
    :param target_means: detached tensor: [num_targets, tvs_dim]
    :param target_stds: detached tensor: [num_targets, tvs_dim]
    :param num_processes
    :param mode
    :return:
    """
    num_samples = zs.shape[0]
    device = target_means.device
    zs = zs.cpu().numpy()
    num_frames = num_frames.cpu().numpy()
    target_means = target_means.numpy()
    target_stds = target_stds.numpy() if target_stds is not None else None

    # if mode == 'linear':
    #     interpolate_for_a_z_sleep = interpolate_for_a_z_sleep_linear
    # elif mode == 'pchip':
    #     interpolate_for_a_z_sleep = interpolate_for_a_z_sleep_pchip
    # else:
    #     raise NotImplementedError('Please check your mode !')

    # single process
    # tvs = list()
    #
    # for i in range(num_samples):
    #     tvs.append(interpolate_for_a_z_sleep(zs[i], num_frames[i],
    #                                          target_means, target_stds))

    # explicit multiprocess
    # tvs_list = list()
    # tvs_append = tvs_list.append
    # pool = multiprocessing.Pool(num_processes)
    # pool_apply_async = pool.apply_async
    # for i in range(num_samples):
    #     tvs_append(pool_apply_async(interpolate_for_a_z_sleep,
    #                                 (zs[i], num_frames[i], target_means, target_stds)))
    # pool.close()
    # pool.join()
    # tvs = [tvs.get() for tvs in tvs_list]

    # ray multiprocess
    tvs_list = [interpolate_for_a_z_sleep.remote(zs[i], num_frames[i], target_means, target_stds, mode=mode)
                for i in range(num_samples)]
    tvs = ray.get(tvs_list)

    result = torch.stack(tvs, dim=0).float()  # numpy的浮点默认是64位的double，而tensor为32位的float

    result = result.to(device)
    return result  # [num_samples, max_num_frame, obs_embedding_dim]


def get_sequence_mask(num_frames, max_len=None) -> torch.tensor:
    '''
    convert [2, 1, 3] into [[1, 1, 0],
                            [1, 0, 0],
                            [1, 1, 1]]
    max_len: The len to specifiy, >= max(sequence_lengths)
    '''
    if max_len is None:
        max_len = num_frames.max()
    mask_ = torch.arange(max_len).unsqueeze(0) < num_frames.unsqueeze(1)  # can't run '<' on cuda

    return mask_


class GradComputeModel(nn.Module):

    def __init__(self, target_means: np.ndarray, target_stds: np.ndarray = None, mode='pchip'):
        super(GradComputeModel, self).__init__()

        self.target_means = nn.Parameter(torch.tensor(target_means))
        self.target_stds = nn.Parameter(torch.tensor(target_stds)) if target_stds is not None else None

        self.mode = mode

    # def forward_linear(self, z):
    #     """
    #     Reconstruct a tvs for a sample
    #     :param z: 已经去padding的z [num_frame]
    #     :return: x: tensor [num_frame, tvs_dim]
    #     """
    #     last_target = 0
    #     num_frame = z.shape[0]
    #     tvs_dim = self.target_means.shape[1]
    #
    #     result = torch.zeros(num_frame, tvs_dim)
    #
    #     for i in range(0, num_frame):
    #         if z[i] == 0:  # 0 for non-target element
    #             continue
    #         else:
    #             values_i = (self.target_means[z[i]] + torch.randn(1) * self.target_stds[z[i]]).clamp(-1, 1) \
    #                 if self.target_stds is not None\
    #                 else (self.target_means[z[i]] + torch.randn(1)).clamp(-1, 1)
    #             points_to_interpolate = i - last_target - 1
    #             if points_to_interpolate > 0:
    #                 left_end = (self.target_means[z[last_target]] + torch.randn(1) * self.target_stds[z[last_target]]) \
    #                     .clamp(-1, 1) if self.target_stds is not None \
    #                     else (self.target_means[z[i]] + torch.randn(1)).clamp(-1, 1)
    #                 right_end = values_i
    #
    #                 ys = torch.stack([left_end, right_end]).T  # [obs_embedding_dim, 2]
    #                 y_new = F.interpolate(ys[None],
    #                                       size=(points_to_interpolate + 2),
    #                                       mode='linear',
    #                                       align_corners=True).squeeze().T
    #                 for p in range(points_to_interpolate):
    #                     assert result[i - p - 1].sum() == 0
    #                     result[i - p - 1] = y_new[points_to_interpolate - p]
    #
    #             last_target = i
    #             result[i] = values_i  # 当前位置是target，就直接赋值
    #     if last_target != num_frame - 1:
    #         left_end = (self.target_means[z[last_target]] + torch.randn(1) * self.target_stds[z[last_target]]) \
    #             .clamp(-1, 1) if self.target_stds is not None\
    #             else (self.target_means[z[last_target]] + torch.randn(1)).clamp(-1, 1)
    #         right_end = torch.zeros(tvs_dim, device=self.target_means.device)
    #         points_to_interpolate = num_frame - 1 - last_target - 1
    #
    #         ys = torch.stack([left_end, right_end]).T  # [obs_embedding_dim, 2]
    #         y_new = F.interpolate(ys[None],
    #                               size=(points_to_interpolate + 2),
    #                               mode='linear', align_corners=True).squeeze().T
    #         for p in range(points_to_interpolate):
    #             assert result[num_frame - 1 - p - 1].sum() == 0
    #             result[num_frame - 1 - p - 1] = y_new[points_to_interpolate - p]
    #
    #     return result

    def forward(self, z):
        """
        Reconstruct a tvs for a sample, interpolation mode = pchip
        :param z: 已经去padding的z [num_frame]
        :return: x: tensor [num_frame, tvs_dim]
        """
        num_frame = z.shape[0]
        tvs_dim = self.target_means.shape[1]
        silence = torch.zeros((1, tvs_dim), device=self.target_means.device)

        target_idxs = z.nonzero()[:, 0]  # index of target frame
        if len(target_idxs) == 0:
            return torch.zeros(num_frame, tvs_dim, requires_grad=True)

        res = torch.zeros(len(target_idxs), tvs_dim)
        # 由于每一帧的重参数化都需要使用不同的n，故这里无法做并行
        for i, target_idx in enumerate(target_idxs):
            res[i] = (self.target_means[z[target_idx]] + torch.randn(1) * self.target_stds[z[target_idx]]).clamp(-1, 1)

        # 边界条件
        # 1.如果全是target，则直接返回
        if len(target_idxs) == num_frame:
            return res
        # 2. 首帧不是target，不管中间还有多少帧不是target，都认为从第一帧开始，发音器官就运动了
        if target_idxs[0] != 0:
            res = torch.cat([silence, res], dim=0)
            target_idxs = torch.cat([torch.zeros(1), target_idxs])
        # 3. 尾帧不是target，不管中间还有多少帧不是target，都认为到最后一帧,发音器官正好停止
        # 这两个其实不算很强的假设，如果预先做一下静音检测，去掉静音帧，问题应该不大
        if target_idxs[-1] != (num_frame - 1):  # 如果最后一个target后面还有non-target
            res = torch.cat([res, silence], dim=0)
            target_idxs = torch.cat([target_idxs, torch.tensor([num_frame - 1])])

        if self.mode == 'linear':
            f = Tinterp1d(target_idxs, res)
        elif self.mode == 'pchip':
            f = TPchip(target_idxs, res)
        else:
            raise NotImplementedError('Please check your interpolation mode !')

        x_new = torch.linspace(0, num_frame - 1, num_frame)
        res = f(x_new).clamp(-1, 1)

        return res


@ray.remote
def grad_compute_for_a_sample(target_means, target_stds, x_expanded, z, mode, diff_lambda, diff_n):
    """
    :param target_means: np.ndarray [num_targets, tvs_dim]
    :param target_stds: np.ndarray [num_targets, tvs_dim] or None
    :param x_expanded: np.ndarray [num_frame, tvs_dim]
    :param z: np.ndarray [num_frame]
    :param mode: str interpolation mode, 'linear' or 'pchip'
    :param diff_lambda:
    :param diff_n:
    :return tensor: grad_means, grad_stds, scalar: loss_sample, np.ndarray: sample
    """
    model = GradComputeModel(target_means, target_stds, mode)

    # Forward prop.
    sample = model(torch.tensor(z))  # [num_frame, tvs_dim]
    # loss = F.mse_loss(torch.tensor(x_expanded), sample)
    reconstruction_loss = F.l1_loss(torch.tensor(x_expanded), sample)
    diff_loss = diff(sample, n=diff_n).abs().mean()
    loss = reconstruction_loss + diff_lambda * diff_loss

    # Backward prop.
    if target_stds is not None:
        grad_means, grad_stds = grad(loss, (model.target_means, model.target_stds), allow_unused=True)
        if type(grad_means) != torch.Tensor:  # 如果z全是0，就不会用到mean和std，它们就没有梯度，这时直接把梯度赋0
            grad_means = torch.zeros_like(model.target_means)
            grad_stds = torch.zeros_like(model.target_stds)
    else:
        grad_means = grad(loss, model.target_means)[0]
        if type(grad_means) != torch.Tensor:
            grad_means = torch.zeros_like(model.target_means)
        grad_stds = None
    loss_sample = loss.item()
    reconstruction_loss_sample = reconstruction_loss.item()
    diff_loss_sample = diff_loss.item()
    sample = sample.detach().numpy()

    del model, loss

    return grad_means, grad_stds, loss_sample, reconstruction_loss_sample, diff_loss_sample, sample


class InferenceModel(nn.Module):

    def __init__(self,
                 lstm_hidden_dim,
                 obs_embedding_dim=15,  # 发音轨迹16维，但是r1为常量
                 num_layers=1,
                 num_targets=64,
                 entropy_lambda=0.1,
                 non_target_lambda=0.1,
                 auto_regression=False
                 ):
        super().__init__()

        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_targets = num_targets
        self.entropy_lambda = entropy_lambda
        self.non_target_lambda = non_target_lambda
        self.auto_regression = auto_regression

        if auto_regression:
            self.lstm_cell = nn.LSTMCell(input_size=num_targets + obs_embedding_dim, hidden_size=self.lstm_hidden_dim)
            self.linear = nn.Linear(self.lstm_hidden_dim, num_targets)
        else:
            self.lstm_cell = nn.LSTM(input_size=obs_embedding_dim, hidden_size=self.lstm_hidden_dim,
                                     num_layers=num_layers, batch_first=True, bidirectional=True)
            # self.linear = nn.Linear(self.lstm_hidden_dim * 2
            #                         if self.lstm_cell.bidirectional else self.lstm_hidden_dim,
            #                         self.num_targets)
            self.classifier = nn.Linear(self.lstm_hidden_dim * 2
                                        if self.lstm_cell.bidirectional else self.lstm_hidden_dim,
                                        self.num_targets)
            # self.classifier = nn.Sequential(
            #     nn.Linear(self.lstm_hidden_dim * 2 if self.lstm_cell.bidirectional else self.lstm_hidden_dim,  # 512*2
            #               self.lstm_hidden_dim + 128),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(self.lstm_hidden_dim + 128, self.num_targets)
            # )

    def forward(self, xs, num_frames):
        """

        :param xs: [batch_size, max_num_frame, tvs_dim]
        :param num_frames: [batch_size]
        :return: logits: [batch_size, max_num_frame, num_targets]
        """
        # Only for Dynamic RNN mode
        assert not self.auto_regression
        packed_x = pack_padded_sequence(xs, num_frames, batch_first=True, enforce_sorted=False)
        packed_lstm_outputs, _ = self.lstm_cell(packed_x)
        lstm_outputs, _ = pad_packed_sequence(packed_lstm_outputs, batch_first=True)
        # logits = self.linear(lstm_outputs)
        logits = self.classifier(lstm_outputs)  # [batch_size, max_num_frame, num_targets]

        return logits

    def inference(self, xs):
        batch_size, max_num_frame, tvs_dim = xs.shape
        device = next(self.lstm_cell.parameters()).device
        lstm_inputs = torch.zeros((batch_size, self.num_targets + tvs_dim), device=device)
        h = torch.zeros((batch_size, self.lstm_hidden_dim), device=device)
        c = torch.zeros((batch_size, self.lstm_hidden_dim), device=device)
        lstm_inputs[:, -tvs_dim:] = xs[:, 0, :]

        zs = torch.zeros((batch_size, max_num_frame), device=device)
        for i in range(max_num_frame):
            h, c = self.lstm_cell(lstm_inputs, (h, c))
            logit = self.linear(h)  # [batch_size * num_samples, num_targets]
            _, z = logit.max(dim=-1)
            zs[:, i] = z
            if i != max_num_frame - 1:
                lstm_inputs = torch.cat(
                    [
                        F.one_hot(z, num_classes=self.num_targets).float(),  # [batch_size * num_samples, num_targets]
                        xs[:, i + 1, :]  # [batch_size, tvs_dim]
                    ], dim=-1
                )
        zs = zs.long()

        return zs

    def sample(self, xs, num_frames, num_samples=10):
        """
        Sample for wake phase, we sample num_samples z for each x
       :param xs: [batch_size, max_num_frame, tvs_dim]
       :param num_frames: [batch_size]
       :return: unmasked zs: [batch_size, num_samples, max_num_frame]
       """
        batch_size, max_num_frame, tvs_dim = xs.shape
        device = next(self.lstm_cell.parameters()).device
        # Auto Regression mode
        if self.auto_regression:
            xs_expanded = xs[:, None, ...].expand(batch_size, num_samples, max_num_frame, tvs_dim)\
                .reshape(-1, max_num_frame, tvs_dim)
            lstm_inputs = torch.zeros((batch_size * num_samples, self.num_targets + tvs_dim), device=device)
            h = torch.zeros((batch_size * num_samples, self.lstm_hidden_dim), device=device)
            c = torch.zeros((batch_size * num_samples, self.lstm_hidden_dim), device=device)
            lstm_inputs[:, -tvs_dim:] = xs_expanded[:, 0, :]  # [batch_size * num_samples, num_targets + tvs_dim]

            logits = torch.zeros((batch_size * num_samples, max_num_frame, self.num_targets), device=device)
            zs = torch.zeros((batch_size * num_samples, max_num_frame), device=device)
            for i in range(max_num_frame):
                h, c = self.lstm_cell(lstm_inputs, (h, c))
                logit = self.linear(h)  # [batch_size * num_samples, num_targets]
                logits[:, i, :] = logit
                categorical_dist = torch.distributions.Categorical(logits=logit)
                z = categorical_dist.sample()  # [batch_size * num_samples]
                zs[:, i] = z
                if i != max_num_frame - 1:
                    lstm_inputs = torch.cat(
                        [
                            F.one_hot(z, num_classes=self.num_targets).float(),  # [batch_size * num_samples, num_targets]
                            xs_expanded[:, i + 1, :]  # [batch_size, tvs_dim]
                        ], dim=-1
                    )
            zs = zs.long()
        # Dynamic RNN mode
        else:
            logits = self.forward(xs, num_frames)
            logits = logits[:, None, ...].expand(batch_size, num_samples, -1, self.num_targets)\
                .reshape(batch_size * num_samples, -1, self.num_targets)
            categorical_dist = torch.distributions.Categorical(logits=logits)
            zs = categorical_dist.sample().to(device)  # [batch_size * num_samples, max_num_frame]

        return logits, zs

    def get_log_prob(self, x, z, num_frames):
        """
        Training for sleep phase, the inputs are num_samples z sampled from prior distribution
        :param x: [num_samples, max_num_frame, tvs_dim]
        :param z: [num_samples, max_num_frame]
        :param num_frames: [num_samples]
        :return:
        """
        device = next(self.lstm_cell.parameters()).device
        num_samples, max_num_frame, tvs_dim = x.shape

        # Auto Regression mode
        if self.auto_regression:
            lstm_input = torch.zeros((num_samples, self.num_targets + tvs_dim), device=device)
            h = torch.zeros((num_samples, self.lstm_hidden_dim), device=device)
            c = torch.zeros((num_samples, self.lstm_hidden_dim), device=device)
            lstm_input[:, -tvs_dim:] = x[:, 0, :]

            ce_loss = torch.zeros((num_samples, max_num_frame), device=device)
            entropy_loss = torch.zeros((num_samples, max_num_frame), device=device)
            for i in range(max_num_frame):
                # Teacher Forcing
                if i != 0:
                    lstm_input = torch.cat(
                        [
                            F.one_hot(z[..., i-1], num_classes=self.num_targets).float(),
                            x[:, i, :]
                        ], dim=-1
                    )
                h, c = self.lstm_cell(lstm_input, (h, c))
                logits = self.classifier(h)
                categorical_dist = torch.distributions.Categorical(logits=logits)  # p(z|x)
                ce_loss[:, i] = -categorical_dist.log_prob(z[:, i])
                entropy_loss[:, i] = categorical_dist.entropy()
            mask = get_sequence_mask(num_frames).to(device)  # [num_samples, max_num_frame]
            ce_loss *= mask
            entropy_loss *= mask
            loss = ce_loss - self.entropy_lambda * entropy_loss
            loss = loss.mean(dim=-1)
        # Dynamic RNN mode
        else:
            logits = self.forward(x, num_frames)  # [num_samples, max_num_frame, num_targets]
            logits = logits.view(-1, self.num_targets)  # [num_samples * max_num_frame, num_targets]
            categorical_dist = torch.distributions.Categorical(logits=logits)
            z = z.view(-1)  # [num_samples * max_num_frame]
            # ce_loss = -categorical_dist.log_prob(z).view(num_samples, max_num_frame)  # [num_samples, max_num_frame]
            ce_loss = F.cross_entropy(logits.view(-1, self.num_targets), z.view(-1), reduction='none').view(
                num_samples, max_num_frame
            )
            mask = get_sequence_mask(num_frames).to(device)
            non_target_probs = F.softmax(logits)[:, 0].view(num_samples, max_num_frame) * mask
            ce_loss *= mask  # [num_samples, max_num_frame]
            entropy_loss = (categorical_dist.entropy()).view(num_samples, max_num_frame) * mask
            loss = ce_loss - self.entropy_lambda * entropy_loss - self.non_target_lambda * non_target_probs
            loss = loss.mean(dim=-1).to(device)  # [num_samples]

        return loss, ce_loss.mean().item(), entropy_loss.mean().item(), non_target_probs.mean().item()


class GenerativeModelPriorDistribution(nn.Module):

    def __init__(self, lstm_input_dim, lstm_hidden_dim, num_layers, num_targets):
        super().__init__()

        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_targets = num_targets
        self.num_layers = num_layers

        self.lstm_cell = nn.LSTM(
            input_size=self.lstm_input_dim, hidden_size=self.lstm_hidden_dim, num_layers=num_layers,
            bidirectional=False, batch_first=True
        )

        # self.lstm_cell = nn.LSTMCell(
        #     input_size=self.lstm_input_dim, hidden_size=self.lstm_hidden_dim)

        self.linear = nn.Linear(self.lstm_hidden_dim, num_targets)

        self._max_num_frame = 1000

    def sample(self, num_samples):
        """
        sample from the prior distribution p_theta(z)
        :param: num_samples
        :return: z: [num_samples, max_num_frames]
        """
        device = next(self.lstm_cell.parameters()).device
        # tvs data
        num_frames = torch.from_numpy(np.random.randn(num_samples, ) * self._max_num_frame).long().clamp(100, 800)

        # m01
        # half_samples = int(num_samples / 2)
        # num_frames = torch.from_numpy(np.concatenate(
        #     [np.random.normal(140, 18, half_samples), np.random.normal(210, 28, num_samples - half_samples)]
        # )).long().clamp(80, 300)

        max_num_frame = num_frames.max()

        lstm_input = torch.zeros((num_samples, self.lstm_input_dim), device=device)[:, None, :]
        h = torch.zeros((self.num_layers, num_samples, self.lstm_hidden_dim), device=device)
        c = torch.zeros((self.num_layers, num_samples, self.lstm_hidden_dim), device=device)

        targets = list()
        for i in range(max_num_frame):
            outputs, (h, c) = self.lstm_cell(lstm_input, (h, c))
            logits = self.linear(outputs)  # [num_samples, 1, num_targets]
            categorical_dist = torch.distributions.Categorical(logits=logits)
            targets.append(categorical_dist.sample())  # [num_samples, 1]

            lstm_input = F.one_hot(targets[-1], num_classes=self.num_targets).float()

        z = torch.hstack(targets).to(device)  # [num_samples, max_num_frame]

        return z, num_frames

    def log_prob(self, z, num_frames):
        """
        Get the log probability of "z in p_theta(z)"
        Used in wake phase to train prior distribution model
        :param z: tensor [num_samples, max_num_frame]
        :param num_frames: tensor [num_samples]
        :return: tensor: [num_samples]
        """
        device = next(self.lstm_cell.parameters()).device
        num_samples, max_num_frame = z.shape

        lstm_input = torch.zeros((num_samples, self.lstm_input_dim), device=device)[:, None, :]
        h = torch.zeros((self.num_layers, num_samples, self.lstm_hidden_dim), device=device)
        c = torch.zeros((self.num_layers, num_samples, self.lstm_hidden_dim), device=device)

        logits = torch.zeros((num_samples, max_num_frame, self.num_targets), device=device)
        for i in range(max_num_frame):
            outputs, (h, c) = self.lstm_cell(lstm_input, (h, c))
            logit = self.linear(outputs)  # [num_samples, 1, num_targets]
            logits[:, i] = logit[:, 0, :]
            # Teacher Forcing
            lstm_input = F.one_hot(z[:, i], num_classes=self.num_targets).float()[:, None, :]

        # log prob fashion
        # categorical_dist = torch.distributions.Categorical(logits=logits)
        # loss = -categorical_dist.log_prob(z)  # [num_samples, max_num_frame]

        # cross entropy fashion
        loss = F.cross_entropy(logits.view(-1, self.num_targets),
                               z.view(-1),
                               reduction='none').view(num_samples, max_num_frame)  # [num_samples, max_num_frame]

        mask = get_sequence_mask(num_frames).to(device)  # [num_samples, max_num_frame]
        loss *= mask
        loss = loss.mean(dim=-1)  # [num_samples]

        return loss

    # def log_prob(self, z, num_frames):
    #     """
    #     Get the log probability of "z in p_theta(z)"
    #     Used in wake phase to train prior distribution model
    #     :param z: tensor [num_samples, max_num_frame]
    #     :param num_frames: tensor [num_samples]
    #     :return: tensor: [num_samples]
    #     """
    #     device = next(self.lstm_cell.parameters()).device
    #     num_samples, max_num_frame = z.shape
    #
    #     lstm_input = torch.zeros((num_samples, self.lstm_input_dim), device=device)
    #     h = torch.zeros((num_samples, self.lstm_hidden_dim), device=device)
    #     c = torch.zeros((num_samples, self.lstm_hidden_dim), device=device)
    #
    #     logits = torch.zeros((num_samples, max_num_frame, self.num_targets), device=device)
    #     for i in range(max_num_frame):
    #         h, c = self.lstm_cell(lstm_input, (h, c))
    #         logit = self.linear(h)  # [num_samples, num_targets]
    #         logits[:, i] = logit
    #         # Teacher Forcing
    #         lstm_input = F.one_hot(z[:, i], num_classes=self.num_targets).float()

        # cross entropy fashion
        # loss = F.cross_entropy(logits.view(-1, self.num_targets),
        #                        z.view(-1),
        #                        reduction='none').view(num_samples, max_num_frame)  # [num_samples, max_num_frame]
        #
        # mask = get_sequence_mask(num_frames).to(device)  # [num_samples, max_num_frame]
        # loss *= mask
        # loss = loss.mean(dim=-1)  # [num_samples]
        #
        # return loss


class GenerativeModelConditionDistribution(nn.Module):

    def __init__(self, num_targets, tvs_dim, use_stds, num_processes=32, mode='linear', diff_lambda=0.1, diff_n=1):
        """
        :param num_targets: int
        :param tvs_dim: int
        :param num_processes: int
        """
        super().__init__()

        self.num_targets = num_targets
        self.num_processes = num_processes
        self.use_stds = use_stds
        self.target_means = nn.Parameter(torch.randn(num_targets, tvs_dim).clamp(-1, 1))
        self.target_stds = nn.Parameter(torch.randn(num_targets, tvs_dim).clamp(-1, 1)) if use_stds else None
        self.mode = mode
        self.diff_lambda = diff_lambda
        self.diff_n = diff_n

    @compute_running_time
    def sample_for_sleep(self, z, num_frames):
        # [num_samples, max_num_frames, tvs_dim]
        result = sample_for_sleep(z, num_frames, self.target_means, self.target_stds if self.use_stds else None,
                                  mode=self.mode)

        return result

    @compute_running_time
    def grad_compute(self, xs_expanded, zs, num_frames):
        """
        Compute grad for parameters
        :param xs_expanded: tensor [num_samples, max_num_frame, tvs_dim]
        :param zs: tensor [num_samples, max_num_frame]
        :param num_frames: [num_samples]
        """

        grad_means = list()
        grad_stds = list()
        samples = list()
        loss_batch = 0.
        reconstruction_loss_batch = 0.
        diff_loss_batch = 0.

        # Single process
        if self.num_processes == 0:
            for x_expanded, z, num_frame in zip(xs_expanded, zs, num_frames):
                grad_mean, grad_std, loss = self.grad_compute_for_a_sample(x_expanded[:num_frame], z[:num_frame])
                grad_means.append(grad_mean)
                grad_stds.append(grad_std)
                loss_batch += loss
        else:
            # Multiprocess
            # grads = list()
            # pool = multiprocessing.Pool(self.num_processes)
            # pool_apply_async = pool.apply_async
            #
            # for x_expanded, z, num_frame in zip(xs_expanded, zs, num_frames):
            #     x_expanded_ = x_expanded[:num_frame]
            #     z_ = z[:num_frame]
            #     target_means = self.target_means.detach().numpy()
            #     target_stds = self.target_stds.detach().numpy() if self.use_stds else None
            #     grads.append(pool_apply_async(grad_compute_for_a_sample,
            #                                   (target_means, target_stds, x_expanded_, z_)))
            # pool.close()
            # pool.join()
            # grads_means_stds_losses_samples = [grad_mean_std_loss_sample.get() for grad_mean_std_loss_sample in grads]

            # Ray multiprocess
            target_means = self.target_means.detach().numpy()
            target_stds = self.target_stds.detach().numpy() if self.use_stds else None
            grads = [grad_compute_for_a_sample.remote(
                target_means, target_stds, x_expanded[:num_frame], z[:num_frame],
                self.mode, self.diff_lambda, self.diff_n)
                for x_expanded, z, num_frame in zip(xs_expanded, zs, num_frames)]
            grads_means_stds_losses_samples = ray.get(grads)

            for grad_mean, grad_std, loss, reconstruction_loss, diff_loss, sample in grads_means_stds_losses_samples:
                grad_means.append(grad_mean)
                grad_stds.append(grad_std)
                samples.append(sample)
                loss_batch += loss
                reconstruction_loss_batch += reconstruction_loss
                diff_loss_batch += diff_loss

            del grads, grads_means_stds_losses_samples

        grad_mean = torch.stack(grad_means).mean(dim=0)
        grad_std = torch.stack(grad_stds).mean(dim=0) if self.use_stds else None
        loss = loss_batch / num_frames.shape[0]
        reconstruction_loss = reconstruction_loss_batch / num_frames.shape[0]
        diff_loss = diff_loss_batch / num_frames.shape[0]

        return grad_mean, grad_std, loss, reconstruction_loss, diff_loss, samples

    def grad_assignment(self, xs_expanded, zs, num_frames):
        """
        Manually assign grad for parameters
        :param xs_expanded: np.ndarray [num_samples, max_num_frame, tvs_dim]
        :param zs: np.ndarray [num_samples, max_num_frame]
        :param num_frames: [num_samples]
        """
        grad_mean_std_loss_sample, wake_interpolate_time = self.grad_compute(xs_expanded, zs, num_frames)
        grad_mean, grad_std, loss, reconstruction_loss, diff_loss, samples = grad_mean_std_loss_sample
        self.target_means.grad = grad_mean
        if self.use_stds:
            self.target_stds.grad = grad_std

        return loss, reconstruction_loss, diff_loss, wake_interpolate_time, samples


class UnitTest(object):

    def __init__(self):

        self._inference_model = InferenceModel(128, num_targets=64)
        self._generative_prior_model = GenerativeModelPriorDistribution(lstm_input_dim=64, lstm_hidden_dim=128,
                                                                        num_layers=2,
                                                                        num_targets=64)
        self._generative_condition_model = GenerativeModelConditionDistribution(num_targets=64, tvs_dim=15,
                                                                                num_processes=48,
                                                                                use_stds=True,
                                                                                mode='pchip')

    def inference_test(self, x, z, num_frames):
        """

        :param x: tensor: [batch_size, max_num_frame, tvs_dim]
        :param z: tensor: [batch_size, max_num_frame]
        :param num_frames: tensor: [batch_size]
        :return:
        """
        inference_loss = self._inference_model.get_log_prob_from_posterior_dist(
            self._inference_model.get_posterior_dist(x, num_frames), z[:, None, ...]
        ).mean()
        inference_loss.backward()

        print('Inference Test Complete ! Inference Loss: {}'.format(inference_loss.item()))

    def generative_prior_dist_test(self, z, num_frames):
        prior_loss = self._generative_prior_model.log_prob(z, num_frames).mean()
        prior_loss.backward()

        print('Generative Prior Test Complete ! Prior Loss: {}'.format(prior_loss.item()))

    def generative_condition_dist_test(self, x, z, num_frames):

        condition_loss, reconstruction_loss, diff_loss, _ = self._generative_condition_model.log_prob(x, z, num_frames)
        condition_loss = condition_loss.mean()
        condition_loss.backward()

        print('Generative Condition Test Complete ! Condition Loss: {}'.format(condition_loss.item()))

    def sample_for_sleep(self, z, num_frames):
        with torch.no_grad():
            sample, sample_time = self._generative_condition_model.sample_for_sleep(z, num_frames)

        print('Sample for sleep x complete ! Shape check: {}. Time: {}'.format(sample.shape, sample_time))

    def grad_assignment(self, xs, zs, num_frames):
        loss, sample_time, _ = self._generative_condition_model.grad_assignment(xs, zs, num_frames)
        print(self._generative_condition_model.target_means.grad)
        print(self._generative_condition_model.target_stds.grad)

        print('Grad assignment for wake phase complete ! Loss: {}, Time: {}'.format(loss, sample_time))

    def sample_prior_z(self):
        sample, num_frames = self._generative_prior_model.sample(10)

        print('Sample for prior z complete ! Shape check: {}, Num_frames: {}'.format(sample.shape, num_frames))

    def sample_posterior_z(self, x, num_frames):
        sample = self._inference_model.sample(x, num_frames, 10)

        print('Sample for posterior z complete ! Shape check: {}'.format(sample.shape))


if __name__ == '__main__':

    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
    # Unit
    torch.multiprocessing.set_start_method('spawn')
    num_samples = 100
    num_frames_ = torch.randint(300, 1000, (num_samples,))  # (num_samples)
    max_num_frame_ = num_frames_.max()
    obs = torch.randn(num_samples, max_num_frame_, 15)
    latent = torch.randint(0, 64, (num_samples, max_num_frame_))  # (num_samples, num_frames)

    ut = UnitTest()
    ray.init(num_cpus=64, num_gpus=0)
    # ut.inference_test(obs, latent, num_frames)
    ut.generative_prior_dist_test(latent, num_frames_)
    # ut.generative_condition_dist_test(obs, latent, num_frames_)
    # ut.sample_for_sleep(latent, num_frames_)
    # ut.grad_assignment(obs.numpy(), latent.numpy(), num_frames_)
    # ut.sample_prior_z()
    # ut.sample_posterior_z(obs, num_frames_)
    ray.shutdown()

    # target_means = np.random.randn(64, 15)
    # target_stds = np.random.randn(64, 15)
    # for i in range(num_samples):
    #     sample = interpolate_for_a_z_sleep_pchip(latent[i], num_frames_[i], target_means, target_stds)
    #     print(sample.shape)


