# -*- coding: utf-8 -*-
# @File: PhonemeExtraction/losses.py
# @Author: Qinlong Huang
# @Create Date: 2021/03/31 14:34
# @Contact: qinlonghuang@gmail.com
# @Description:

import torch
import math
import utils
import models
import time
import warnings
import torch.nn.functional as F

warnings.simplefilter('ignore')


def get_sleep_loss(generative_prior_model: models.GenerativeModelPriorDistribution,
                   generative_condition_model: models.GenerativeModelConditionDistribution,
                   inference_model: models.InferenceModel,
                   num_samples=1):

    # z: detached tensor [num_samples, max_num_frame]
    # x: detached tensor [num_samples, max_num_frame, obs_embedding_dim]
    # num_frames: [num_samples]
    # 从先验分布p_theta(z)中采样得到z
    with torch.no_grad():
        z, num_frames = generative_prior_model.sample(num_samples)
        # 根据z得到分布p_phi(x|z)，并采样得到x
        x, sleep_interpolate_time = generative_condition_model.sample_for_sleep(z, num_frames)
        x = x.to(z.device)

    # 根据x得到后验分布r_psi(z|x)，并从中采样得到z'，求z和z'的loss
    # 具体的，就是求z属于分布r_psi(z|x)的log_prob
    loss, ce_loss, entropy_loss, non_target_prob = inference_model.get_log_prob(x, z, num_frames)  # [num_samples]
    sleep_loss = loss.mean()

    return sleep_loss, sleep_interpolate_time, ce_loss, entropy_loss, non_target_prob


def get_wake_loss(generative_prior_model: models.GenerativeModelPriorDistribution,
                  generative_condition_model: models.GenerativeModelConditionDistribution,
                  inference_model: models.InferenceModel,
                  x: torch.tensor, num_frames: torch.tensor, num_samples=10, alpha=1., plot_samples=5):
    """

    :param generative_prior_model:
    :param generative_condition_model:
    :param inference_model:
    :param x: tensor: [batch_size, max_num_frame, tvs_dim]
    :param num_frames: tensor: [batch_size]
    :param num_samples: number of samples per x_i
    :param alpha: float
    :param plot_samples: number of plots
    :return: scalar: condition_loss, tensor with grad: prior_loss, float: wake_interpolate_time,
    np.ndarray: logits, z_preds, x_preds
    """
    # 根据x得到分布r_psi(z|x)，并从中采样num_samples个得到zs
    # logits: [batch_size * num_samples, max_num_frames, num_targets]
    # zs: [batch_size * num_samples, max_num_frames]
    with torch.no_grad():
        batch_size, max_num_frame, tvs_dim = x.shape
        logits, zs = inference_model.sample(x, num_frames, num_samples)  # [batch_size * num_samples, max_num_frames]

    # log p(x, z) = log p(z) + log p(x|z)
    num_frames_expanded = num_frames[:, None].expand(batch_size, num_samples).reshape(-1)

    # 求prior loss的时候mask就行了，其他时候都不用mask，condition因为是一条一条数据计算的，所以也不用mask
    log_prior = generative_prior_model.log_prob(zs, num_frames_expanded)  # [batch_size * num_samples]
    log_prior = log_prior.view(batch_size, num_samples)

    xs_expanded = x[:, None, ...].expand(batch_size, num_samples, max_num_frame, tvs_dim) \
        .reshape(batch_size * num_samples, max_num_frame, tvs_dim).cpu().numpy()
    zs = zs.cpu().numpy()
    num_frames_expanded = num_frames_expanded.numpy()

    condition_loss, reconstruction_loss, diff_loss, wake_interpolate_time, reconstructions = \
        generative_condition_model.grad_assignment(xs_expanded, zs, num_frames_expanded)

    # log_joint = log_prior + alpha * log_condition  # [batch_size, num_samples]

    # TODO: 是从中采样一个还是对前面采样的多个z求均值或相加？
    # Qi_dist = torch.distributions.Categorical(logits=log_joint)
    # sampled_id = Qi_dist.sample()  # [batch_size]
    # sampled_id_log_prob = Qi_dist.log_prob(sampled_id)
    # log_joint = torch.gather(log_joint, 1, sampled_id[:, None])[:, 0]  # [batch_size]
    # wake_loss = torch.mean(log_joint - sampled_id_log_prob.detach())

    prior_loss = torch.mean(log_prior)

    z_logits = logits.cpu()
    # list of np.ndarray, means normalized probability
    probs = [F.softmax(z_logits[i * num_samples, :num_frames[i]], dim=-1).numpy() for i in range(plot_samples)]

    z_preds = [zs[i * num_samples, :num_frames[i]] for i in range(plot_samples)]  # list of np.ndarray
    x_preds = [reconstructions[i * num_samples][:num_frames[i]] for i in range(plot_samples)]  # list of np.ndarray

    return condition_loss, reconstruction_loss, diff_loss, prior_loss, wake_interpolate_time, probs, z_preds, x_preds


# def get_mws_loss(
#         generative_model: models.GenerativeModel, inference_model: models.InferenceModel, memory: torch.Tensor,
#         obs: torch.Tensor, obs_id: torch.Tensor, num_particles: int
# ):
#     """
#
#     :param generative_model: models.GenerativeModel object
#     :param inference_model: models.InferenceModel object
#     :param memory: tensor: [num_data, memory_size, num_frames], we maintain a memory for every single x_i
#     :param obs: tensor: [batch_size, num_frames, obs_embedding_dim]
#     :param obs_id: tensor: [batch_size]
#     :param num_particles: number of samples per x_i
#     :return:
#     """
#     memory_size = memory.shape[1]
#
#     # Propose latents from inference network
#     # 1.对给定的x_i，构建分布r_psi(z|x_i)
#     latent_dist = inference_model.get_latent_dist(obs)
#
#     # 2.从r_psi(z|x_i)分布中采样num_particles个z
#     # [batch_size, num_particles, num_frames]
#     latent = inference_model.sample_from_latent_dist(latent_dist, num_particles).detach()
#     batch_size = latent.shape[0]
#
#     # Evaluate log p of proposed latents
#     # 3.计算2中采样得到的num_particles个z与p_theta(z)分布的log probability——log_prior
#     # 具体的，也是使用lstm运行num_arcs步，在每一步的过程中也是和2一样，构建分布，然后计算z中相应时间步和这个分布的log_prob，并累加
#     # 4.使用2中的z构建一个Bernoulli分布p_phi(x|z)，并计算x_i与这个分布的log_prob——log_likelihood。
#     log_prior, log_likelihood = generative_model.get_log_probss(latent, obs)
#     log_p = log_prior + log_likelihood  # [batch_size, num_particles]
#
#     # Select latents from memory
#     # 5.每个x_i都有自己的memory，这一步就是根据id取出memory中的z，即伪码中的psi
#     # TODO: 是否应该把memory构造成一个list, 构造成一个tensor的话会需要取所有语音中最长的一个作为frames,
#     #  而且就还需要一个memory去维护每个example的num_frame
#     memory_latent = memory[obs_id]  # [batch_size, memory_size, num_frames]
#
#     # Evaluate log p of memory latents
#     # 6.同理计算psi的log_prior和log_likelihood
#     memory_log_prior, memory_log_likelihood = generative_model.get_log_probss(memory_latent, obs)
#     memory_log_p = memory_log_prior + memory_log_likelihood  # [batch_size, memory_size]
#     memory_log_likelihood = None  # don't need this anymore
#
#     # 7.将新采样得到的z和memory中的psi合并
#     # Merge proposed latents and memory latents
#     # [batch_size, memory_size + num_particles, num_frames]
#     memory_and_latent = torch.cat([memory_latent, latent])
#     # [batch_size, memory_size + num_particles]
#     memory_and_latent_log_p = torch.cat([memory_log_p, log_p])
#     memory_and_latent_log_prior = torch.cat([memory_log_prior, log_prior])
#
#     # Compute new map
#     # 8.统计这一个batch中需要更新的memory的比例，即z的log_p大于psi的log_p的比例
#     # 这一个batch中，每个x_i采样R个z_i，memory中本身有M个zeta_i
#     # 然后分别去R个中的最大值和M个中的最大值，并对他俩进行比较
#     # 最后统计新采样的log_p大于memory中的log_p的比例，即new_map
#     new_map = (log_p.max(dim=1).values > memory_log_p.max(dim=1).values).float().mean()
#
#     # Sort log_ps, replace non-unique values with -inf, choose the top k remaining
#     # (valid as long as two distinct latents don't have the same logp)
#     # 9.对合并后的memory依据log_p进行排序，然后取最大的memory_size个，作为新的memory
#     sorted1, indices1 = memory_and_latent_log_p.sort(dim=-1)
#     is_same = sorted1[:, 1:] == sorted1[:, :-1]
#     novel_proportion = 1 - (is_same.float().sum() / num_particles / batch_size)
#     sorted1[1:].masked_fill_(is_same, float("-inf"))
#     sorted2, indices2 = sorted1.sort(dim=-1)
#     memory_log_p = sorted2[:, -memory_size:]  # [batch_size, memory_size]
#     indices = indices1.gather(1, indices2)[:, -memory_size:]  # [batch_size, memory_size]
#     memory_latent = torch.gather(
#         memory_and_latent,
#         1,
#         indices[:, :, None]
#         .expand(batch_size, memory_size, generative_model.num_frames)
#         .contiguous(),
#     )
#     memory_log_prior = torch.gather(memory_and_latent_log_prior, 1, indices)  # [batch_size, memory_size]
#     # Update memory
#     # [batch_size, memory_size, num_frames]
#     memory[obs_id] = memory_latent
#
#     # Compute losses
#     # memory_log_p.t(): [batch_size, memory_size]
#     # 10.对新的memory进行采样，采样的概率正比于各psi的log_p
#     # 同时计算采出来的样本属于分布Qi的log_prob
#     dist = torch.distributions.Categorical(logits=memory_log_p)  # Qi
#     sampled_memory_id = dist.sample()  # [batch_size]
#     sampled_memory_id_log_prob = dist.log_prob(sampled_memory_id)  # [batch_size]
#     sampled_memory_latent = torch.gather(  # 按上面sample得到的index取出z_Q
#         memory_latent,  # [batch_size, memory_size, num_frames]
#         1,
#         sampled_memory_id[:, None, None].expand(batch_size, 1, generative_model.num_frames).contiguous(),
#     )  # [batch_size, 1, num_frames]
#
#     # 根据刚才sample的z_Q，从memory中取出log_p，即log_p之前就算好了，现在只是把它拿出来
#     # log_q是刚才sample的z_Q属于分布r_psi(z|x_i)的log prob
#     # log_q就是拿来训练recognition model的loss
#     log_p = torch.gather(memory_log_p, 1, sampled_memory_id[:, None])[:, 0]  # [batch_size]
#     log_q = inference_model.get_log_prob_from_latent_dist(latent_dist, sampled_memory_latent)[0]  # [batch_size]
#
#     prior_loss = -torch.gather(memory_log_prior, 1, sampled_memory_id[:, None]).mean().detach()
#     theta_loss = -torch.mean(log_p - sampled_memory_id_log_prob.detach())  # 为什么要减去这个？
#     phi_loss = -torch.mean(log_q)
#
#     return (
#         theta_loss,
#         phi_loss,
#         prior_loss.item(),
#         novel_proportion.item(),  # 新采样num_particles个，和历史的memory_size个，他们的重复率，体现在log_p是否相等
#         new_map.item(),  # 在这个batch中，平均新采样的最好的(即log_p最大的)比memory中最好的，更好的比例
#     )


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    torch.multiprocessing.set_start_method('spawn')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    generative_prior_model_ = models.GenerativeModelPriorDistribution(64, 128, 64).cuda()
    generative_condition_model_ = models.GenerativeModelConditionDistribution(64, 15)
    inference_model_ = models.InferenceModel(64).cuda()
    optimizer_prior = torch.optim.Adam(generative_prior_model_.parameters())
    optimizer_condition = torch.optim.Adam(generative_condition_model_.parameters())
    optimizer_inference = torch.optim.Adam(inference_model_.parameters())
    batch_size = 48
    num_samples = 10
    num_frames_ = torch.randint(300, 1000, size=(batch_size, ))
    x_ = torch.randn(batch_size, num_frames_.max(), 15).cuda()
    for i in range(50):
        start_time = time.time()
        sleep_loss, sleep_interpolate_time_ = get_sleep_loss(generative_prior_model_, generative_condition_model_,
                                                             inference_model_, num_samples=batch_size * num_samples)
        optimizer_inference.zero_grad()
        sleep_loss.backward()
        optimizer_inference.step()
        sleep_time = time.time() - start_time

        condition_loss, prior_loss, wake_interpolate_time_ = get_wake_loss(generative_prior_model_,
                                                                           generative_condition_model_,
                                                                           inference_model_,
                                                                           x_, num_frames_, num_samples)
        optimizer_prior.zero_grad()
        optimizer_condition.zero_grad()
        prior_loss.backward()
        optimizer_prior.step()
        optimizer_condition.step()
        wake_time = time.time() - start_time - sleep_time
        print("Epoch {}\nsleep_time: {:.2f}s, wake_time: {:.2f}s\n"
              "sleep_interpolate_time: {:.2f}s, wake_interpolate_time: {:.2f}s\n"
              "sleep efficiency: {:.2f}, sleep interpolation efficiency: {:.2f}\n"
              "wake efficiency: {:.2f}, wake interpolation efficiency: {:.2f}\n"
              .format(i, sleep_time, wake_time,
                      sleep_interpolate_time_, wake_interpolate_time_,
                      sleep_time / (sleep_time + wake_time), sleep_interpolate_time_ / (sleep_time + wake_time),
                      wake_time / (sleep_time + wake_time), wake_interpolate_time_ / (sleep_time + wake_time)))




