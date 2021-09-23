# -*- coding: utf-8 -*-
# @File: PhonemeExtraction/manually_grad.py
# @Author: Qinlong Huang
# @Create Date: 2021/04/15 16:21
# @Contact: qinlonghuang@gmail.com
# @Description:

import torch
from torch import nn, optim
from torch.autograd import grad
from tqdm import tqdm
from multiprocessing.pool import Pool
from src.utils import setup_seed
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


def process_a_sample(x: torch.tensor, a, b, c, y):

    y_pred = a * torch.pow(x, 2) + b * x + c
    loss = ((y - y_pred) ** 2).sum()
    grad_a, grad_b, grad_c = grad(loss, (a, b, c))

    return grad_a, grad_b, grad_c


def process_a_batch(xs, a, b, c, ys):

    y_pred = a * torch.pow(xs, 2) + b * xs + c
    loss = ((ys - y_pred) ** 2).sum()
    grad_a, grad_b, grad_c = grad(loss, (a, b, c))

    return grad_a, grad_b, grad_c


def grad_compute_for_a_sample(a, b, c, x, y):

    model = SimpleToyModel(a, b, c)
    y_pred = model(x)
    loss = ((y - y_pred) ** 2).sum()
    grad_a, grad_b, grad_c = grad(loss, (model.param_a, model.param_b, model.param_c))

    del model, y_pred, loss

    return grad_a, grad_b, grad_c


class SimpleToyModel(nn.Module):

    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        super(SimpleToyModel, self).__init__()

        self.param_a = nn.Parameter(torch.tensor(a))
        self.param_b = nn.Parameter(torch.tensor(b))
        self.param_c = nn.Parameter(torch.tensor(c))

    def forward(self, x):
        y_pred = self.param_a * x ** 2 + self.param_b * x + self.param_c
        return y_pred


class ToyModel(nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()

        self.param_a = nn.Parameter(torch.randn(1))
        self.param_b = nn.Parameter(torch.randn(1))
        self.param_c = nn.Parameter(torch.randn(1))

    def forward(self, xs):
        y_pred = self.param_a * xs ** 2 + self.param_b * xs + self.param_c
        return y_pred

    def grad_compute(self, xs, ys):

        pool = Pool(processes=32)
        grads = list()

        grad_as = list()
        grad_bs = list()
        grad_cs = list()

        for x, y in zip(xs, ys):
            grads.append(pool.apply_async(process_a_sample, (x, self.param_a, self.param_b, self.param_c, y)))
        pool.close()
        pool.join()
        grads_abc = [grad_.get() for grad_ in grads]

        for grad_a, grad_b, grad_c in grads_abc:
            grad_as.append(grad_a)
            grad_bs.append(grad_b)
            grad_cs.append(grad_c)

        grad_a = torch.stack(grad_as).sum(dim=0)
        grad_b = torch.stack(grad_bs).sum(dim=0)
        grad_c = torch.stack(grad_cs).sum(dim=0)

        # grad_a, grad_b, grad_c = process_a_batch(xs, self.param_a, self.param_b, self.param_c, ys)

        return grad_a, grad_b, grad_c

    def grad_compute_submodel(self, xs, ys):

        grad_as = list()
        grad_bs = list()
        grad_cs = list()

        pool = Pool(processes=32)
        grads = list()

        for x, y in zip(xs, ys):
            x_ = x.clone()
            y_ = y.clone()
            a = self.param_a.detach().cpu().clone().numpy()
            b = self.param_b.detach().cpu().clone().numpy()
            c = self.param_c.detach().cpu().clone().numpy()
            grads.append(pool.apply_async(grad_compute_for_a_sample, (a, b, c, x, y)))

        pool.close()
        pool.join()
        grads_abc = [grad_.get() for grad_ in grads]

        for grad_a, grad_b, grad_c in grads_abc:
            grad_as.append(grad_a)
            grad_bs.append(grad_b)
            grad_cs.append(grad_c)

        del grads_abc, grads

        grad_a = torch.stack(grad_as).sum(dim=0)
        grad_b = torch.stack(grad_bs).sum(dim=0)
        grad_c = torch.stack(grad_cs).sum(dim=0)

        return grad_a, grad_b, grad_c

    def grad_assignment(self, xs, ys):
        grad_a, grad_b, grad_c = self.grad_compute(xs, ys)
        self.param_a.grad, self.param_b.grad, self.param_c.grad = grad_a, grad_b, grad_c


if __name__ == '__main__':

    setup_seed(666)
    model = ToyModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    xs_ = torch.randn(2000, 1)
    print(xs_)
    print(model.param_a, model.param_b, model.param_c)
    ys_ = torch.stack([5 * x ** 2 + 6 * x + 1 for x in xs_])

    for i in range(10):
        optimizer.zero_grad()
        # 1.
        # y_pred = model(xs_)
        # loss = ((ys_ - y_pred) ** 2).sum()

        # 2.
        # y_preds = list()
        # for x in xs_:
        #     y_pred = model(x[None])
        #     y_preds.append(y_pred)
        # y_pred = torch.cat(y_preds, dim=0)
        # loss = ((ys_ - y_pred) ** 2).sum()

        # 3.
        # loss = 0.
        # for x, y in zip(xs_, ys_):
        #     y_pred = model(x[None])
        #     loss += (y[None] - y_pred) ** 2

        # loss.backward()

        # 4.
        # model.grad_assignment(xs_, ys_)

        optimizer.step()

        # 1和4的结果一样，2和3的结果一样，两种结果之间有精度问题
        print(model.param_a.grad, model.param_b.grad, model.param_c.grad)

    # print(model.param_a, model.param_b, model.param_c)






