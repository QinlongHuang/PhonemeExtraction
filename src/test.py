# -*- coding: utf-8 -*-
# @File: PhonemeExtraction/test.py
# @Author: Qinlong Huang
# @Create Date: 2021/04/16 00:07
# @Contact: qinlonghuang@gmail.com
# @Description:

from multiprocessing import Process, Queue
import time

import torch
import torch.nn as nn
import torch.optim as optim

print("PyTorch: {}".format(torch.__version__))


def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.uniform_()
        m.bias.data.uniform_()


def do_neural_stuff():
    # Initialize the network
    fcffnn = nn.Sequential(
        nn.Linear(1, 10),
        nn.Sigmoid(),
        nn.Linear(10, 10),
        nn.Sigmoid(),
        nn.Linear(10, 2)
    )
    fcffnn.apply(init_weights)

    # Run the network
    loss_f = nn.CrossEntropyLoss()
    opt = optim.Adam(fcffnn.parameters(), lr=3e-4, weight_decay=0.001)

    # Compute loss, backpropagate, adjust params along neg error gradient
    output = fcffnn(torch.tensor([0], dtype=torch.float))
    loss = loss_f(output.view(1, 2), torch.tensor([1], dtype=torch.long))
    fcffnn.zero_grad()
    loss.backward()
    opt.step()


# Run the network on the parent process
do_neural_stuff()  # PyTorch works as expected when this line is commented out


def parallelModelRunnerProc(proc_nb, response_queue):
    print("Forked Process {}".format(proc_nb))
    do_neural_stuff()

    # This is important -- without the sleep(), the threads exit
    # without ever crashing due to a native error in this simplified example
    time.sleep(5)
    r_queue.put(True)


# 1) Run the network on 5 child processes
# 2) Wait for background activity (CUDA initialization -- even though it is not asked for)
# 3) Observe and report error
r_queue = Queue()
servers = []
nb_responses = 0
child_ct = 5
for proc_nb in range(child_ct):
    server = Process(target=parallelModelRunnerProc, args=(proc_nb, r_queue))
    server.start()
    servers.append(server)

while nb_responses < child_ct:
    nb_dead = sum([not x.is_alive() for x in servers])
    if not r_queue.empty():
        r_queue.get()
        nb_responses += 1

    if nb_dead > nb_responses:
        raise Exception("Something died unexpectedly")

