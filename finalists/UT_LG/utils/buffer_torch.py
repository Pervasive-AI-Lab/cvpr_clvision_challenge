import numpy as np
import torch
from utils.setup_elements import cal_memsize
from collections import Counter

class Buffer():
    def __init__(self, buffer_size, img_size, eps_mem_batch):
        self.episodic_images = torch.full([buffer_size]+img_size, 0.0,dtype=torch.float32)
        self.episodic_labels = torch.full([buffer_size], 0.0,dtype=torch.long)
        self.episodic_task = Counter()
        self.episodic_mem_size = buffer_size
        self.eps_mem_batch = eps_mem_batch
        self.examples_seen_so_far = 0
        print('buffer has {} slots'.format(buffer_size))

    def update_buffer(self, batch_x, batch_y, task_id):
        for er_x, er_y in zip(batch_x, batch_y):
            if self.episodic_mem_size > self.examples_seen_so_far:
                self.episodic_images[self.examples_seen_so_far] = er_x
                self.episodic_labels[self.examples_seen_so_far] = er_y
                self.episodic_task[self.examples_seen_so_far] = task_id
            else:
                j = np.random.randint(0, self.examples_seen_so_far)
                if j < self.episodic_mem_size:
                    self.episodic_images[j] = er_x
                    self.episodic_labels[j] = er_y
                    self.episodic_task[j] = task_id

            self.examples_seen_so_far += 1

    def get_mem(self, model, current_x=None, current_y=None, exclude=None):
        mem_filled_so_far = self.examples_seen_so_far if (
                self.examples_seen_so_far < self.episodic_mem_size) else self.episodic_mem_size
        if mem_filled_so_far < self.eps_mem_batch:
            er_mem_indices = np.arange(mem_filled_so_far)
            np.random.shuffle(er_mem_indices)
            final_x, final_y = self.episodic_images[er_mem_indices], self.episodic_labels[er_mem_indices]
        else:
            er_mem_indices = np.random.choice(mem_filled_so_far, self.eps_mem_batch, replace=False)
            np.random.shuffle(er_mem_indices)
            final_x, final_y = self.episodic_images[er_mem_indices], self.episodic_labels[er_mem_indices]
        return final_x, final_y

class Buffer_batch_level():
    def __init__(self, scenario, replay_size, img_size=(128, 128, 3)):
        episodic_mem_size = cal_memsize(scenario, replay_size)
        self.episodic_images = np.zeros((episodic_mem_size, ) + img_size).astype(np.uint8)
        self.episodic_labels = np.zeros((episodic_mem_size, )).astype(np.uint8)
        self.episodic_mem_size = episodic_mem_size
        self.replay_size = replay_size
        self.cur_mem = 0
        self.scenario = scenario
        #print('buffer has {} slots'.format(episodic_mem_size))

    def update_buffer(self, x, y, task_id):
        idxs_cur = np.random.choice(
            x.shape[0], self.replay_size, replace=False
        )
        if task_id == 0:
            if self.scenario == 'nic':
                self.episodic_images[self.cur_mem:self.cur_mem+y.shape[0]] = x
                self.episodic_labels[self.cur_mem:self.cur_mem+y.shape[0]] = y
                self.cur_mem += y.shape[0]
            else:
                size = x[idxs_cur].shape[0]
                self.episodic_images[self.cur_mem:self.cur_mem+size] = x[idxs_cur]
                self.episodic_labels[self.cur_mem:self.cur_mem + size] = y[idxs_cur]
                self.cur_mem += size
        else:
            size = x[idxs_cur].shape[0]
            self.episodic_images[self.cur_mem:self.cur_mem + size] = x[idxs_cur]
            self.episodic_labels[self.cur_mem:self.cur_mem + size] = y[idxs_cur]
            self.cur_mem += size
        #print('current buffer filled {}'.format(self.cur_mem))

    def get_mem(self, get_size):
        if self.cur_mem < get_size:
            idxs_use = np.arange(self.cur_mem)
        else:
            idxs_use = np.random.choice(
                self.cur_mem, get_size, replace=False
            )
        return self.episodic_images[idxs_use], self.episodic_labels[idxs_use]

class Buffer_naive_add():
    def __init__(self, replay_examples, eps_mem_batch, args):
        self.replay_examples = replay_examples
        self.eps_mem_batch = eps_mem_batch
        self.args = args
        self.mem = None

    def update_buffer(self, x, y, task_id):
        idxs_cur = np.random.choice(
            x.shape[0], self.replay_examples, replace=False
        )
        if task_id == 0:
            if self.args.scenario == 'nic':
                self.mem = [x, y]
            else:
                self.mem = [x[idxs_cur], y[idxs_cur]]
        else:
            self.mem = [
                np.concatenate((x[idxs_cur], self.mem[0])),
                np.concatenate((y[idxs_cur], self.mem[1]))]

    def get_mem(self):
        idxs_use = np.random.choice(
            self.mem[1].shape[0], self.eps_mem_batch, replace=False
        )
        return self.mem[0][idxs_use], self.mem[1][idxs_use]

