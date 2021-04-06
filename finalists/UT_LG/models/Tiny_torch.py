import torch
from utils.buffer_torch import Buffer_naive_add
from utils.common import pad_data, shuffle_in_unison, check_ext_mem, check_ram_usage
from utils.train_test import maybe_cuda
from models.task_mem import task_mem
import time
import numpy as np
from utils.train_test import train_net, test_multitask, preprocess_imgs
import copy
from torch.utils import data

class Tiny(task_mem):

    def __init__(self, args, loss, use_cuda=True):
        super().__init__(args, loss, use_cuda)
        self.seperate = args.seperate
        self.buffer = Buffer_naive_add(args.replay_examples, args.eps_mem_batch, args)

    def _train_step(self, optimizer, model, criterion, batch_size, x, y, t,
              epochs, preproc=None, use_cuda=True, resize=False):
        print(t)
        stats = {"ram": [], "disk": []}

        if preproc:
            x = preproc(x)

        (train_x, train_y), it_x_ep = pad_data(
            [x, y], batch_size
        )

        shuffle_in_unison(
            [train_x, train_y], 0, in_place=True
        )

        model = maybe_cuda(model, use_cuda=use_cuda)
        acc = None
        ave_loss = 0
        mem_batch_size = self.buffer.eps_mem_batch

        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.LongTensor)
        training_set = data.TensorDataset(train_x, train_y)
        params = {'batch_size': batch_size,
                  'shuffle': True,
                  'num_workers': 6}
        dataloader = data.DataLoader(training_set, **params)
        for ep in range(epochs):

            stats['disk'].append(check_ext_mem("cl_ext_mem"))
            stats['ram'].append(check_ram_usage())

            model.active_perc_list = []
            model.train()

            print("training ep: ", ep)
            correct_cnt, ave_loss = 0, 0

            for it, batch in enumerate(dataloader):
                x_mb, y_mb = batch
                optimizer.zero_grad()

                if t > 0:
                    mem_images, mem_labels = self.buffer.get_mem()
                    mem_images = torch.from_numpy(mem_images).type(torch.FloatTensor)
                    mem_labels = torch.from_numpy(mem_labels).type(torch.LongTensor)
                    er_train_x_batch = torch.cat((mem_images, x_mb), dim=0)
                    er_train_y_batch = torch.cat((mem_labels, y_mb), dim=0)
                else:
                    er_train_x_batch = x_mb
                    er_train_y_batch = y_mb

                if self.seperate:
                    x_mb = maybe_cuda(x_mb, use_cuda=use_cuda)
                    y_mb = maybe_cuda(y_mb, use_cuda=use_cuda)
                    logits = model(x_mb)
                    loss = criterion(logits, y_mb)
                    loss.backward()
                    optimizer.step()
                    ave_loss += (loss.item())
                    _, pred_label = torch.max(logits, 1)
                    correct_cnt += (pred_label == y_mb).sum()
                    if t > 0:
                        mem_images = maybe_cuda(mem_images, use_cuda=use_cuda)
                        mem_labels = maybe_cuda(mem_labels, use_cuda=use_cuda)
                        mem_logits = model(mem_images)
                        loss_mem = criterion(mem_logits, mem_labels)
                        loss_mem.backward()
                        optimizer.step()
                        ave_loss += (loss_mem.item())
                        _, pred_label_mem = torch.max(mem_logits, 1)
                        correct_cnt += (pred_label_mem == mem_labels).sum()
                        acc = correct_cnt.item() / \
                              ((it + 1) * (batch_size + mem_batch_size))
                        ave_loss /= ((it + 1) * (batch_size + mem_batch_size))
                    else:
                        acc = correct_cnt.item() / \
                              ((it + 1) * batch_size)
                        ave_loss /= ((it + 1) * batch_size)

                else:
                    er_train_x_batch = maybe_cuda(er_train_x_batch, use_cuda=use_cuda)
                    er_train_y_batch = maybe_cuda(er_train_y_batch, use_cuda=use_cuda)

                    logits = model(er_train_x_batch)

                    _, pred_label = torch.max(logits, 1)
                    correct_cnt += (pred_label == er_train_y_batch).sum()
                    loss = criterion(logits, er_train_y_batch)

                    ave_loss += loss.item()

                    loss.backward()
                    optimizer.step()

                    acc = correct_cnt.item() / \
                          ((it + 1) * er_train_y_batch.size(0))
                    ave_loss /= ((it + 1) * er_train_y_batch.size(0))

                if it % 80 == 0:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(it, ave_loss, acc)
                    )
        return ave_loss, acc, stats

    #################################################################################
    #### External APIs of the class. These will be called/ exposed externally #######
    #################################################################################
    def train_model(self, tune=True, resize=False):
        start = time.time()
        # vars to update over time
        valid_acc = []
        ext_mem_sz = []
        ram_usage = []
        heads = []
        # loop over the training incremental batches (x, y, t)
        for i, train_batch in enumerate(self.dataset):
            train_x, train_y, _ = train_batch

            print("----------- batch {0} -------------".format(i))
            print("x shape: {0}, y shape: {1}"
                  .format(train_x.shape, train_y.shape))

            # train the classifier on the current batch/task
            _, _, stats = self._train_step(
                optimizer=self.optimizer,
                model=self.model,
                criterion=self.loss,
                batch_size=self.args.batch_size,
                x=train_x,
                y=train_y,
                t=i,
                epochs=self.args.epochs,
                preproc=preprocess_imgs,
                use_cuda=self.use_cuda,
                resize=resize)

            ext_mem_sz += stats['disk']
            ram_usage += stats['ram']
            #print(ram_usage)
            if self.args.scenario == "multi-task-nc":
                heads.append(copy.deepcopy(self.model.fc))
            # test on the validation set
            stats, _ = test_multitask(
                self.model, self.full_valdidset, mb_size=64,
                preproc=preprocess_imgs, multi_heads=heads, verbose=True, use_cuda=self.use_cuda, resize=resize
            )
            train_x = preprocess_imgs(train_x)
            self.buffer.update_buffer(train_x, train_y, i)
            print(self.buffer.mem[0].shape)
            del train_x, train_y, train_batch
            valid_acc += stats['acc']
            print("------------------------------------------")
            print("Avg. acc: {}".format(stats['acc']))
            print("------------------------------------------")

        # final review

        if self.args.review_epoch == -1:
            mem_size = self.buffer.mem[1].shape[0]
            review_size = 15000
            if mem_size % review_size == 0:
                num_it = mem_size // review_size
            else:
                num_it = mem_size // review_size + 1
            for it in range(num_it):
                start = it * review_size
                end = (it + 1) * review_size
                train_x = self.buffer.mem[0][start:end]
                train_y = self.buffer.mem[1][start:end]
                _, _, stats = train_net(
                    self.optimizer, self.model, self.loss, self.args.batch_size, train_x, train_y, 0,
                    1, preproc=None, use_cuda=self.use_cuda, resize=resize
                )
                stats, _ = test_multitask(
                    self.model, self.full_valdidset, mb_size=64,
                    preproc=preprocess_imgs, multi_heads=heads, verbose=True, use_cuda=self.use_cuda, resize=resize
                )
                print("------------------------------------------")
                print("review Avg. acc: {}".format(stats['acc']))
                print("------------------------------------------")

        else:
            for replay_epoch in range(self.args.review_epoch):
                idxs_use = np.random.choice(
                    self.buffer.mem[1].shape[0], self.args.review_size, replace=False
                )
                train_x = self.buffer.mem[0][idxs_use]
                train_y = self.buffer.mem[1][idxs_use]
                _, _, stats = train_net(
                    self.optimizer, self.model, self.loss, self.args.batch_size, train_x, train_y, 0,
                    1, preproc=None, use_cuda=self.use_cuda, resize=resize
                )
                stats, _ = test_multitask(
                    self.model, self.full_valdidset, mb_size=64,
                    preproc=preprocess_imgs, multi_heads=heads, verbose=True, use_cuda=self.use_cuda, resize=resize
                )
                print("------------------------------------------")
                print("review Avg. acc: {}".format(stats['acc']))
                print("------------------------------------------")


        final_val = stats['acc']
        del self.full_valdidset, self.buffer
        elapsed = (time.time() - start) / 60
        print("Training Time: {}m".format(elapsed))
        full_testset = self.dataset.get_full_test_set()
        if not tune:
            # test_preds.txt: with a list of labels separated by "\n"
            print("Final inference on test set...")
            stats, preds = test_multitask(
                self.model, full_testset, mb_size=32, preproc=preprocess_imgs,
                multi_heads=heads, verbose=True, use_cuda=self.use_cuda, resize=resize
            )
            return valid_acc, elapsed, ram_usage, ext_mem_sz, preds
        else:
            return final_val[0], np.average(valid_acc), elapsed, np.average(ram_usage), np.max(ram_usage)