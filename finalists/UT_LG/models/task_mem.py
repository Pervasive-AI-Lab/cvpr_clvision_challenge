import torch
from utils.train_test import train_net, test_multitask, preprocess_imgs, train_net_aug, test_multitask_aug
import numpy as np
import time
from utils.setup_elements import setup_classifier, setup_opt, setup_dataset, setup_aug
import copy
from utils.buffer_torch import Buffer_batch_level

class task_mem(object):

    def __init__(self, args, loss, use_cuda=True):
        self.args = args
        self.loss = loss
        self.buffer = Buffer_batch_level(args.scenario, args.replay_examples)
        self.model = setup_classifier(cls_name=args.cls, n_classes=args.n_classes, args=args)
        self.optimizer = setup_opt(self.args.optimizer, self.model, self.args)
        self.dataset, self.full_valdidset, self.num_tasks = setup_dataset(self.args.scenario, preload=args.preload_data)
        self.use_cuda = use_cuda
        self.train_aug, self.test_aug = setup_aug(self.args.aug_type)
        self.replay_used = args.replay_used
        self.review_size = args.review_size
        self.test_bz = 128

    def train_model(self, tune=True, resize=False):
        print('train start')
        start = time.time()
        # vars to update over time
        valid_acc = []
        ext_mem_sz = []
        ram_usage = []
        heads = []
        num_tasks = self.dataset.nbatch[self.dataset.scenario]
        # loop over the training incremental batches (x, y, t)
        for i, train_batch in enumerate(self.dataset):
            train_x_raw, train_y_raw, t = train_batch
            train_x_raw = train_x_raw.astype(np.uint8)
            train_y_raw = train_y_raw.astype(np.uint8)
            del train_batch
            if self.args.verbose:
                print("----------- batch {0} -------------".format(i))
            for replay_epoch in range(self.args.replay_epochs):
                if self.args.verbose:
                    print("replay epoch {} for batch {}".format(replay_epoch, i))
                if i > 0:
                    mem_imgs, mem_labels = self.buffer.get_mem(self.replay_used)
                    train_x = np.concatenate((train_x_raw, mem_imgs))
                    train_y = np.concatenate((train_y_raw, mem_labels))
                else:
                    train_x = train_x_raw
                    train_y = train_y_raw

                if self.args.verbose:
                    print("x shape: {0}, y shape: {1}"
                          .format(train_x.shape, train_y.shape))

                # train the classifier on the current batch/task
                if self.args.aug:
                    _, _, stats = train_net_aug(
                        self.optimizer, self.model, self.loss, self.args.batch_size, train_x, train_y, i,
                        self.args.epochs, self.train_aug, use_cuda=self.use_cuda, verbose=self.args.verbose
                    )
                else:
                    _, _, stats = train_net(
                        self.optimizer, self.model, self.loss, self.args.batch_size, train_x, train_y, i,
                        self.args.epochs, preproc=preprocess_imgs, use_cuda=self.use_cuda, resize=resize
                    )
                # collect statistics
                ext_mem_sz += stats['disk']
                ram_usage += stats['ram']
                if self.args.verbose and self.args.scenario != 'nic':
                    print(ram_usage)


            if self.args.scenario == "multi-task-nc":
                heads.append(copy.deepcopy(self.model.fc))

            # adding eventual replay patterns to the current batch
            self.buffer.update_buffer(train_x_raw, train_y_raw, i)

            # test on the validation set
            if self.args.aug:
                stats, _ = test_multitask_aug(
                    self.model, self.full_valdidset, self.test_bz, self.test_aug,
                    verbose=self.args.verbose,
                    use_cuda=self.use_cuda
                )

            else:
                stats, _ = test_multitask(
                    self.model, self.full_valdidset, self.args.batch_size,
                    preproc=preprocess_imgs, multi_heads=heads, verbose=self.args.verbose,
                    use_cuda=self.use_cuda,
                    resize=resize
                )
            if self.args.verbose:
                print("------------------------------------------")
                print("Avg. acc:", stats['acc'])
                print("------------------------------------------")
            valid_acc += stats['acc']

            # final review
            if i == num_tasks - 1:
                for replay_epoch in range(self.args.review_epoch):
                    for g in self.optimizer.param_groups:
                        g['lr'] = self.args.review_lr_factor * self.args.lr
                    train_x, train_y = self.buffer.get_mem(self.review_size)
                    if self.args.aug:
                        _, _, stats = train_net_aug(
                            self.optimizer, self.model, self.loss, self.args.batch_size, train_x, train_y, i,
                            self.args.epochs, self.train_aug, use_cuda=self.use_cuda
                        )
                        stats, _ = test_multitask_aug(
                            self.model, self.full_valdidset, self.test_bz, self.test_aug,
                            verbose=self.args.verbose,
                            use_cuda=self.use_cuda
                        )
                    else:
                        _, _, stats = train_net(
                            self.optimizer, self.model, self.loss, self.args.batch_size, train_x, train_y, i,
                            self.args.epochs, preproc=preprocess_imgs, use_cuda=self.use_cuda, resize=resize
                        )
                        stats, _ = test_multitask(
                            self.model, self.full_valdidset, self.args.batch_size,
                            preproc=preprocess_imgs, multi_heads=heads, verbose=self.args.verbose,
                            use_cuda=self.use_cuda,
                            resize=resize
                        )
                    if self.args.verbose:
                        print("------------------------------------------")
                        print("Review Avg. acc:", stats['acc'])
                        print("------------------------------------------")


        final_val = stats['acc']
        # clear mem
        del self.buffer, self.full_valdidset
        full_testset = self.dataset.get_full_test_set()

        # generating metadata.txt: with all the data used for the CLScore
        elapsed = (time.time() - start) / 60
        print("Training Time: {}m".format(elapsed))
        if not tune:
            # test_preds.txt: with a list of labels separated by "\n"
            print("Final inference on test set...")
            if self.args.aug:
                stats, preds = test_multitask_aug(
                    self.model, full_testset, self.test_bz, self.test_aug,
                    verbose=self.args.verbose,
                    use_cuda=self.use_cuda
                )

            else:
                stats, preds = test_multitask(
                    self.model, full_testset, self.args.batch_size,
                    preproc=preprocess_imgs, multi_heads=heads, verbose=self.args.verbose,
                    use_cuda=self.use_cuda,
                    resize=resize
                )

            return valid_acc, elapsed, ram_usage, ext_mem_sz, preds
        else:
            return final_val[0], np.average(valid_acc), elapsed, np.average(ram_usage), np.max(ram_usage)
