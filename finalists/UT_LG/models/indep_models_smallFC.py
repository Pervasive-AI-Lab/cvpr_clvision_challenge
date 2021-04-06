import torch
from utils.train_test import train_net, preprocess_imgs, test_multi_models, test_multi_models_small
import numpy as np
import time
from core50.dataset import CORE50
from utils.setup_elements import setup_classifier, setup_opt, setup_dataset


class indep_models_smallFC(object):

    def __init__(self, args, loss, use_cuda=True):
        self.args = args
        self.loss = loss
        self.dataset, self.full_valdidset, self.num_tasks = setup_dataset(self.args.scenario)
        self.models_list = {}
        self.opt_list = {}
        self.match_list = {}
        self.use_cuda = use_cuda


    #
    def train_model(self, tune=True, resize=False):
        start = time.time()
        # vars to update over time
        valid_acc = []
        ext_mem_sz = []
        ram_usage = []
        # loop over the training incremental batches (x, y, t)
        for i, train_batch in enumerate(self.dataset):
            train_x, train_y, t = train_batch
            # preprocess data
            unique = np.unique(train_y)
            num_class = unique.shape[0]
            new_class = np.arange(num_class)
            dictionary = dict(zip(unique, new_class))
            self.match_list[t] = dictionary

            train_y_new = np.copy(train_y)
            for k, v in dictionary.items():
                train_y_new[train_y == k] = v
            del train_y

            self.models_list[t] = setup_classifier(cls_name=self.args.cls, n_classes=num_class, args=self.args)
            self.opt_list[t] = setup_opt(self.args.optimizer, self.models_list[t], self.args)

            print("----------- batch {0} -------------".format(i))
            print("x shape: {0}, y shape: {1}"
                  .format(train_x.shape, train_y_new.shape))
            print("Task Label: ", t)

            model = self.models_list[t]
            _, _, stats = train_net(
                self.opt_list[t], model, self.loss, self.args.batch_size, train_x, train_y_new, t,
                self.args.epochs, preproc=preprocess_imgs, use_cuda=self.use_cuda, resize=resize
            )

            # collect statistics
            ext_mem_sz += stats['disk']
            ram_usage += stats['ram']

            # test on the validation set
            stats, _ = test_multi_models_small(
                self.models_list, self.match_list, self.full_valdidset, mb_size=64,
                preproc=preprocess_imgs, use_cuda=self.use_cuda, verbose=True
            )
            model.cpu()

            valid_acc += stats['acc']
            print("------------------------------------------")
            print("Avg. acc: {}".format(stats['acc']))
            print("------------------------------------------")
            del train_x, train_y_new, t, train_batch

        final_val = stats['acc']
        # generating metadata.txt: with all the data used for the CLScore
        elapsed = (time.time() - start) / 60
        print("Training Time: {}m".format(elapsed))
        del self.full_valdidset
        full_testset = self.dataset.get_full_test_set()

        if not tune:
            # test_preds.txt: with a list of labels separated by "\n"
            print("Final inference on test set...")

            stats, preds = test_multi_models_small(
                self.models_list, self.match_list, full_testset, mb_size=64,
                preproc=preprocess_imgs, use_cuda=self.use_cuda, verbose=True
            )
            return valid_acc, elapsed, ram_usage, ext_mem_sz, preds
        else:
            return final_val[0], np.average(valid_acc), elapsed, np.average(ram_usage), np.max(ram_usage)
