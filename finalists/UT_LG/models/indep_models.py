from utils.train_test import train_net, preprocess_imgs, test_multi_models, train_net_aug_indep, test_multi_models_aug
import numpy as np
import time
from utils.setup_elements import setup_classifier, setup_opt, setup_dataset, setup_aug

class indep_models(object):

    def __init__(self, args, loss, use_cuda=True):
        self.args = args
        self.loss = loss
        self.dataset, self.full_valdidset, self.num_tasks = setup_dataset(self.args.scenario)
        self.models_list = {}
        self.opt_list = {}
        self.use_cuda = use_cuda
        self.train_aug, self.test_aug = setup_aug(self.args.aug_type)
        self.test_bz = 128

    def train_model(self, tune=True, resize=False):
        start = time.time()
        # vars to update over time
        valid_acc = []
        ext_mem_sz = []
        ram_usage = []
        # loop over the training incremental batches (x, y, t)
        for i, train_batch in enumerate(self.dataset):
            train_x, train_y, t = train_batch
            train_x = train_x.astype(np.uint8)
            train_y = train_y.astype(np.uint8)
            print("----------- batch {0} -------------".format(i))
            print("x shape: {0}, y shape: {1}"
                  .format(train_x.shape, train_y.shape))
            print("Task Label: ", t)

            self.models_list[t] = setup_classifier(cls_name=self.args.cls, n_classes=self.args.n_classes, args=self.args)
            self.opt_list[t] = setup_opt(self.args.optimizer, self.models_list[t], self.args)


            if self.args.aug:
                _, _, stats = train_net_aug_indep(
                    self.opt_list[t], self.models_list, self.loss, self.args.batch_size, train_x, train_y, t,
                    self.args.epochs, self.full_valdidset, self.args.batch_size, self.train_aug, self.test_aug,
                    use_cuda=self.use_cuda, verbose=self.args.verbose
                )

            else:
                model = self.models_list[t]
                _, _, stats = train_net(
                    self.opt_list[t], model, self.loss, self.args.batch_size, train_x, train_y, t,
                    self.args.epochs, preproc=preprocess_imgs, use_cuda=self.use_cuda, resize=resize
                )

            # collect statistics
            ext_mem_sz += stats['disk']
            ram_usage += stats['ram']
            print(ram_usage)
            # test on the validation set
            if self.args.aug:
                stats, _ = test_multi_models_aug(
                    self.models_list, self.full_valdidset, self.test_bz, self.test_aug,
                    verbose=self.args.verbose,
                    use_cuda=self.use_cuda
                )

            else:
                stats, _ = test_multi_models(
                    self.models_list, self.full_valdidset, self.args.batch_size,
                    preproc=preprocess_imgs, use_cuda=self.use_cuda, verbose=True
                )

            valid_acc += stats['acc']

            del train_x, train_y, t, train_batch
        final_val = stats['acc']

        del self.full_valdidset
        full_testset = self.dataset.get_full_test_set()
        # generating metadata.txt: with all the data used for the CLScore
        elapsed = (time.time() - start) / 60
        print("Training Time: {}m".format(elapsed))
        if not tune:
            # test_preds.txt: with a list of labels separated by "\n"
            print("Final inference on test set...")

            if self.args.aug:
                stats, preds = test_multi_models_aug(
                    self.models_list, full_testset, self.test_bz, self.test_aug,
                    verbose=self.args.verbose,
                    use_cuda=self.use_cuda
                )

            else:
                stats, preds = test_multi_models(
                    self.models_list, full_testset, mb_size=64,
                    preproc=preprocess_imgs, use_cuda=self.use_cuda, verbose=True
                )
            return valid_acc, elapsed, ram_usage, ext_mem_sz, preds
        else:
            return final_val[0], np.average(valid_acc), elapsed, np.average(ram_usage), np.max(ram_usage)
