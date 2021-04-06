import torch
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from utils.common import pad_data, shuffle_in_unison, check_ext_mem, check_ram_usage
from utils.train_test import maybe_cuda


class EWC(object):

    def __init__(self, model, args, loss, optmizer, num_tasks=None, lr=0.001, use_cuda=True):
        self.model = maybe_cuda(model, use_cuda=use_cuda)
        self.lambda_ = args.lambda_
        self.num_tasks = num_tasks
        self.loss = loss
        self.optimizer = optmizer(self.model.parameters(), lr)
        self.use_cuda = use_cuda

    ####################################################################################
    #### Internal APIs of the class. These should not be called/ exposed externally ####
    ####################################################################################
    def _compute_ewc_loss(self):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                losses.append((getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name)) * (
                        param - getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))) ** 2).sum())
            return (self.lambda_ / 2) * sum(losses)
        except AttributeError:
            return 0

    def _update_ewc_params(self, x, y, batch_size):
        current_data = TensorDataset(
            maybe_cuda(x, use_cuda=self.use_cuda),
            maybe_cuda(y, use_cuda=self.use_cuda))

        dl = DataLoader(current_data, batch_size, shuffle=True)
        log_liklihood_list = []
        for i, (x, y) in enumerate(dl):
            if i <= self.num_tasks:
                tmp = F.log_softmax(self.model(x), dim=1)
                log_liklihood_list.append(tmp[:, y])
            else:
                break
        grad_log_liklihood = autograd.grad(torch.cat(log_liklihood_list).mean(), self.model.parameters())

        for (param_name, param), grad in zip(self.model.named_parameters(), grad_log_liklihood):
            self.model.register_buffer(param_name.replace('.', '__') + '_estimated_fisher',
                                       grad.data.clone() ** 2)

        for param_name, param in self.model.named_parameters():
            self.model.register_buffer(param_name.replace('.', '__') + '_estimated_mean', param.data.clone())

    #################################################################################
    #### External APIs of the class. These will be called/ exposed externally #######
    #################################################################################
    def train_model(self, x, y, t, batch_size, epochs, args, preproc=None, **kwargs):
        ewc_cap_factor = args.ewc_cap_factor
        stats = {"ram": [], "disk": []}

        if preproc:
            x = preproc(x)

        (train_x, train_y), it_x_ep = pad_data(
            [x, y], batch_size
        )

        shuffle_in_unison(
            [train_x, train_y], 0, in_place=True
        )

        acc = None
        ave_loss = 0

        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.LongTensor)

        for ep in range(epochs):

            stats['disk'].append(check_ext_mem("cl_ext_mem"))
            stats['ram'].append(check_ram_usage())

            self.model.active_perc_list = []
            self.model.train()

            print("training ep: ", ep)
            correct_cnt, ave_loss = 0, 0
            cap_factor = maybe_cuda(torch.tensor(0.0), use_cuda=self.use_cuda)
            for it in range(it_x_ep):

                start = it * batch_size
                end = (it + 1) * batch_size

                self.optimizer.zero_grad()

                x_mb = maybe_cuda(train_x[start:end], use_cuda=self.use_cuda)
                y_mb = maybe_cuda(train_y[start:end], use_cuda=self.use_cuda)
                logits = self.model(x_mb)

                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == y_mb).sum()
                loss_normal = self.loss(logits, y_mb)
                ewc_loss = maybe_cuda(torch.as_tensor(self._compute_ewc_loss(), dtype=torch.float32),
                                      use_cuda=self.use_cuda).atan()
                cap_factor = torch.max(cap_factor, ewc_cap_factor * loss_normal)

                loss = loss_normal + cap_factor.data * ewc_loss
                ave_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                acc = correct_cnt.item() / \
                      ((it + 1) * y_mb.size(0))
                ave_loss /= ((it + 1) * y_mb.size(0))

                if it % 100 == 0:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(it, ave_loss, acc)
                    )

        self._update_ewc_params(train_x, train_y, batch_size)
        return ave_loss, acc, stats, train_x, train_y
