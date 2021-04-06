import numpy as np
import torch

from torch.utils import data
from .common import check_ext_mem, check_ram_usage
from .wrapper import CustomTensorDataset


def train_net(optimizer, scheduler, model, criterion, data_loader, reg_coef,
              train_ep, device="cpu"):

    cur_ep = 0
    stats = {"ram": [], "disk": []}

    for ep in range(train_ep):

        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())
        model.train()
        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0
        it = 0

        for x_mb, y_mb, p_logits_mb in data_loader:
            x_mb = x_mb.to(device)
            y_mb = y_mb.to(device)
            all_out = model(x_mb)[0]

            loss = criterion["cls"](all_out, y_mb)
            dist_loss = 0
            if not isinstance(p_logits_mb, list):
                p_logits_mb = p_logits_mb.to(device)
                dist_loss = criterion["dist"](torch.log_softmax(all_out, dim=1),
                                              torch.softmax(p_logits_mb, dim=1))
                # dist_loss = criterion["dist"](all_out, p_logits_mb)
                all_loss = loss + reg_coef*dist_loss
            else:
                all_loss = loss
            ave_loss += all_loss.item()

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            pred_label = torch.argmax(all_out, dim=1)
            correct_cnt += (pred_label.detach() == y_mb).sum().cpu()

            acc = correct_cnt.item() / ((it + 1) * y_mb.size(0))
            ave_loss /= ((it + 1) * y_mb.size(0))

            if it % 100 == 0:
                # print(f'==>>> it: {it}, avg. loss: {ave_loss: .6f}, running train acc: {acc: .3f}')
                print(f'==>>> it: {it}, dist. loss: {dist_loss: .6f}, cls loss: {loss: .6f}, running train acc: {acc: .3f}')
            it += 1
        cur_ep += 1
        scheduler.step()

    return stats


def nn_mean_classify(model, test_set, mb_size, Mus, preproc=None, use_cuda="cpu", verbose=True):
    model.eval()

    acc_x_task = []
    stats = {'accs': [], 'acc': []}
    preds = []

    for (x, y), t in test_set:

        if preproc:
            x = preproc(x)
        ds = CustomTensorDataset(x, torch.from_numpy(y).type(torch.LongTensor))
        dataloader = torch.utils.data.DataLoader(ds, batch_size=mb_size, shuffle=False, num_workers=4)

        model = model.to(use_cuda)

        correct_cnt, ave_loss = 0, 0
        with torch.no_grad():

            for x_mb, y_mb in dataloader:
                x_mb = x_mb.to(use_cuda)

                feas = model.features(x_mb)
                dists = torch.zeros(feas.shape[0], len(Mus))
                for i in range(len(Mus)):
                    dists[:, i] = torch.sqrt(torch.pow(feas.sub(Mus[i].to(use_cuda)), 2).sum(dim=1))

                # pred_label = torch.argmin(dists, dim=1)
                pred_label = torch.topk(dists, 1, dim=1, largest=False)[1]

                correct_cnt += sum(pred_label.view(-1).numpy() == np.array(y_mb))
                preds += list(pred_label)

            acc = correct_cnt / len(ds)

        if verbose:
            print(f'TEST Acc. Task {t}==>>> acc: {acc:.3f}')

        acc_x_task.append(acc)
        stats['accs'].append(acc)

    stats['acc'].append(np.mean(acc_x_task))

    return stats, preds
