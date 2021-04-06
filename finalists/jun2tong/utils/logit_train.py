import numpy as np
import torch

from torch.utils import data
from .common import check_ext_mem, check_ram_usage
from .wrapper import CustomTensorDataset


def train_net(optimizer, scheduler, model, criterion, data_loader, t, distill_coef,
              train_ep, device="cpu"):

    cur_ep = 0
    cur_train_t = t
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
            all_out = model(x_mb)
            if len(all_out) > 1:
                logits = all_out[-1]
            else:
                logits = all_out[0]
            pred = torch.cat(all_out, dim=1)
            loss = criterion["cls"](pred, y_mb)
            dist_loss = 0
            if cur_train_t > 0:
                p_logits = torch.cat(all_out[:-1], dim=1)
                dist_loss = criterion["dist"](torch.log_softmax(p_logits, dim=1),
                                              torch.softmax(p_logits_mb, dim=1).to(device))
                all_loss = loss + distill_coef*dist_loss
            else:
                all_loss = loss

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            pred_label = torch.argmax(logits, dim=1)
            correct_cnt += (pred_label.detach() == y_mb).sum().cpu()

            acc = correct_cnt.item() / ((it + 1) * y_mb.size(0))

            if it % 100 == 0:
                print(f'==>>> it: {it}, dist loss: {dist_loss: .6f}, cls loss: {loss: .6f}, running train acc: {acc: .3f}')
            it += 1
        cur_ep += 1
        scheduler.step()

    return stats


def preprocess_imgs(img_batch, scale=True, norm=True, channel_first=True):

    if scale:
        # convert to float in [0, 1]
        img_batch = img_batch / 255

    if norm:
        # normalize
        img_batch[:, :, :, 0] = ((img_batch[:, :, :, 0] - 0.485) / 0.229)
        img_batch[:, :, :, 1] = ((img_batch[:, :, :, 1] - 0.456) / 0.224)
        img_batch[:, :, :, 2] = ((img_batch[:, :, :, 2] - 0.406) / 0.225)

    if channel_first:
        # Swap channel dimension to fit the caffe format (c, w, h)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))

    return img_batch


def maybe_cuda(what, device):

    what = what.to(device)
    return what


def test_multitask(model, test_set, mb_size, preproc=None, use_cuda="cpu", multi_heads=True, verbose=True):

    model.eval()

    acc_x_task = []
    stats = {'accs': [], 'acc': []}
    preds = []

    for (x, y), t in test_set:

        if preproc:
            x = preproc(x)

        ds = CustomTensorDataset(x, torch.from_numpy(y).type(torch.LongTensor))
        # dataset = LogitAppend(ds, torch.from_numpy(y).type(torch.LongTensor), t, remap=False)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=mb_size, shuffle=False, num_workers=4)

        model = maybe_cuda(model, device=use_cuda)

        correct_cnt, ave_loss = 0, 0
        boundary = [0] + [tt for tt in range(10, 50, 5)]
        with torch.no_grad():

            for x_mb, y_mb in dataloader:
                x_mb = x_mb.to(use_cuda)
                all_out = model(x_mb)
                task_idx = len(all_out) - 1
                pred = torch.zeros((all_out[-1].size(0), 50))

                start_idx = boundary[t]
                end_idx = boundary[t+1] if t < 8 else boundary[t]+5
                if t <= task_idx:
                    if not multi_heads:
                        pred[:, :] = all_out[0]
                    else:
                        pred[:, start_idx:end_idx] = all_out[t]

                _, pred_label = torch.max(pred, 1)
                correct_cnt += (pred_label == y_mb).sum().cpu()
                preds += list(pred_label.data.cpu().numpy())

            acc = correct_cnt.item() / len(ds)

        if verbose:
            print(f'TEST Acc. Task {t}==>>> acc: {acc:.3f}')

        acc_x_task.append(acc)
        stats['accs'].append(acc)

    stats['acc'].append(np.mean(acc_x_task))

    return stats, preds


def select_mean_idx(model, x_set, y_label, num_most_likely_sample, largest, device):
    with torch.no_grad():
        model.eval()
        class_labels = np.unique(y_label)
        MuCs = [torch.zeros(2048) for _ in range(50)]
        ext_x = [[] for _ in range(len(class_labels))]
        ext_y = [[] for _ in range(len(class_labels))]
        for class_idx, class_label in enumerate(class_labels):
            idx = y_label == class_label
            subset_x = x_set[idx]
            subset_y = y_label[idx]
            ds = CustomTensorDataset(subset_x, subset_y)
            dataloader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
            current_class_feature = []
            for x_mb, _ in dataloader:
                feas = model.features(x_mb.to(device))
                current_class_feature.append(feas)
            current_class_feature = torch.cat(current_class_feature, dim=0)

            Mu = torch.mean(current_class_feature, dim=0)
            dists = torch.sqrt(torch.pow(current_class_feature.sub(Mu), 2).sum(dim=1))
            if dists.size(0) >= num_most_likely_sample:
                top_num = torch.topk(dists, num_most_likely_sample, largest=largest)
                MuCs[class_label] = torch.mean(current_class_feature[top_num[1], :], dim=0)
                ext_x[class_idx] = subset_x[top_num[1], :].cpu()
                ext_y[class_idx] = subset_y[top_num[1]].cpu()
            else:
                MuCs[class_label] = torch.mean(current_class_feature, dim=0)
                ext_x[class_idx] = subset_x[:, :].cpu()
                ext_y[class_idx] = subset_y.cpu()
        ext_data = [torch.cat(ext_x, dim=0), torch.cat(ext_y, dim=0)]
    return MuCs, ext_data


def test_MLE_multitask(model, test_set, mb_size, Mus, preproc=None, use_cuda="cpu", multi_heads=True, verbose=True):
    model.eval()

    acc_x_task = []
    stats = {'accs': [], 'acc': []}
    preds = []

    for (x, y), t in test_set:

        if preproc:
            x = preproc(x)
        ds = CustomTensorDataset(x, torch.from_numpy(y).type(torch.LongTensor))
        dataloader = torch.utils.data.DataLoader(ds, batch_size=mb_size, shuffle=False, num_workers=4)

        model = maybe_cuda(model, device=use_cuda)

        correct_cnt, ave_loss = 0, 0
        boundary = [0] + [tt for tt in range(10, 50, 5)]
        with torch.no_grad():

            for x_mb, y_mb in dataloader:
                x_mb = x_mb.to(use_cuda)

                all_out = model(x_mb)
                feas = model.features(x_mb)
                dists = torch.zeros(feas.shape[0], len(Mus))
                for i in range(len(Mus)):
                    dists[:, i] = torch.sqrt(torch.pow(feas.sub(Mus[i].to(use_cuda)), 2).sum(dim=1))
                task_idx = len(all_out) - 1
                mle_pred = torch.ones((all_out[-1].size(0), len(Mus))) * 1e10
                pred = torch.zeros((all_out[-1].size(0), 50))

                start_idx = boundary[t]
                end_idx = boundary[t+1] if t < 8 else boundary[t]+5
                if t <= task_idx:
                    if not multi_heads:
                        pred[:, :] = all_out[0]
                        mle_pred[:, :] = dists
                    else:
                        pred[:, start_idx:end_idx] = all_out[t]
                        mle_pred[:, start_idx:end_idx] = dists[:, start_idx:end_idx]

                mle_pred = 1. / np.array(mle_pred.cpu())
                mle_pred = np.divide(mle_pred, np.tile(np.sum(mle_pred, axis=1), [mle_pred.shape[1], 1]).transpose())

                pred = torch.softmax(pred, dim=1).cpu().numpy()
                pred = np.divide(pred, np.tile(np.sum(pred, axis=1), [pred.shape[1], 1]).transpose())

                if not multi_heads:
                    task_idx_ni = int(len(Mus)/50)
                    pred_ni = np.tile(pred, [1, task_idx_ni])
                    pred_label = (np.argmax(mle_pred + pred_ni, axis=1).astype(int))
                else:
                    pred_label = np.argmax(mle_pred[:,:pred.shape[1]] + pred, axis=1).astype(int)

                correct_cnt += sum(pred_label == np.array(y_mb.data.cpu()))
                preds += list(pred_label)

            acc = correct_cnt / len(ds)

        if verbose:
            print(f'TEST Acc. Task {t}==>>> acc: {acc:.3f}')

        acc_x_task.append(acc)
        stats['accs'].append(acc)

    stats['acc'].append(np.mean(acc_x_task))

    return stats, preds
