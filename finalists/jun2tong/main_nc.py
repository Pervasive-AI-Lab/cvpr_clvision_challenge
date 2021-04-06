
import argparse
import os
import time
import torch.nn as nn
import torch
import numpy as np

from core50.dataset import CORE50
from utils.wrapper import LogitAppend, CustomTensorDataset
from torch.utils import data
from utils.logit_train import train_net, preprocess_imgs, test_MLE_multitask
from utils.logit_train import select_mean_idx

from models import resnet_128
from utils.common import create_code_snapshot
import gc


def main(args):

    # print args recap
    print(args, end="\n\n")

    # do not remove this line
    start = time.time()

    # Create the dataset object for example with the "ni, multi-task-nc, or nic
    # tracks" and assuming the core50 location in ./core50/data/
    dataset = CORE50(root='core50/data/', scenario=args.scenario, preload=args.preload_data)

    # Get the validation set
    print("Recovering validation set...")
    full_valdidset = dataset.get_full_valid_set()

    # model
    ft_num_target = args.n_classes
    if args.scenario == "multi-task-nc":
        ft_num_target = len(dataset.labs_for_task[0])
    if args.classifier == 'ResNet18':
        classifier = resnet_128.resnet18(pretrained=True, num_classes=ft_num_target, use_nl=args.use_mle).to(args.device)
    elif args.classifier == 'ResNet50':
        classifier = resnet_128.resnet50(pretrained=True, num_classes=ft_num_target, use_nl=args.use_mle).to(args.device)

    optim_config = {"momentum": 0.9, "weight_decay": 0.0001, "schedule": args.scheduler_step}

    criterion = {"cls": torch.nn.CrossEntropyLoss(),
                 "dist": torch.nn.KLDivLoss(reduction='batchmean')}

    # vars to update over time
    valid_acc = []
    ext_mem_sz = []
    ram_usage = []
    ext_mem = None
    MuCs = [[] for _ in range(args.n_classes)]
    # loop over the training incremental batches (x, y, t)
    for i, train_batch in enumerate(dataset):
        cur_train_x, cur_train_y, t = train_batch

        cur_train_x = preprocess_imgs(cur_train_x)
        cur_train_x = torch.from_numpy(cur_train_x)

        cur_train_y = torch.from_numpy(cur_train_y).type(torch.LongTensor)
        if i > 0:
            train_x = torch.cat((cur_train_x, ext_mem[0]), dim=0)
            train_y = torch.cat((cur_train_y, ext_mem[1]), dim=0)
        else:
            train_x = cur_train_x
            train_y = cur_train_y

        ds = CustomTensorDataset(train_x, train_y)
        train_dataset = LogitAppend(ds, train_y, t)

        print(f"----------- batch {i} -------------")
        print(f"x shape: {train_x.shape}, y shape: {train_y.shape}, unique y: {len(np.unique(train_y))}")
        print(f"Task Label: {t}")
        if i > 0:
            print("generating logits.")
            train_dataset.update_logits(classifier, args.device)
            if args.scenario == "multi-task-nc":
                in_size = classifier.last[t-1].in_features
                classifier.last.append(nn.Linear(in_size, len(dataset.labs_for_task[t])))
                classifier.to(args.device)
        opt = torch.optim.SGD(classifier.parameters(), lr=args.lr,
                              weight_decay=optim_config["weight_decay"],
                              momentum=optim_config["momentum"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=optim_config['schedule'], gamma=0.1)

        # train the classifier on the current batch/task
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4)

        stats = train_net(opt, scheduler, classifier, criterion, train_loader, t, args.distill_coef,
                          args.epochs, device=args.device)
        # collect statistics
        ext_mem_sz += stats['disk']
        ram_usage += stats['ram']
        ################################################
        # Select most likely samples for fine tuning from current task (NC)
        MuCs, ext_mem = select_mean_idx(classifier, train_x, train_y, args.replay_examples, True, args.device)
        ################################################
        gc.collect()
        stats, preds = test_MLE_multitask(classifier, full_valdidset, args.batch_size, MuCs,
                                          preproc=preprocess_imgs,
                                          multi_heads=True, verbose=True,
                                          use_cuda=args.device)

        valid_acc += stats['acc']
        print("------------------------------------------")
        print(f"Avg. acc: {stats['acc']}")
        print("------------------------------------------")

    # Generate submission.zip
    # directory with the code snapshot to generate the results
    sub_dir = f'submissions/{args.sub_dir}'
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # copy code
    create_code_snapshot(".", f"{sub_dir}/code_snapshot")

    # generating metadata.txt: with all the data used for the CLScore
    elapsed = (time.time() - start) / 60
    print(f"Training Time: {elapsed}m, valid_acc: {np.average(valid_acc)}")
    with open(sub_dir + "/metadata.txt", "w") as wf:
        for obj in [np.average(valid_acc), elapsed, np.average(ram_usage),
                    np.max(ram_usage), np.average(ext_mem_sz), np.max(ext_mem_sz)]:
            wf.write(str(obj) + "\n")
    # test_preds.txt: with a list of labels separated by "\n"
    print("Final inference on test set...")
    full_testset = dataset.get_full_test_set()
    stats, preds = test_MLE_multitask(classifier, full_testset, args.batch_size, MuCs,
                                      preproc=preprocess_imgs,
                                      multi_heads=True, verbose=True,
                                      use_cuda=args.device)

    with open(sub_dir + "/test_preds.txt", "w") as wf:
        for pred in preds:
            wf.write(str(pred) + "\n")

    print("Experiment completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # General
    parser.add_argument('--scenario', type=str, default="multi-task-nc",
                        choices=['ni', 'multi-task-nc', 'nic'])
    parser.add_argument('--preload_data', type=bool, default=True,
                        help='preload data into RAM')

    # Model
    parser.add_argument('-cls', '--classifier', type=str, default='ResNet18',
                        choices=['ResNet18', "ResNet50", "ResNet101"])

    # Optimization
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--ft_lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--ft_batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--ft_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument("--scheduler_step", type=int, nargs='+', help="Scheduled LR rate decrease.")

    # Continual Learning
    parser.add_argument('--replay_examples', type=int, default=100,
                        help='data examples to keep in memory per class')
    parser.add_argument('--distill_coef', type=float, default=1,
                        help='data examples to keep in memory per class')
    parser.add_argument('--use_mle', type=int, default=0,
                        help='data examples to keep in memory per class')

    # Misc
    parser.add_argument('--sub_dir', type=str, default="multi-task-nc",
                        help='directory of the submission file for this exp.')
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")

    args = parser.parse_args()
    args.n_classes = 50
    args.input_size = [3, 128, 128]

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    main(args)
