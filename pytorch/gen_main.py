import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import pdb
import wandb

from data   import *
from utils  import get_logger, get_temp_logger, logging_per_task, onehot, distillation_KL_loss, \
                   naive_cross_entropy_loss,
from copy   import deepcopy
from pydoc  import locate
from model  import ResNet18, MLP, classifier
from VAE    import VAE
from VAE.loss import calculate_loss

# Arguments
# -----------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default = 'split_mnist',
    choices=['split_mnist', 'permuted_mnist', 'split_cifar10', 'split_cifar100', 'miniimagenet'])
parser.add_argument('--n_tasks', type=int, default=-1,
    help='total number of tasks. -1 does default amount for the dataset')
parser.add_argument('-r','--reproc', type=int, default=1,
    help='if on, no randomness in numpy and torch')
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--cls_iters', type=int, default=1,
    help='number of training iterations for the classifier')
parser.add_argument('--gen_iters', type=int, default=1,
    help='number of training iterations for the generator')
parser.add_argument('--cls_hiddens', type=int, default=400,
    help='number hidden dim in the classifier')
parser.add_argument('-bs', '--batch_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--use_conv', action='store_true'
    help='If True, use a ResNet, if not MLP')
parser.add_argument('-s', '--samples_per_task', type=int, default=-1,
    help='if negative, full dataset is used')
parser.add_argument('--n_runs', type=int, default=1,
    help='number of runs to average performance')
parser.add_argument('-pe', '--print_every', type=int, default=100,
    help="print metrics every this minibatch")
# logging
parser.add_argument('-l', '--log', type=str, default='off', choices=['off', 'online', 'offline'],
    help='enable WandB logging')
parser.add_argument('--result_dir', type=str, default='temp',
    help='directory inside Results/ where we save results, samples and WandB project')
parser.add_argument('--name', type=str, default=None,
    help='name of the run in WandB project')

parser.add_argument('-gm', '--gen_method', type=str, default='rand_gen',
    choices=['no_rehearsal', 'gen_replay'])
parser.add_argument('--reuse_samples', type=int, default=0
    help='at every iteration, resample or not')
parser.add_argument('--n_mem', type=int, default=1,
    help='number of retrieved memories')
parser.add_argument('-mc', '--mem_coeff', type=float, default=1.0,
    help='replay loss relative weight')

#------- Generative Model ------#
parser.add_argument('-o', '--output_loss', type=str, default=None,
    choices=[None, 'bernouilli', 'mse', 'multinomial'])
parser.add_argument('-ga', '--gen_architecture', type=str, default='MLP',
    choices=['MLP', 'GatedConv'])
parser.add_argument('-gd', '--gen_depth', type=int, default=1,
    help='depth of the generator (fixed to 6 in GatedConv)')
parser.add_argument('-gh', '--gen_hiddens', type=int, default=256,
    help='number of hidden variable in generator')
parser.add_argument('-do', '--dropout', type=float, default=0,
    help='dropout probability')
parser.add_argument('-w', '--warmup', type=int, default=1000, metavar='N',
    help='number of datapoints for warm-up. Set to 0 to turn warmup off.')
parser.add_argument('--max_beta', type=float, default=1.,
    help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0,
    help='min beta for warm-up')
parser.add_argument('--z_size', type=int, default=64,
    help='how many stochastic hidden units')
#---------------------------------#

args = parser.parse_args()

# Obligatory overhead
# -----------------------------------------------------------------------------------------

result_path = os.path.join('Results', args.result_dir)
if not os.path.exists(result_path): os.mkdir(result_path)
sample_path = os.path.join(*['Results', args.result_dir, 'samples/'])
if not os.path.exists(sample_path): os.mkdir(sample_path)

args.cuda = torch.cuda.is_available()
if args.cuda: args.device = 'cuda:0'
else: args.device = 'cpu'

if args.reproc:
    seed=0
    torch.manual_seed(seed)
    np.random.seed(seed)

# this needs to be fixed:
if args.gen_architecture=='GatedConv':
    args.gen_depth = 6

# fetch data
data = locate('data.get_%s' % args.dataset)(args)

# make dataloaders
train_loader, val_loader, test_loader  = [CLDataLoader(elem, args, train=t) \
        for elem, t in zip(data, [True, False, False])]

if args.log != 'off':
    if args.name is None: wandb.init(project=args.result_dir)
    else: wandb.init(project=args.result_dir, name=args.name)
    wandb.config.update(args)
else:
    wandb = None

# create logging containers
LOG = get_logger(['gen_loss', 'cls_loss', 'acc'],
        n_runs=args.n_runs, n_tasks=args.n_tasks)


# Train the model
# -----------------------------------------------------------------------------------------

# --------------
# Begin Run Loop
for run in range(args.n_runs):

    # REPRODUCTIBILITY
    if args.reproc:
        np.random.seed(run)
        torch.manual_seed(run)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # CLASSIFIER
    if args.use_conv:
        cls = ResNet18(args.n_classes, nf=20, input_size=args.input_size)
    else:
        cls = classifier(args).to(args.device)

    opt = torch.optim.SGD(cls.parameters(), lr=args.lr)
    if run == 0:
        print("number of classifier parameters:",
                sum([np.prod(p.size()) for p in cls.parameters()]))

    # GENERATIVE MODELING
    gen = VAE.VAE(args).to(args.device)
    opt_gen = torch.optim.Adam(gen.parameters())
    if run == 0:
        print("number of generator parameters: ",
                sum([np.prod(p.size()) for p in gen.parameters()]))
    # INIT
    prev_gen, prev_model = None, None

    #----------------
    # Begin Task Loop
    for task, tr_loader in enumerate(train_loader):

        print('\n--------------------------------------')
        print('Run #{} Task #{} TRAIN'.format(run, task))
        print('--------------------------------------\n')

        cls = cls.train()
        if gen is not None:
            gen = gen.train()

        sample_amt = 0

        # ----------------
        # Begin Epoch Loop
        for epoch in range(args.n_epochs):

            #---------------------
            # Begin Minibatch Loop
            for i, (data, target) in enumerate(tr_loader):

                if args.unit_test and i > 10: break
                if sample_amt > args.samples_per_task > 0: break
                sample_amt += data.size(0)

                data, target = data.to(args.device), target.to(args.device)

                # KL cost annealing
                args.beta = min([(sample_amt) / max([args.warmup, 1.]), args.max_beta])

                #------ Train Generator ------#

                #-------------------------------
                # Begin Generator Iteration Loop
                for it in range(args.gen_iters):

                    x_mean, z_mu, z_var, ldj, z0, zk = gen(data)
                    gen_loss, rec, kl, _ = calculate_loss(x_mean, data, z_mu,
                            z_var, z0, zk, ldj, args, beta=args.beta)

                    tot_gen_loss = 0 + gen_loss

                    if task > 0 and args.gen_method != 'no_rehearsal':

                        if it == 0 or not args.reuse_samples:

                            if args.gen_method == 'rand_gen':
                                mem_x = prev_gen.generate(args.batch_size*args.n_mem).detach()

                        mem_x_mean, z_mu, z_var, ldj, z0, zk = gen(mem_x)
                        mem_gen_loss, mem_rec, mem_kl, _ = calculate_loss(mem_x_mean, mem_x, z_mu,
                                z_var, z0, zk, ldj, args, beta=args.beta)

                        tot_gen_loss += args.mem_coeff * mem_gen_loss

                    opt_gen.zero_grad()
                    tot_gen_loss.backward()
                    opt_gen.step()

                # End Generator Iteration Loop
                #------------------------------

                if i % args.print_every == 0:
                    print('current VAE loss = {:.4f} (rec: {:.4f} + beta: {:.2f} * kl: {:.2f})'
                        .format(gen_loss.item(), rec.item(), args.beta, kl.item()))
                    if task > 0:
                        print('memory VAE loss = {:.4f} (rec: {:.4f} + beta: {:.2f} * kl: {:.2f})'
                            .format(mem_gen_loss.item(), mem_rec.item(), args.beta, mem_kl.item()))


                #------ Train Classifier-------#

                #--------------------------------
                # Begin Classifier Iteration Loop
                for it in range(args.cls_iters):

                    logits = cls(data)
                    cls_loss = F.cross_entropy(logits, target, reduction='mean')
                    tot_cls_loss = 0 + cls_loss

                    if task > 0:

                        if it == 0 or not args.reuse_samples:

                            mem_x = prev_gen.generate(args.batch_size*args.n_mem).detach()
                            mem_y = torch.softmax(prev_cls(mem_x), dim=1).detach()

                        mem_logits = cls(mem_x)

                        mem_cls_loss = naive_cross_entropy_loss(mem_logits, mem_y)

                        tot_cls_loss += args.mem_coeff * mem_cls_loss

                    opt.zero_grad()
                    tot_cls_loss.backward()
                    opt.step()

                # End Classifer Iteration Loop
                #-----------------------------

                if i % args.print_every == 0:
                    pred = logits.argmax(dim=1, keepdim=True)
                    acc = pred.eq(target.view_as(pred)).sum().item() / pred.size(0)
                    print('current training accuracy: {:.2f}'.format(acc))
                    if task > 0:
                        pred = mem_logits.argmax(dim=1, keepdim=True)
                        mem_y = mem_y.argmax(dim=1, keepdim=True)
                        acc = pred.eq(mem_y.view_as(pred)).sum().item() / pred.size(0)
                        print('memory training accuracy: {:.2f}'.format(acc))

            # End Minibatch Loop
            #-------------------

        # End Epoch Loop
        #---------------

        # ------------------------ eval ------------------------ #

        print('\n--------------------------------------')
        print('Run #{} Task #{} EVAL'.format(run, task))
        print('--------------------------------------\n')

        with torch.no_grad():

            cls = cls.eval()
            prev_cls = deepcopy(cls)

            if gen is not None:
                gen = gen.eval()
                prev_gen = deepcopy(gen)

                if args.dataset is not 'permuted_mnist':

                    # save some pretty images:
                    gen_images = gen.generate(25).to('cpu')
                    sample_path_ = os.path.join(sample_path,'task{}.png'.format(task))
                    save_image(gen_images, sample_path_, nrow=5)
                    if wandb is not None:
                        logged_gen = wandb.Image(gen_images[:16],
                            caption="generations task {}".format(task))
                        wandb.log({"Generations Run {}".format(run): logged_gen}, step=task)

            eval_loaders = [('valid', val_loader), ('test', test_loader)]

            #----------------
            # Begin Eval Loop
            for mode, loader_ in eval_loaders:

                #----------------
                # Begin Task Eval Loop
                for task_t, te_loader in enumerate(loader_):
                    if task_t > task: break
                    LOG_temp = get_temp_logger(None, ['gen_loss', 'cls_loss', 'acc'])

                    #---------------------
                    # Begin Minibatch Eval Loop
                    for i, (data, target) in enumerate(te_loader):

                        #if args.cuda:
                        data, target = data.to(args.device), target.to(args.device)

                        logits = cls(data)

                        loss = F.cross_entropy(logits, target)
                        pred = logits.argmax(dim=1, keepdim=True)

                        LOG_temp['acc'] += [pred.eq(target.view_as(pred)).sum().item()/pred.size(0)]
                        LOG_temp['cls_loss'] += [loss.item()]

                        if gen is not None:
                            x_mean, z_mu, z_var, ldj, z0, zk = gen(data)
                            gen_loss, rec, kl, bpd = calculate_loss(x_mean, data, z_mu, z_var, z0,
                                    zk, ldj, args, beta=args.beta)
                            LOG_temp['gen_loss'] += [gen_loss.item()]

                    # End Minibatch Eval Loop
                    #-------------------

                    logging_per_task(wandb, LOG, run, mode, 'acc', task, task_t,
                             np.round(np.mean(LOG_temp['acc']),2))
                    logging_per_task(wandb, LOG, run, mode, 'cls_loss', task, task_t,
                             np.round(np.mean(LOG_temp['cls_loss']),2))
                    logging_per_task(wandb, LOG, run, mode, 'gen_loss', task, task_t,
                             np.round(np.mean(LOG_temp['gen_loss']),2))


                # End Task Eval Loop
                #-------------------

                print('\n{}:'.format(mode))
                print(LOG[run][mode]['acc'])

            # End Eval Loop
            #--------------

        # End torch.no_grad()
        #--------------------

    # End Task Loop
    #--------------

    print('--------------------------------------')
    print('Run #{} Final Results'.format(run))
    print('--------------------------------------')
    for mode in ['valid','test']:

        # accuracy
        final_accs = LOG[run][mode]['acc'][:,task]
        logging_per_task(wandb, LOG, run, mode, 'final_acc', task,
            value=np.round(np.mean(final_accs),2))

        # forgetting
        best_acc = np.max(LOG[run][mode]['acc'], 1)
        final_forgets = best_acc - LOG[run][mode]['acc'][:,task]
        logging_per_task(wandb, LOG, run, mode, 'final_forget', task,
                value=np.round(np.mean(final_forgets[:-1]),2))

        # VAE loss
        final_elbos = LOG[run][mode]['gen_loss'][:,task]
        logging_per_task(wandb, LOG, run, mode, 'final_elbo', task,
            value=np.round(np.mean(final_elbos),2))

        print('\n{}:'.format(mode))
        print('final accuracy: {}'.format(final_accs))
        print('average: {}'.format(LOG[run][mode]['final_acc']))
        print('final forgetting: {}'.format(final_forgets))
        print('average: {}'.format(LOG[run][mode]['final_forget']))
        print('final VAE loss: {}'.format(final_elbos))
        print('average: {}\n'.format(LOG[run][mode]['final_elbo']))

        try:
            mir_worked_frac = mir_success/(mir_tries)
            logging_per_task(wandb, LOG, run, mode, 'final_mir_worked_frac', task,
                mir_worked_frac)
            print('mir worked \n', mir_worked_frac)
        except:
            pass

# End Run Loop
#-------------

print('--------------------------------------')
print('--------------------------------------')
print('FINAL Results')
print('--------------------------------------')
print('--------------------------------------')
for mode in ['valid','test']:

    # accuracy
    final_accs = [LOG[x][mode]['final_acc'] for x in range(args.n_runs)]
    final_acc_avg = np.mean(final_accs)
    final_acc_se = np.std(final_accs) / np.sqrt(args.n_runs)

    # forgetting
    final_forgets = [LOG[x][mode]['final_forget'] for x in range(args.n_runs)]
    final_forget_avg = np.mean(final_forgets)
    final_forget_se = np.std(final_forgets) / np.sqrt(args.n_runs)

    # VAE loss
    final_elbos = [LOG[x][mode]['final_elbo'] for x in range(args.n_runs)]
    final_elbo_avg = np.mean(final_elbos)
    final_elbo_se = np.std(final_elbos) / np.sqrt(args.n_runs)

    print('\nFinal {} Accuracy: {:.3f} +/- {:.3f}'.format(mode, final_acc_avg, final_acc_se))
    print('\nFinal {} Forget: {:.3f} +/- {:.3f}'.format(mode, final_forget_avg, final_forget_se))
    print('\nFinal {} ELBO: {:.3f} +/- {:.3f}'.format(mode, final_elbo_avg, final_elbo_se))

    if wandb is not None:
        wandb.log({mode+'final_acc_avg':final_acc_avg})
        wandb.log({mode+'final_acc_se':final_acc_se})
        wandb.log({mode+'final_forget_avg':final_forget_avg})
        wandb.log({mode+'final_forget_se':final_forget_se})
        wandb.log({mode+'final_elbo_avg':final_elbo_avg})
        wandb.log({mode+'final_elbo_se':final_elbo_se})
