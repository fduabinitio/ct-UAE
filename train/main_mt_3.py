import argparse
import os
import shutil
import sys
sys.path.append("./")
import time
import warnings
from random import sample
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR,StepLR
from ct.data_mt_3 import CIFData
from ct.data_mt_3 import collate_pool, get_train_val_test_loader,collate_pool_train
from ct.model_mt_3 import CrystalTransformer
from sklearn.metrics import r2_score
import random
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('--load', metavar='PATH', nargs='+', default=r'mp_13',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--file',default='mp_13_3_fbe.csv')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,

                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.006, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--embedding_lr', '--embedding-learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial embedding learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=20, nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.001, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=300, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--load_embedding_only', default=True)

train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=0.8, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=5, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=3, type=int, metavar='N',
                    help='number of hidden layers after pooling')

parser.add_argument('--wandb', default=False,type=bool, # default is false
                    help='use wandb to record training process')

parser.add_argument('--device', default='cuda:1',type=str)

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
    best_r2_score=0.001
else:
    best_mae_error = 0.

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)

if args.wandb:
    import wandb
    _config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight-decay": args.weight_decay,
        "optim": args.optim,
        "atom-fea-len": args.atom_fea_len,
        "h-fea-len": args.h_fea_len,
        "n-conv": args.n_conv,
        "n-h": args.n_h
    }
    wandb.init(project="cgcnn_trans", config=_config)



def main():
    set_seed(42)
    global args, best_mae_error,best_r2_score

    
    dataset = CIFData(args.load,file=args.file)

    collate_fn = collate_pool
    collate_fn_train=collate_pool_train
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        collate_fn_train=collate_fn_train,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)


    # obtain target value normalizer
    if args.task == 'classification':
        # FIXME RobustScalar may not work properly, needs double check
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        # TODO there is not the same as ours'
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, sample_target1,sample_target2,sample_target3, _ = collate_pool(sample_data_list)
        normalizer1 = Normalizer(sample_target1)
        normalizer2 = Normalizer(sample_target2)
        normalizer3 = Normalizer(sample_target3)
    # build model

    model = CrystalTransformer(feature_size=256, num_layers=8, num_heads=8, dim_feedforward=512)
    if args.cuda:
        model.to(args.device)

    criterion = nn.MSELoss()

    # 判断是否需要单独为embedding层和其他层设置不同的学习率
    if args.load_embedding_only and args.resume:
        # 分别获取embedding层和其他层的参数
        embedding_params = list(model.atom_embed.parameters())
        other_params = [param for name, param in model.named_parameters() if 'atom_embed' not in name]

        if args.optim == 'SGD':
            optimizer = optim.SGD([
                {'params': embedding_params, 'lr': args.embedding_lr},  # 为embedding层设置一个学习率
                {'params': other_params}  # 其他层使用默认的学习率
            ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        elif args.optim == 'Adam':
            optimizer = optim.Adam([
                {'params': embedding_params, 'lr': args.embedding_lr},  # 为embedding层设置一个学习率
                {'params': other_params}  # 其他层使用默认的学习率
            ], lr=args.lr, weight_decay=args.weight_decay)

        else:
            raise NameError('Only SGD or Adam is allowed as --optim')
    else:
        if args.optim == 'SGD':
            optimizer = optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            optimizer = optim.Adam(model.parameters(), args.lr,
                                   weight_decay=args.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed as --optim')


    scheduler = StepLR(optimizer, step_size=args.lr_milestones, gamma=0.5)

    # scheduler = CosineAnnealingLR(optimizer, T_max=150)
    #train_one_batch(train_loader, ema.model, criterion, optimizer, 0, normalizer)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer1,normalizer2,normalizer3)

        # evaluate on validation set
        mae_error,r2_score,mae_error2,r2_score2 = validate(val_loader, model, criterion, normalizer1,normalizer2,normalizer3)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        is_best = mae_error < best_mae_error
        if is_best:
            best_mae_error = min(mae_error, best_mae_error)
            best_mae_error_r2 = r2_score
        is_best_r2 = r2_score > best_r2_score
        if is_best_r2:
            best_r2_score = max(r2_score, best_r2_score)
            best_r2_score_mae = mae_error
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer1.state_dict(),
            'args': vars(args)
        }, is_best)

    print(best_mae_error)
    print(best_r2_score)
    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer1,normalizer2, test=True)

    if args.wandb:
        wandb.run.summary["best_mae_error"] = best_mae_error
        wandb.run.summary["best_mae_error_r2"] = best_mae_error_r2
        wandb.run.summary["best_r2_score"] = best_r2_score
        wandb.run.summary["best_r2_score_mae"] = best_r2_score_mae

def train(train_loader, model, criterion, optimizer, epoch, normalizer1,normalizer2,normalizer3):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()
    if args.task == 'regression':
        mae_errors1 = AverageMeter()
        r2_scores1 = AverageMeter()
        mae_errors2 = AverageMeter()
        r2_scores2 = AverageMeter()
        mae_errors3 = AverageMeter()
        r2_scores3 = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()



    end = time.time()
    for i, (input, target1,target2,target3, _) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)
        if args.cuda:
            input_var = (Variable(input[0].to(args.device)),
                         Variable(input[1].to(args.device)),
                         input[2].to(args.device),)
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2])
        # normalize target
        if args.task == 'regression':


            target_normed1 = normalizer1.norm(target1)
            target_normed2 = normalizer2.norm(target2)
            target_normed3 = normalizer3.norm(target3)
        else:
            target_normed = target1.view(-1).long()


        if args.cuda:
            target_var1 = Variable(target_normed1.to(args.device))
            target_var2 = Variable(target_normed2.to(args.device))
            target_var3 = Variable(target_normed3.to(args.device))
        else:
            target_var = Variable(target_normed)

        # compute output
        output1,output2,output3 = model(*input_var)
        output1=output1.squeeze(-1)
        output2=output2.squeeze(-1)
        output3=output3.squeeze(-1)
        loss1 = criterion(output1, target_var1)
        loss2 = criterion(output2, target_var2)
        loss3 = criterion(output3, target_var3)
        # measure accuracy and record loss
        if args.task == 'regression':
            
            mae_error1 = mae(normalizer1.denorm(output1.data.cpu()), target1)
            mae_error2 = mae(normalizer2.denorm(output2.data.cpu()), target2)
            mae_errors1.update(mae_error1, target1.size(0))
            mae_errors2.update(mae_error2, target2.size(0))
            losses1.update(loss1.data.cpu(), target1.size(0))
            losses2.update(loss2.data.cpu(), target2.size(0))
            r21 = r2_score(target1.cpu().numpy(), normalizer1.denorm(output1.data.cpu()))
            r22 = r2_score(target2.cpu().numpy(), normalizer2.denorm(output2.data.cpu()))
            r2_scores1.update(r21, target1.size(0))
            r2_scores2.update(r22, target2.size(0))


        # compute gradient and do SGD step
        loss=loss1+loss2+loss3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}] '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                      'Loss1 {loss1.val:.4f} ({loss1.avg:.4f}) '
                      'Loss2 {loss2.val:.4f} ({loss2.avg:.4f}) '
                      'MAE {mae_errors1.val:.3f} ({mae_errors1.avg:.3f}) '
                      'R2 {r21.val:.3f} ({r21.avg:.3f}) '
                      'MAE {mae_errors2.val:.3f} ({mae_errors2.avg:.3f}) '
                      'R2 {r22.val:.3f} ({r22.avg:.3f}) '
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss1=losses1,loss2=losses2 ,mae_errors1=mae_errors1, r21=r2_scores1,mae_errors2=mae_errors2, r22=r2_scores2)
                )


def validate(val_loader, model, criterion, normalizer1,normalizer2,normalizer3,test=False):
    batch_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    if args.task == 'regression':
        mae_errors1 = AverageMeter()
        r2_scores1 = AverageMeter()
        mae_errors2 = AverageMeter()
        r2_scores2 = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target1,target2,target3, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].to(args.device)),
                             Variable(input[1].to(args.device)),
                             input[2].to(args.device))
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2])

        if args.cuda:
            with torch.no_grad():
                target_var1 = Variable(target1.to(args.device))
                target_var2 = Variable(target2.to(args.device))


        output1,output2,output3 = model(*input_var)
        output1=output1.squeeze(-1)
        output2=output2.squeeze(-1)
        loss1 = criterion(output1, target_var1)
        loss2 = criterion(output2, target_var2)
        # measure accuracy and record loss
       
        _loss = loss1.data.cpu()+loss2.data.cpu()
        mae_error1 = mae(normalizer1.denorm(output1.data.cpu()), target1)
        mae_error2 = mae(normalizer2.denorm(output2.data.cpu()), target2)
        mae_errors1.update(mae_error1, target1.size(0))
        mae_errors2.update(mae_error2, target2.size(0))
        losses1.update(loss1.data.cpu(), target1.size(0))
        losses2.update(loss2.data.cpu(), target2.size(0))
        r21 = r2_score(target1.cpu().numpy(), normalizer1.denorm(output1.data.cpu()))
        r22 = r2_score(target2.cpu().numpy(), normalizer2.denorm(output2.data.cpu()))
        r2_scores1.update(r21, target1.size(0))
        r2_scores2.update(r22, target2.size(0))
        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids
        if args.wandb:
            wandb.log({'test_loss': _loss.item(), 'test_MAE': mae_error.item(),'test_r2':r2.item()})


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                      'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                      'MAE {mae_errors1.val:.3f} ({mae_errors1.avg:.3f})\t'
                      'R2 {r21.val:.3f} ({r21.avg:.3f})\t'
                      'MAE {mae_errors2.val:.3f} ({mae_errors2.avg:.3f})\t'
                      'R2 {r22.val:.3f} ({r22.avg:.3f})\t'
                    .format(
                    i, len(val_loader), batch_time=batch_time,
                     loss1=losses1,loss2=losses2 ,mae_errors1=mae_errors1, r21=r2_scores1,mae_errors2=mae_errors2, r22=r2_scores2)
                )
            

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    if args.task == 'regression':
        print(' {star} MAE {mae_errors1.avg:.3f}\t'
              'R2 {r21.avg:.3f} ' 'MAE {mae_errors2.avg:.3f}\t'
              'R2 {r22.avg:.3f}'.format(star=star_label,
                                       mae_errors1=mae_errors1, r21=r2_scores1,mae_errors2=mae_errors2, r22=r2_scores2))

        return mae_errors1.avg,r2_scores1.avg,mae_errors2.avg,r2_scores2.avg

    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))

        return auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_mt3_256.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == '__main__':
    main()
