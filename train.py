# -*- encoding:utf8 -*-

import os
import pickle
import time
import warnings

import pandas as pd
from torch.autograd import Variable
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader


from data import ProDatasetTrain, graph_collate
from evalution import compute_roc, compute_aupr, compute_mcc, micro_score, acc_score, \
    compute_performance

from model import *

device = torch.device('cuda')
warnings.filterwarnings("ignore")


# if hasattr(torch.cuda, 'empty_cache'):
# 	torch.cuda.empty_cache()


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

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


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        xavier_normal_(m.weight.data)
    elif isinstance(m, nn.Linear):
        xavier_normal_(m.weight.data)


def train_epoch(model, train_loader, optimizer, epoch, all_epochs, print_freq=100):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    los = 0
    for batch_idx, (sequence_names, labels, feature, adj_sc, G_dgl, adj_ca) in enumerate(train_loader):
        # Create vaiables
        if torch.cuda.is_available():
            feature = Variable(feature.cuda())
            adj_ca = Variable(adj_ca.cuda())
            adj_sc = Variable(adj_sc.cuda())
            G_dgl.edata['ex'] = Variable(G_dgl.edata['ex'].float())
            G_dgl = G_dgl.to(torch.device('cuda:0'))

            y_true_site = Variable(labels.cuda().to(torch.float32))
        else:
            feature = Variable(feature)
            adj_sc = Variable(adj_sc)
            adj_ca = Variable(adj_ca)
            G_dgl.edata['ex'] = Variable(G_dgl.edata['ex'].float())
            y_true_site = Variable(labels.to(torch.float32))
        adj_ca = torch.squeeze(adj_ca)
        feature = torch.squeeze(feature)
        adj_sc = torch.squeeze(adj_sc)

        y_true_site = torch.squeeze(y_true_site)

        y_pred_site = model(feature, adj_sc, G_dgl, adj_ca)
        shapes = y_pred_site.data.shape
        output = y_pred_site.view(shapes[0] * shapes[1])

        loss = criterion(output, y_true_site)
        # measure accuracy and record loss
        batch_size = y_pred_site.size(0)
        # pred_out = output.ge(THREADHOLD)
        MiP, MiR, MiF, PNum, RNum = micro_score(output.data.cpu().numpy(),
                                                y_true_site.data.cpu().numpy())
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, all_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(train_loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'f_max:%.6f' % (MiP),
                'p_max:%.6f' % (MiR),
                'r_max:%.6f' % (MiF),
                't_max:%.2f' % (PNum)])
            print(res)

    # print(los)

    return batch_time.avg, losses.avg


def eval_epoch(model, loader, print_freq=10, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    # Model on eval mode
    model.eval()
    all_trues = []
    all_preds = []

    end = time.time()
    for batch_idx, (sequence_names, labels, feature, adj_sc, G_dgl, adj_ca) in enumerate(loader):
        # Create vaiables
        with torch.no_grad():
            if torch.cuda.is_available():
                feature = Variable(feature.cuda())
                adj_ca = Variable(adj_ca.cuda())
                adj_sc = Variable(adj_sc.cuda())
                G_dgl.edata['ex'] = Variable(G_dgl.edata['ex'].float())
                G_dgl = G_dgl.to(torch.device('cuda:0'))

                y_true_site = Variable(labels.cuda().to(torch.float32))
            else:
                feature = Variable(feature)
                adj_sc = Variable(adj_sc)
                adj_ca = Variable(adj_ca)
                G_dgl.edata['ex'] = Variable(G_dgl.edata['ex'].float())
                y_true_site = Variable(labels.to(torch.float32))
            adj_ca = torch.squeeze(adj_ca)
            feature = torch.squeeze(feature)
            adj_sc = torch.squeeze(adj_sc)

            y_true_site = torch.squeeze(y_true_site)
            # print(feature[-1])
            # compute output
            y_pred_site = model(feature, adj_sc, G_dgl, adj_ca)
            shapes = y_pred_site.data.shape
            output = y_pred_site.view(shapes[0] * shapes[1])

            criterion = nn.BCELoss()
            loss = criterion(output, y_true_site)

            # measure accuracy and record loss
            batch_size = y_true_site.size(0)
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                ])
                print(res)
            all_trues.append(y_true_site.cpu().detach().numpy())
            all_preds.append(output.data.cpu().numpy())

    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    auc = compute_roc(all_preds, all_trues)
    aupr = compute_aupr(all_preds, all_trues)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_preds, all_trues)
    acc_val = acc_score(predictions_max, all_trues)
    mcc = compute_mcc(predictions_max, all_trues)

    return batch_time.avg, losses.avg, acc_val, f_max, p_max, r_max, auc, aupr, t_max, mcc


def train(model, optimizer, train=None, valid=None, save=None, batch_size=32, train_file=None):
    torch.manual_seed(2023)

    train_loader = DataLoader(dataset=ProDatasetTrain(train), batch_size=batch_size, shuffle=True,
                              pin_memory=False, collate_fn=graph_collate,
                              num_workers=0, drop_last=True)
    valid_loader = DataLoader(dataset=ProDatasetTrain(valid), batch_size=batch_size, shuffle=True,
                              pin_memory=False, collate_fn=graph_collate,
                              num_workers=0, drop_last=True)

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model

    with open(os.path.join(save,
                           ' GHGPR-PPIS_results.csv'),
              'w') as f:
        f.write('epoch,loss,acc,F_value, precision,recall,auc,aupr,mcc,threadhold\n')

        # Train model
        best_F = 0
        for epoch in range(100):
            _, train_loss = train_epoch(
                model=model_wrapper,
                train_loader=train_loader,
                optimizer=optimizer,
                epoch=epoch,
                all_epochs=100,
            )
            _, valid_loss, acc, f_max, p_max, r_max, auc, aupr, t_max, mcc = eval_epoch(
                model=model_wrapper,
                loader=valid_loader,
                is_test=(not valid_loader)
            )

            print('Test_60',
                  'epoch:%03d,valid_loss:%0.5f\nacc:%0.6f,F_value:%0.6f, precision:%0.6f,recall:%0.6f,auc:%0.6f,aupr:%0.6f,mcc:%0.6f,threadhold:%0.6f\n' % (
                      (epoch + 1), valid_loss, acc, f_max, p_max, r_max, auc, aupr, mcc, t_max))

            if f_max > best_F:
                best_F = f_max
                THREADHOLD = t_max
                print("new best F_value:{0}(threadhold:{1})".format(f_max, THREADHOLD))
                torch.save(model.state_dict(), os.path.join(save, model_name + '_model.dat'))

            f.write('%03d,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f\n' % (
                (epoch + 1), valid_loss, acc, f_max, p_max, r_max, auc, aupr, mcc, t_max))


def demo(model=None, optimizer=None, save=None, train_d=None, valid_data=None):
    train(model=model, optimizer=optimizer, train=train_d, valid=valid_data, save=save,
          batch_size=1,
          train_file=train_data)
    print('Done!')


if __name__ == '__main__':
    path_dir = './Result'
    data_path = './Dataset/'
    model_name = "GHGPR_PPIS"
    train_data = data_path + 'train_335.pkl'
    with open(train_data, "rb") as fp:
        train_file = pickle.load(fp)

        IDs, sequences, labels = [], [], []

        for ID in train_file:
            if ID != '2j3rA':
                IDs.append(ID)
                item = train_file[ID]
                sequences.append(item[0])
                labels.append(list(item[1]))

    train_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    X_train = pd.DataFrame(train_dic)

    with open(data_path+'test_60.pkl', "rb") as fp:
        test_dataSet = pickle.load(fp)
        IDs, sequences, labels = [], [], []

        for ID in test_dataSet:
            IDs.append(ID)
            item = test_dataSet[ID]
            sequences.append(item[0])
            labels.append(list(item[1]))

    train_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    X_test = pd.DataFrame(train_dic)


    criterion = nn.BCELoss()
    # torch.backends.cudnn.enabled = False
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    model = GPR_HESGAT(256, 256, 0.5, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5,
                                 eps=1e-7, betas=(0.9, 0.999))

    model.to(device)
    demo(model, optimizer, path_dir, X_train, X_test)
