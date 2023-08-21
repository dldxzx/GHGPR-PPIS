# -*- encoding:utf8 -*-

import os
import pickle
import warnings

import pandas as pd
from torch.autograd import Variable
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader

from data import ProDatasetTrain, graph_collate
from evalution import compute_roc, compute_aupr, compute_mcc, acc_score, \
    compute_performance
from model import *

warnings.filterwarnings("ignore")


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


def predict(model, loader, path_dir, pre_num=1):
    # Model on eval mode
    model.eval()
    length = len(loader)
    result = []
    all_trues = []

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
                adj_ca = Variable(adj_ca)
                adj_sc = Variable(adj_sc)
                G_dgl.edata['ex'] = Variable(G_dgl.edata['ex'].float())
                y_true_site = Variable(labels.to(torch.float32))
            adj_ca = torch.squeeze(adj_ca)
            feature = torch.squeeze(feature)
            adj_sc = torch.squeeze(adj_sc)

            y_true_site = torch.squeeze(y_true_site)
            y_pred_site = model(feature, adj_sc, G_dgl, adj_ca)
            shapes = y_pred_site.data.shape
            output = y_pred_site.view(shapes[0] * shapes[1])

        result.append(output.data.cpu().detach().numpy())
        all_trues.append(y_true_site.cpu().detach().numpy())

    # caculate
    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(result, axis=0)

    auc = compute_roc(all_preds, all_trues)
    aupr = compute_aupr(all_preds, all_trues)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_preds, all_trues)

    acc = acc_score(predictions_max, all_trues)
    mcc = compute_mcc(predictions_max, all_trues)

    print(
        'acc:%0.6f,F_value:%0.6f, precision:%0.6f,recall:%0.6f,auc:%0.6f,aupr:%0.6f,mcc:%0.6f,threadhold:%0.6f\n' % (
            acc, f_max, p_max, r_max, auc, aupr, mcc, t_max))

    predict_result = {}
    predict_result["pred"] = all_preds
    predict_result["label"] = all_trues
    result_file = path_dir+'GHGPR_PPIS_predict.pkl'
    with open(result_file, "wb") as fp:
        pickle.dump(predict_result, fp)


def demo(model_file, test_data, batch_size, path_dir):
    # Datasets
    with open(test_data, "rb") as fp:
        test_dataSet = pickle.load(fp)
        IDs, sequences, labels = [], [], []

        for ID in test_dataSet:
            IDs.append(ID)
            item = test_dataSet[ID]
            sequences.append(item[0])
            labels.append(list(item[1]))

    train_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    X_test = pd.DataFrame(train_dic)
    test_loader = DataLoader(dataset=ProDatasetTrain(X_test), batch_size=batch_size, shuffle=True,
                             pin_memory=False, collate_fn=graph_collate,
                             num_workers=0, drop_last=True)

    model = GPR_HESGAT(256, 256, 0.5, 5)
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    predict(model, test_loader, path_dir)
    print('Done!')


if __name__ == '__main__':

    path_dir = "./"
    data_path = './Dataset/'
    data_test = 'test_60.pkl'
    datas = data_path + data_test
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    model_path = './Model/'
    model_file = model_path + "5_layer_model.dat"
    demo(model_file, datas, batch_size=1, path_dir=path_dir)

