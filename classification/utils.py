import numpy as np

import torch
import torch.nn as nn
import ipdb
import logging
logging.getLogger().setLevel(logging.INFO)


class LinearModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.criterion = nn.CrossEntropyLoss()
        self.fc_cls = nn.Linear(in_channels, out_channels)
    
    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        assert init_linear in ['normal', 'kaiming'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        cls_score = self.fc_cls(x)
        return cls_score

    def loss(self, cls_score, labels):
        loss = self.criterion(cls_score, labels)
        return loss

    @torch.no_grad()
    def predict(self, features):
        if isinstance(features, np.ndarray):
            features = torch.tensor(features).cuda()

        cls_score = self.forward(features)
        pred = cls_score.argmax(1)
        return pred.cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, features):
        if isinstance(features, np.ndarray):
            features = torch.tensor(features).cuda()

        cls_score = self.forward(features)
        prob = torch.nn.functional.softmax(cls_score, dim=1)
        return prob.cpu().numpy()


# https://github.com/open-mmlab/OpenSelfSup/blob/master/configs/benchmarks/linear_classification/imagenet/r50_last.py#L66
def linear_eval_protocol(args, features, targets, lr=30., momentum=0.9, batch_size=256, max_epochs=100, calibrate=False):

    assert not calibrate, "Calibration is not supported yet"
    npr = np.random.RandomState(args.seed)
    n = len(features)

    in_channels = features.shape[1]
    out_channels = len(np.unique(targets))
    model = LinearModel(in_channels, out_channels)
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr, momentum=momentum
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 80])

    model.cuda()
    index = np.arange(n)
    npr.shuffle(index)
    for epoch in range(1, max_epochs+1):
        avg_loss = []
        for chunk in np.array_split(index, n // batch_size):
            optimizer.zero_grad()

            inp = torch.tensor(features[chunk]).cuda()
            tgt = torch.tensor(targets[chunk]).cuda()

            logits = model(inp)
            loss = model.loss(logits, tgt)
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_loss.append(loss.item())

        if epoch % 10 == 0:
            logging.info("=> Epoch {}: loss={:.2f}".format(epoch, np.array(avg_loss).mean()))

    return model


def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin
  

#https://github.com/markus93/NN_calibration/blob/master/scripts/utility/evaluation.py#L130-L154
def cal_ece(conf, pred, true, bin_size = 0.1):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        ece: expected calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)        
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece
        
        
def cal_mce(conf, pred, true, bin_size = 0.1):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        mce: maximum calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    cal_errors = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        cal_errors.append(np.abs(acc-avg_conf))
        
    return max(cal_errors)

