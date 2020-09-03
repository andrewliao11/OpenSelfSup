import argparse
import json
import yaml
import wandb
import time
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from datetime import date
from utils import *

import ipdb
import logging
logging.getLogger().setLevel(logging.INFO)


data_dir = Path("/h/andrewliao/UofT/cvpr2021/OpenSelfSup/data/imagenet")

def parse_args():
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--work_dir', type=str, default="./outputs")
    parser.add_argument('--filename', type=str)
    parser.add_argument('--task', type=str, choices=["semantic", "attribute"])
    parser.add_argument('--fit_fn', type=str, choices=["svm", "linear"])
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    return args


def fit_svm(args, train_features, train_targets, val_features):

    logging.info("==> Fit SVM")
    t1 = time.time()
    if args.task == "semantic":
        clf = SVC(kernel="linear", probability=True, random_state=args.seed)
        clf.fit(train_features, train_targets)
    elif args.task == "attribute":
        n_attr = train_targets.shape[1]
        clf = []
        for i in range(n_attr):
            clf_i = SVC(kernel="linear", probability=True, random_state=args.seed, class_weight="balanced")
            valid_index = np.where(train_targets[:, i] != 0)[0]
            tgt = train_targets[valid_index, i]
            tgt = (tgt == 1) + 0    # -1 -> 0, 1 -> 1
            clf_i.fit(train_features[valid_index], tgt)
            clf.append(clf_i)

    logging.info("==> Takes {:.2f} secs".format(time.time() - t1))
    return clf


def fit_linear(args, train_features, train_targets, val_features):

    logging.info("==> Fit linear classifier")
    t1 = time.time()
    if args.task == "semantic":
        clf = linear_eval_protocol(args, train_features, train_targets)
    elif args.task == "attribute":
        n_attr = train_targets.shape[1]
        clf = []
        for i in range(n_attr):
            valid_index = np.where(train_targets[:, i] != 0)[0]
            tgt = train_targets[valid_index, i]
            tgt = (tgt == 1) + 0    # -1 -> 0, 1 -> 1
            clf_i = linear_eval_protocol(args, train_features[valid_index], tgt)
            clf.append(clf_i)

    logging.info("==> Takes {:.2f} secs".format(time.time() - t1))
    return clf


def eval(args, clf, val_features, val_targets):
    
    prob = clf.predict_proba(val_features)
    pred = prob.argmax(1)
    acc = (pred == val_targets).mean()
    cm = confusion_matrix(val_targets, pred)
    per_class_acc = cm.diagonal() / cm.sum(1)
    pred_prob = np.take_along_axis(prob, pred[:, np.newaxis], axis=1).squeeze()
    ece = cal_ece(pred_prob, pred, val_targets)
    mce = cal_ece(pred_prob, pred, val_targets)
    
    return prob, {
        "acc": acc, 
        "per_class_acc": per_class_acc.tolist(), 
        "ece": ece, 
        "mce": mce
    }


def log(args, info, ratio, tag=""):

    wandb_info = {}
    for k, v in info.items():
        if isinstance(v, list):

            #for i, l in enumerate(v):
            #    wandb_info.update({tag+str(i): l})
            logging.info("==> {}{}: {}".format(tag, k, v))
            wandb_info.update({tag+k: wandb.Histogram(v)})
        else:
            logging.info("==> {}{}: {:.2f}".format(tag, k, v))
            wandb_info.update({tag+k: v})        
    
    wandb_info.update({"ratio": ratio})
    wandb.log(wandb_info)


def load(args, ratio):

    features = np.load(args.filename)
    npr = np.random.RandomState(args.seed)

    if args.task == "semantic":
        logging.info("==> Load semantic label")
        n_classes = 100
        n_per_class = 300
        class_idx = np.arange(1000)
        #npr.shuffle(class_idx)
        class_idx = class_idx[:n_classes]

        with (data_dir / "meta" / "train_labeled.txt").open() as f:
            content = f.readlines()
            targets = np.array([int(i.split(" ")[1].replace("\n", "")) for i in content])

        data_idx = np.array([j for i in class_idx for j in np.where(targets == i)[0][:n_per_class]])
        npr.shuffle(data_idx)
        n = len(data_idx)
        
        logging.info("==> Split train({})/val({}) from original train set".format(int(n*ratio), n-int(n*ratio)))
        train_features = features[data_idx[:int(n*ratio)]]
        val_features = features[data_idx[int(n*ratio):n]]

        train_targets = targets[data_idx[:int(n*ratio)]]
        val_targets = targets[data_idx[int(n*ratio):n]]
        class_names = [None]

    elif args.task == "attribute":
        logging.info("==> Load attribute label")

        with (data_dir / "meta_attr" / "train_labeled.txt").open() as f:
            content = f.readlines()
            targets = np.array([i.split(" ")[1:] for i in content])
            targets = targets.astype(np.int)
            
        logging.info("==> Split train/val from original train set")
        n = len(targets)
        train_features = features[:int(n*ratio)]
        val_features = features[int(n*ratio):n]

        train_targets = targets[:int(n*ratio)]
        val_targets = targets[int(n*ratio):n]
        with (data_dir / "meta_attr" / "class_name_to_index.json").open() as f:
            d = json.load(f)
            index_to_class_name = {v: k for k, v in d.items()}
            class_names = [index_to_class_name[i] for i in range(len(d))]
            

    return train_features, train_targets, val_features, val_targets, class_names
    

def main():

    args = parse_args()
    today = date.today()
    filename = args.filename.split("/")[-1]
    wandb.init(project="self-supervised-compare", config=args, name="{}-{}-{}-{}".format(today.strftime("%m-%d-%y"), filename, args.task, args.fit_fn))

    work_dir = Path(args.work_dir)

    with open(work_dir / "config.yaml", "w") as f:
        yaml.dump(args, f, default_flow_style=False)

    info = {}
    if args.fit_fn == "svm":
        fit_fn = fit_svm
    elif args.fit_fn == "linear":
        fit_fn = fit_linear


    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        logging.info("=> Ratio: {:.1f}".format(ratio))
        train_features, train_targets, val_features, val_targets, class_names = load(args, ratio)
        clf = fit_fn(args, train_features, train_targets, val_features)

        if args.task == "semantic":
            prob, info = eval(args, clf, val_features, val_targets)
            log(args, info, ratio)
        elif args.task == "attribute":
            n_attr = val_targets.shape[1]
            prob_list = []
            for i in range(n_attr):
                valid_index = np.where(val_targets[:, i] != 0)[0]
                tgt = val_targets[valid_index, i]
                tgt = (tgt == 1) + 0    # -1 -> 0, 1 -> 1
                prob, info = eval(args, clf[i], val_features[valid_index], tgt)
                prob_list.append(prob)
                log(args, info, ratio, tag=class_names[i]+"/")

            prob = np.array(prob_list)
            
        
        info.update({"ratio-{}".format(ratio): prob})

    np.savez(work_dir / "{}-{}.json".format(filename, args.task), **info)


if __name__ == '__main__':
    main()
