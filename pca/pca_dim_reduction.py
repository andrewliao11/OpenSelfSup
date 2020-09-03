import time
import yaml
import json
import argparse
import numpy as np
import os.path as osp
from sklearn import decomposition
from collections import Counter
from annoy import AnnoyIndex
from datetime import date
from tqdm import tqdm
from pathlib import Path


import ipdb
import logging
logger = logging.getLogger(__name__)

data_dir = Path("/h/andrewliao/UofT/cvpr2021/OpenSelfSup/data/imagenet")

def construct_knn(features, seed):
    logger.info("Construcing Tree")
    d = features.shape[1]
    t = AnnoyIndex(d, "euclidean")
    t.set_seed(seed)

    for i, v in enumerate(features):
        t.add_item(i, v)

    n_tree = 10
    t.build(n_tree) # 10 trees

    return t


def query(t, q_idx, targets, k):
    neighbour, dist = t.get_nns_by_item(q_idx, k+1, include_distances=True)
    neighbour_t = targets[neighbour[1:]]
    b = Counter(neighbour_t)
    return b.most_common(1)[0][0]


def test_features(features, targets, K, seed):

    t1 = time.time()
    t = construct_knn(features, seed)

    pred = [query(t, i, targets, k=K) for i in tqdm(range(len(features)))]
    pred = np.array(pred)

    acc = (pred == targets).mean()
    logger.info("[{:.2f} sec]\tAccuracy: {:.2f}".format(time.time() - t1, acc))
    return acc


def PCA(features, dim, seed):

    logger.info("Fitting PCA")
    t1 = time.time()
    pca = decomposition.PCA(n_components=dim, random_state=seed)
    pca.fit(features)
    pca_features = pca.transform(features)
    logger.info("[{:.2f} sec]\t".format(time.time() - t1))
    return pca_features


def parse_args():
    parser = argparse.ArgumentParser(
        description='PCA dimenstion reduction')
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--split', type=str, choices=["train", "val"])
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    return args


def _find_targets(args):
    with (data_dir / "meta" / (args.split + "_labeled.txt")).open("r") as f:
        content = f.readlines()
        targets = np.array([int(i.split(" ")[1].replace("\n", "")) for i in content])
    return targets


def _find_paths(args):
    with (data_dir / "meta" / (args.split + "_labeled.txt")).open("r") as f:
        content = f.readlines()
        paths = np.array([i.split(" ")[0] for i in content])
    return paths


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)

    assert args.n_classes > 0 and args.n_classes <= 1000
    
    with open(work_dir / "config.yaml", "w") as f:
        yaml.dump(args, f, default_flow_style=False)

    targets = _find_targets(args)
    paths = _find_paths(args)
    class_name_to_index = json.load((data_dir / "meta" / "class_name_to_index.json").open("r"))
    index_to_class_name = {int(v): k for k, v in class_name_to_index.items()}

    filename = Path(args.filename)
    features = np.load(filename)
    dim = args.dim

    idx = np.array([j for i in range(args.n_classes) for j in np.where(targets == i)[0]])
    features = features[idx]
    targets = targets[idx]
    paths = paths[idx]
    class_names = np.array([index_to_class_name[i] for i in range(args.n_classes)])

    logger.info("Number of data points: {}".format(len(features)))

    logger.info("Test features")
    acc = test_features(features, targets, args.k, args.seed)

    pca_features = PCA(features, args.dim, args.seed)
    logger.info("Test PCA features")
    pca_acc = test_features(pca_features, targets, args.k, args.seed)


    logger.info("Saving PCA dim-reduced features")
    data = {"features": features, 
        "pca_features": pca_features, 
        "targets": targets, 
        "paths": paths, 
        "class_names": class_names, 
        "info": "The file includes the original features from ({}) and the \
and the dimension-reduced features (dim={}). \
We sample total {} data examples from the first {} classes. \
The {}-nn top1 accuracy of the original features is {:.2f}. \
The {}-nn top1 accuracy of the PCA features is {:.2f}".format(filename, args.dim, len(idx), args.n_classes, args.k, acc, args.k, pca_acc)
    }
    np.savez(work_dir / ("pca_{}_".format(args.n_classes)+filename.name.replace(".npy", "")), **data)


if __name__ == '__main__':
    main()
