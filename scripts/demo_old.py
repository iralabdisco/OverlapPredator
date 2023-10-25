"""
Scripts for pairwise registration demo

Author: Shengyu Huang
Last modified: 22.02.2021
"""
import os, torch, time, shutil, json, glob, sys, copy, argparse
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch import optim, nn
import open3d as o3d

cwd = os.getcwd()
sys.path.append(cwd)
from datasets.indoor import IndoorDataset
from datasets.dataloader import get_dataloader
from models.architectures import KPFCNN
from lib.utils import load_obj, setup_seed, natural_key, load_config
from lib.benchmark_utils import ransac_pose_estimation, to_o3d_pcd, get_blue, get_yellow, to_tensor
from lib.trainer import Trainer
from lib.loss import MetricLoss
import shutil

setup_seed(0)


class ThreeDMatchDemo(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """

    def __init__(self, config, src_path, tgt_path):
        super(ThreeDMatchDemo, self).__init__()
        self.config = config
        self.src_path = src_path
        self.tgt_path = tgt_path

    def __len__(self):
        return 1

    def __getitem__(self, item):
        # get pointcloud
        src_pcd = torch.load(self.src_path).astype(np.float32)
        tgt_pcd = torch.load(self.tgt_path).astype(np.float32)

        # src_pcd = o3d.io.read_point_cloud(self.src_path)
        # tgt_pcd = o3d.io.read_point_cloud(self.tgt_path)
        # src_pcd = src_pcd.voxel_down_sample(0.025)
        # tgt_pcd = tgt_pcd.voxel_down_sample(0.025)
        # src_pcd = np.array(src_pcd.points).astype(np.float32)
        # tgt_pcd = np.array(tgt_pcd.points).astype(np.float32)

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)

        # fake the ground truth information
        rot = np.eye(3).astype(np.float32)
        trans = np.ones((3, 1)).astype(np.float32)
        correspondences = torch.ones(1, 2).long()

        return src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)


def main(config, demo_loader):
    config.model.eval()
    c_loader_iter = demo_loader.__iter__()
    with torch.no_grad():
        inputs = c_loader_iter.next()
        ##################################
        # load inputs to device.
        for k, v in inputs.items():
            if type(v) == list:
                inputs[k] = [item.to(config.device) for item in v]
            else:
                inputs[k] = v.to(config.device)

        ###############################################
        # forward pass
        feats, scores_overlap, scores_saliency = config.model(inputs)  # [N1, C1], [N2, C2]
        pcd = inputs['points'][0]
        len_src = inputs['stack_lengths'][0][0]
        c_rot, c_trans = inputs['rot'], inputs['trans']
        correspondence = inputs['correspondences']

        src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
        src_raw = copy.deepcopy(src_pcd)
        tgt_raw = copy.deepcopy(tgt_pcd)
        src_feats, tgt_feats = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
        src_overlap, src_saliency = scores_overlap[:len_src].detach().cpu(), scores_saliency[:len_src].detach().cpu()
        tgt_overlap, tgt_saliency = scores_overlap[len_src:].detach().cpu(), scores_saliency[len_src:].detach().cpu()

        ########################################
        # do probabilistic sampling guided by the score
        src_scores = src_overlap * src_saliency
        tgt_scores = tgt_overlap * tgt_saliency

        if (src_pcd.size(0) > config.n_points):
            idx = np.arange(src_pcd.size(0))
            probs = (src_scores / src_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size=config.n_points, replace=False, p=probs)
            src_pcd, src_feats = src_pcd[idx], src_feats[idx]
        if (tgt_pcd.size(0) > config.n_points):
            idx = np.arange(tgt_pcd.size(0))
            probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size=config.n_points, replace=False, p=probs)
            tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

        ########################################
        # run ransac and draw registration
        tsfm = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False)
        print(tsfm)


if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    # model initialization
    config.architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers - 1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
        config.architecture.append('resnetb')
    for i in range(config.num_layers - 2):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
    config.architecture.append('nearest_upsample')
    config.architecture.append('last_unary')
    config.model = KPFCNN(config).to(config.device)

    # create dataset and dataloader
    demo_set = ThreeDMatchDemo(config, config.src_pcd, config.tgt_pcd)

    neighborhood_limits = np.array([38, 36, 36, 38])
    demo_loader, _ = get_dataloader(dataset=demo_set,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=1,
                                    neighborhood_limits=neighborhood_limits)

    # load pretrained weights
    assert config.pretrain != None
    state = torch.load(config.pretrain)
    config.model.load_state_dict(state['state_dict'])

    # do pose estimation
    main(config, demo_loader)