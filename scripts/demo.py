"""
Scripts for pairwise registration demo

Author: Shengyu Huang
Last modified: 22.02.2021
"""
import os, torch, time, shutil, json, glob, sys, copy, argparse
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch import optim, nn
import open3d as o3d
import csv
from tqdm import tqdm

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

import benchmark_helpers
import data_input_utils

setup_seed(0)


def main(config, neighborhood_limits, args):
    config.model.eval()
    with (torch.no_grad()):
        # Load problems txt file
        df = pd.read_csv(args.input_txt, sep=' ', comment='#')
        df = df.reset_index()
        problem_name = os.path.splitext(os.path.basename(args.input_txt))[0]

        # initialize result file
        os.makedirs(args.output_dir, exist_ok=True)
        header_comment = "# " + " ".join(sys.argv[:]) + "\n"
        header = ['id', 'initial_error', 'final_error', 'transformation', 'status_code']
        result_name = problem_name + "_result.txt"
        result_filename = os.path.join(args.output_dir, result_name)
        with open(result_filename, mode='w') as f:
            f.write(header_comment)
            csv_writer = csv.writer(f, delimiter=';')
            csv_writer.writerow(header)

        # Solve for each problem
        n_fails_oom = 0
        n_fails_other = 0
        print(problem_name)
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            problem_id, source_pcd, target_pcd, source_transform, target_pcd_filename = \
                benchmark_helpers.load_problem(row, args)

            # calculate initial error
            moved_source_pcd = copy.deepcopy(source_pcd)
            moved_source_pcd.transform(source_transform)
            initial_error = benchmark_helpers.calculate_error(source_pcd, moved_source_pcd)

            inputs = data_input_utils.get_dict_from_pcds(moved_source_pcd, target_pcd, config, neighborhood_limits)

            ##################################
            # load inputs to device.
            for k, v in inputs.items():
                if type(v) == list:
                    inputs[k] = [item.to(config.device) for item in v]
                else:
                    inputs[k] = v.to(config.device)

            ###############################################
            # forward pass
            try:
                feats, scores_overlap, scores_saliency = config.model(inputs)  # [N1, C1], [N2, C2]
                status = "ok"

            except RuntimeError as e:
                if str(e).startswith('CUDA out of memory.'):
                    status = "OOM"
                    n_fails_oom += 1
                else:
                    status = "runtime_error"
                    n_fails_other += 1

            torch.cuda.empty_cache()

            pcd = inputs['points'][0]
            len_src = inputs['stack_lengths'][0][0]
            c_rot, c_trans = inputs['rot'], inputs['trans']
            correspondence = inputs['correspondences']

            src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
            src_raw = copy.deepcopy(src_pcd)
            tgt_raw = copy.deepcopy(tgt_pcd)
            src_feats, tgt_feats = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
            src_overlap, src_saliency = scores_overlap[:len_src].detach().cpu(), scores_saliency[
                                                                                 :len_src].detach().cpu()
            tgt_overlap, tgt_saliency = scores_overlap[len_src:].detach().cpu(), scores_saliency[
                                                                                 len_src:].detach().cpu()

            ########################################
            # do probabilistic sampling guided by the score
            src_scores = src_overlap * src_saliency
            tgt_scores = tgt_overlap * tgt_saliency


            try:
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
            except Exception as e:
                print(e)
                n_fails_other += 1

            ########################################
            # run ransac and draw registration
            tsfm = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False)

            # TODO save features

    print("N fails OOM: ", n_fails_oom)
    print("N fails runtime: ", n_fails_other)


if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Choose 3DMatch or KITTI')

    # Benchmark files and dirs
    parser.add_argument('--input_txt', type=str,
                        help='Path to the problem .txt')
    parser.add_argument('--input_pcd_dir', type=str,
                        help='Path to the pcd directory')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save results to')

    args = parser.parse_args()

    if args.config == "3DMatch":
        yaml_config_path = "configs/test/indoor.yaml"
    elif args.config == "KITTI":
        yaml_config_path = "configs/test/kitty.yaml"
    else:
        raise NotImplemented
    config = load_config(yaml_config_path)
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

    # Taken from https://github.com/prs-eth/OverlapPredator/issues/21
    if args.config == "3DMatch":
        neighborhood_limits = np.array([38, 36, 36, 38])
    elif args.config == "KITTi":
        neighborhood_limits = np.array([51, 62, 71, 76])
    else:
        raise NotImplemented

    # load pretrained weights
    assert config.pretrain != None
    state = torch.load(config.pretrain)
    config.model.load_state_dict(state['state_dict'])

    # do pose estimation
    main(config, neighborhood_limits, args)
