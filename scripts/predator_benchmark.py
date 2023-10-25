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

def pcd2xyz(pcd):
    return np.asarray(pcd.points)

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
        header = ['id', 'status_code']
        result_name = problem_name + "_status.txt"
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
                torch.cuda.empty_cache()
                results = [problem_id, 'ok']
                with open(result_filename, mode='a') as f:
                    csv_writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')
                    csv_writer.writerow(results)
            except RuntimeError as e:
                if str(e).startswith('CUDA out of memory.'):
                    torch.cuda.empty_cache()
                    n_fails_oom += 1
                    results = [problem_id, 'OOM']
                    with open(result_filename, mode='a') as f:
                        csv_writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')
                        csv_writer.writerow(results)
                    continue
                else:
                    torch.cuda.empty_cache()
                    n_fails_other += 1
                    results = [problem_id, 'runtime_error']
                    with open(result_filename, mode='a') as f:
                        csv_writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')
                        csv_writer.writerow(results)
                    continue

            pcd = inputs['points'][0]
            len_src = inputs['stack_lengths'][0][0]

            src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
            src_feats, tgt_feats = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
            src_overlap, src_saliency = scores_overlap[:len_src].detach().cpu(), scores_saliency[
                                                                                 :len_src].detach().cpu()
            tgt_overlap, tgt_saliency = scores_overlap[len_src:].detach().cpu(), scores_saliency[
                                                                                 len_src:].detach().cpu()

            ########################################
            # do probabilistic sampling guided by the score
            src_scores = src_overlap * src_saliency
            tgt_scores = tgt_overlap * tgt_saliency


            if (src_pcd.size(0) > args.n_points):
                idx_src = np.arange(src_pcd.size(0))
                probs_src = (src_scores / src_scores.sum()).numpy().flatten()
                idx_src = np.random.choice(idx_src, size=args.n_points, replace=False, p=probs_src)
                src_pcd, src_feats = src_pcd[idx_src], src_feats[idx_src]
            if (tgt_pcd.size(0) > args.n_points):
                idx = np.arange(tgt_pcd.size(0))
                probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
                idx = np.random.choice(idx, size=args.n_points, replace=False, p=probs)
                tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

            ########################################
            # Save out the output
            source_features_npz = os.path.join(args.output_dir, '{}'.format(problem_id))
            src_pcd = src_pcd.detach().cpu().numpy()
            src_feats = src_feats.detach().cpu().numpy()
            np.savez_compressed(source_features_npz, xyz_down=src_pcd, features=src_feats)

            target_features_npz = os.path.join(args.output_dir, os.path.splitext(target_pcd_filename)[0])
            if not os.path.exists(target_features_npz):
                tgt_pcd = tgt_pcd.detach().cpu().numpy()
                tgt_feats = tgt_feats.detach().cpu().numpy()
                np.savez_compressed(target_features_npz, xyz_down=tgt_pcd, features=tgt_feats)

    print("N fails OOM: ", n_fails_oom)
    print("N fails runtime: ", n_fails_other)


if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Choose 3DMatch or KITTI')
    parser.add_argument('n_points', type=int, help="Number of points to describe")

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
        yaml_config_path = "configs/test/kitti.yaml"
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
    elif args.config == "KITTI":
        neighborhood_limits = np.array([51, 62, 71, 76])
    else:
        raise NotImplemented

    # load pretrained weights
    assert config.pretrain != None
    state = torch.load(config.pretrain)
    config.model.load_state_dict(state['state_dict'])

    # do pose estimation
    main(config, neighborhood_limits, args)
