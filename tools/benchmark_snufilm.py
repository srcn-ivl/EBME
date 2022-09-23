import cv2
import math
import numpy as np
import argparse
import warnings

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from core.unified_ppl import Pipeline
from core.dataset import SnuFilm
from core.utils.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")


def eval_subset(ppl, val_data, subset_name="easy", test_aug=False):
    psnr_list = []
    ssim_list = []
    nr_val = val_data.__len__()
    for i, data in enumerate(val_data):
        data_gpu = data[0] if isinstance(data, list) else data
        data_gpu = data_gpu.to(DEVICE, non_blocking=True) / 255.

        img0 = data_gpu[:, :3]
        img1 = data_gpu[:, 3:6]
        overlay_input = 0.5 * data_gpu[:, :3] + 0.5 * data_gpu[:, 3:6]
        gt = data_gpu[:, 6:9]

        n, c, h, w = img0.shape
        divisor = 64
        if (h % divisor != 0) or (w % divisor != 0):
            ph = ((h - 1) // divisor + 1) * divisor
            pw = ((w - 1) // divisor + 1) * divisor
            padding = (0, pw - w, 0, ph - h)
            img0 = F.pad(img0, padding, "constant", 0.5)
            img1 = F.pad(img1, padding, "constant", 0.5)

        with torch.no_grad():
            if test_aug:
                pred1, _ = ppl.inference(img0, img1)
                img0 = torch.flip(img0, dims=[2 ,3])
                img1 = torch.flip(img1, dims=[2 ,3])
                pred2, _ = ppl.inference(img0, img1)
                pred2 = torch.flip(pred2, dims=[2 ,3])
                pred = 0.5 * pred1 + 0.5 * pred2
            else:
                pred, _ = ppl.inference(img0, img1)
            pred = pred[:, :, :h, :w]

        batch_psnr = []
        batch_ssim = []
        for j in range(gt.shape[0]):
            this_gt = gt[j]
            this_pred = pred[j]
            this_overlay = overlay_input[j]
            ssim = ssim_matlab(
                    this_pred.unsqueeze(0), this_gt.unsqueeze(0)
                    ).cpu().numpy()
            ssim = float(ssim)
            ssim_list.append(ssim)
            batch_ssim.append(ssim)
            psnr = -10 * math.log10(
                    torch.mean((this_gt - this_pred) * (this_gt - this_pred)
                        ).cpu().data)
            psnr_list.append(psnr)
            batch_psnr.append(psnr)
        print('subset: {}; batch: {}/{}; psnr: {:.4f}; ssim: {:.4f}'
                .format(subset_name, i, nr_val,
                    np.mean(batch_psnr), np.mean(batch_ssim)))

    avg_psnr = np.array(psnr_list).mean()
    print('subset: {}, average psnr: {:.4f}'.format(subset_name, avg_psnr))
    avg_ssim = np.array(ssim_list).mean()
    print('subset: {}, average ssim: {:.4f}'.format(subset_name, avg_ssim))

    return avg_psnr, avg_ssim



def evaluate(ppl, data_root, batch_size, nr_data_worker=1, test_aug=False):
    print('start to evaluate the easy subset ......')
    dataset_val = SnuFilm(data_root=data_root, data_type="easy")
    val_data = DataLoader(dataset_val, batch_size=batch_size,
            num_workers=nr_data_worker, pin_memory=True)
    easy_avg_psnr, easy_avg_ssim = \
            eval_subset(ppl, val_data, subset_name="easy", test_aug=test_aug)

    print('start to evaluate the medium subset ......')
    dataset_val = SnuFilm(data_root=data_root, data_type="medium")
    val_data = DataLoader(dataset_val, batch_size=batch_size,
            num_workers=nr_data_worker, pin_memory=True)
    medium_avg_psnr, medium_avg_ssim = \
            eval_subset(ppl, val_data, subset_name="medium", test_aug=test_aug)

    print('start to evaluate the hard subset ......')
    dataset_val = SnuFilm(data_root=data_root, data_type="hard")
    val_data = DataLoader(dataset_val, batch_size=batch_size,
            num_workers=nr_data_worker, pin_memory=True)
    hard_avg_psnr, hard_avg_ssim = \
            eval_subset( ppl, val_data, subset_name="hard", test_aug=test_aug)

    print('start to evaluate the extreme subset ......')
    dataset_val = SnuFilm(data_root=data_root, data_type="extreme")
    val_data = DataLoader(dataset_val, batch_size=batch_size,
            num_workers=nr_data_worker, pin_memory=True)
    extreme_avg_psnr, extreme_avg_ssim = \
            eval_subset(ppl, val_data, subset_name="extreme", test_aug=test_aug)

    print('easy subset: avg psnr: {:.4f}'.format(easy_avg_psnr))
    print('easy subset: avg ssim: {:.4f}'.format(easy_avg_ssim))

    print('medium subset: avg psnr: {:.4f}'.format(medium_avg_psnr))
    print('medium subset: avg ssim: {:.4f}'.format(medium_avg_ssim))

    print('hard subset: avg psnr: {:.4f}'.format(hard_avg_psnr))
    print('hard subset: avg ssim: {:.4f}'.format(hard_avg_ssim))

    print('extreme subset: avg psnr: {:.4f}'.format(extreme_avg_psnr))
    print('extreme subset: avg ssim: {:.4f}'.format(extreme_avg_ssim))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='benchmarking on snu-film')

    #**********************************************************#
    # => args for data loader
    parser.add_argument('--data_root', type=str, required=True,
            help='root dir of snu-film')
    parser.add_argument('--batch_size', type=int, default=1,
            help='batch size for data loader')
    parser.add_argument('--nr_data_worker', type=int, default=1,
            help='number of the worker for data loader')

    #**********************************************************#
    # => args for optical flow model
    parser.add_argument('--flow_model_file', type=str,
            default="./checkpoints/ebme/bi-flownet.pkl",
            help='weight of the bi-directional flow model')
    parser.add_argument('--pyr_level', type=int, default=5,
            help='the number of pyramid levels in testing')

    #**********************************************************#
    # => args for frame fusion (synthesis) model
    parser.add_argument('--fusion_model_file', type=str,
            default="./checkpoints/ebme/fusionnet.pkl",
            help='weight of the frame fusion model')
    # set `high_synthesis` as True, only when training or loading
    # high-resolution synthesis model.
    parser.add_argument('--high_synthesis', type=bool, default=False,
            help='whether use high-resolution synthesis')


    #**********************************************************#
    # => whether use test augmentation
    parser.add_argument('--test_aug', type=bool, default=False,
            help='whether use test time augmentation')

    args = parser.parse_args()

    #**********************************************************#
    # => init the benchmarking environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.demo = True
    torch.backends.cudnn.benchmark = True

    #**********************************************************#
    # => init the pipeline and start to benchmark
    bi_flownet_args = argparse.Namespace()
    bi_flownet_args.pyr_level = args.pyr_level
    bi_flownet_args.load_pretrain = True
    bi_flownet_args.model_file = args.flow_model_file

    fusionnet_args = argparse.Namespace()
    fusionnet_args.high_synthesis = args.high_synthesis
    fusionnet_args.load_pretrain = True
    fusionnet_args.model_file = args.fusion_model_file

    module_cfg_dict = dict(
            bi_flownet = bi_flownet_args,
            fusionnet = fusionnet_args
            )

    ppl = Pipeline(module_cfg_dict)

    print("benchmarking on snu-film...")
    evaluate(ppl, args.data_root, args.batch_size, args.nr_data_worker,
            args.test_aug)
