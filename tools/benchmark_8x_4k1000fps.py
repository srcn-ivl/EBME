import cv2
import math
import numpy as np
import argparse
import warnings

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from core.unified_ppl import Pipeline
from core.dataset import X_Test
from core.utils.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

def evaluate(ppl, test_data_path, batch_size, nr_data_worker=1, test_aug=False):
    dataset = X_Test(test_data_path=test_data_path, multiple=8)
    val_data = DataLoader(dataset, batch_size=batch_size,
            num_workers=nr_data_worker, pin_memory=True)

    psnr_list = []
    ssim_list = []
    nr_val = val_data.__len__()
    for i, (frames, t_value, scene_name, frame_range) in enumerate(val_data):
        torch.cuda.empty_cache()
        frames = frames.to(DEVICE, non_blocking=True) / 255.
        B, C, T, h, w = frames.size()
        t_value = t_value.to(DEVICE, non_blocking=True)
        img0 = frames[:, :, 0, :, :]
        img1 = frames[:, :, 1, :, :]
        gt = frames[:, :, 2, :, :]
        overlay_input = 0.5 * img0 + 0.5 * img1

        divisor = 256
        if (h % divisor != 0) or (w % divisor != 0):
            ph = ((h - 1) // divisor + 1) * divisor
            pw = ((w - 1) // divisor + 1) * divisor
            divisor = (0, pw - w, 0, ph - h)
            img0 = F.pad(img0, divisor, "constant", 0.5)
            img1 = F.pad(img1, divisor, "constant", 0.5)

        img0 = F.interpolate(img0, (ph, pw), mode="bilinear")
        img1 = F.interpolate(img1, (ph, pw), mode="bilinear")
        with torch.no_grad():
            if test_aug:
                pred1, _ = ppl.inference(img0, img1, time_period=t_value)
                img0 = torch.flip(img0, dims=[2 ,3])
                img1 = torch.flip(img1, dims=[2 ,3])
                pred2, _= ppl.inference(img0, img1, time_period=t_value)
                pred2 = torch.flip(pred2, dims=[2 ,3])
                pred = 0.5 * pred1 + 0.5 * pred2
            else:
                pred, _  = ppl.inference(img0, img1, time_period=t_value)
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

        print('batch: {}/{}; psnr: {:.4f}; ssim: {:.4f}'.format(i, nr_val,
            np.mean(batch_psnr), np.mean(batch_ssim)))

    psnr = np.array(psnr_list).mean()
    print('average psnr: {:.4f}'.format(psnr))
    ssim = np.array(ssim_list).mean()
    print('average ssim: {:.4f}'.format(ssim))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='benchmarking on 4k1000fps' +\
            'dataset for 8x multiple interpolation')

    #**********************************************************#
    # => args for data loader
    parser.add_argument('--test_data_path', type=str, required=True,
            help='the path of 4k10000fps benchmark')
    parser.add_argument('--nr_data_worker', type=int, default=1,
            help='number of the worker for data loader')
    parser.add_argument('--batch_size', type=int, default=1,
            help='batchsize for data loader')


    #**********************************************************#
    # => args for optical flow model
    parser.add_argument('--flow_model_file', type=str,
            default="./checkpoints/ebme/bi-flownet.pkl",
            help='the path of the bi-directional flow model weight')
    parser.add_argument('--pyr_level', type=int, default=7,
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

    print("benchmarking on 4k1000fps...")
    evaluate(ppl, args.test_data_path, args.batch_size, args.nr_data_worker,
            args.test_aug)
