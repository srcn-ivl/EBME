import torch
import argparse
import time

from core.unified_ppl import Pipeline

def test_runtime(high_synthesis=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    bi_flownet_args = argparse.Namespace()
    bi_flownet_args.pyr_level = 3
    bi_flownet_args.load_pretrain = False

    fusionnet_args = argparse.Namespace()
    fusionnet_args.high_synthesis = high_synthesis
    fusionnet_args.load_pretrain = False

    module_cfg_dict = dict(
            bi_flownet = bi_flownet_args,
            fusionnet = fusionnet_args
            )

    ppl = Pipeline(module_cfg_dict)
    ppl.device()
    ppl.eval()

    img0 = torch.randn(1, 3, 480, 640)
    img0 = img0.to(device)
    img1 = torch.randn(1, 3, 480, 640)
    img1 = img1.to(device)

    with torch.no_grad():
        for i in range(100):
            _, _ = ppl.inference(img0, img1, 0.5)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_stamp = time.time()
        for i in range(100):
            _, _ = ppl.inference(img0, img1, 0.5)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("average runtime: %4f second" % \
                ((time.time() - time_stamp) / 100))


if __name__ == "__main__":
    test_runtime()
    # test_runtime(high_synthesis=True)
