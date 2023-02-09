import argparse
from yolov7.utils.general import check_file, set_logging
from yolov7.models import Model
import torch
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/baseline/yolor-csp-x.yaml', help='model.yaml')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = 'cpu'

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    img = torch.rand(1, 3, 640, 640).to(device)
    y = model(img, profile=True)
    print(y)