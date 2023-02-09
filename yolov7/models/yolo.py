import argparse
from copy import deepcopy
from .detection_modules import Detect, IDetect, IKeypoint, IAuxDetect, IBin
from .loading_utils import load_architecture_from_config, parse_config, update_cfg_with_model_parameters
import torch.nn as nn
import math
from .common import Conv, RepConv, RepConv_OREPA, NMS, AutoShape
import torch
from ..utils.autoanchor import check_anchor_order
from ..utils.general import check_file, set_logging
from ..utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
from typing import Union, Dict, List

import logging
logger = logging.getLogger("models")

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Model(nn.Module):
    def __init__(self,
                 cfg: Union[Dict[str, object], str] = 'yolor-csp-c.yaml',
                 n_channels: int = 3,
                 n_classes: int = None,
                 anchors: List[List[int]] = None):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.traced = False
        cfg = parse_config(cfg)
        cfg = update_cfg_with_model_parameters(cfg, n_channels, n_classes, anchors)


        self.model, self.save = load_architecture_from_config(cfg, n_input_channels=[n_channels])  # model, savelist

        self.names = [str(i) for i in range(cfg['nc'])]  # default names




        # Build strides, anchors
        detection_head = self.model[-1]  # Detect()
        if isinstance(detection_head, Detect):
            stride = 256  # 2x min stride
            detection_head.stride = torch.tensor([stride / x.shape[-2] for x in self.forward(torch.zeros(1, n_channels, stride, stride))])  # forward
            check_anchor_order(detection_head)
            detection_head.anchors /= detection_head.stride.view(-1, 1, 1)
            self.stride = detection_head.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % detection_head.stride.tolist())

        if isinstance(detection_head, IDetect):
            stride = 256  # 2x min stride
            detection_head.stride = torch.tensor([stride / x.shape[-2] for x in self.forward(torch.zeros(1, n_channels, stride, stride))])  # forward
            check_anchor_order(detection_head)
            detection_head.anchors /= detection_head.stride.view(-1, 1, 1)
            self.stride = detection_head.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % detection_head.stride.tolist())

        if isinstance(detection_head, IAuxDetect):
            stride = 256  # 2x min stride
            detection_head.stride = torch.tensor([stride / x.shape[-2] for x in self.forward(torch.zeros(1, n_channels, stride, stride))[:4]])  # forward
            #print(detection_head.stride)
            check_anchor_order(detection_head)
            detection_head.anchors /= detection_head.stride.view(-1, 1, 1)
            self.stride = detection_head.stride
            self._initialize_aux_biases()  # only run once
            # print('Strides: %s' % detection_head.stride.tolist())

        if isinstance(detection_head, IBin):
            stride = 256  # 2x min stride
            detection_head.stride = torch.tensor([stride / x.shape[-2] for x in self.forward(torch.zeros(1, n_channels, stride, stride))])  # forward
            check_anchor_order(detection_head)
            detection_head.anchors /= detection_head.stride.view(-1, 1, 1)
            self.stride = detection_head.stride
            self._initialize_biases_bin()  # only run once
            # print('Strides: %s' % detection_head.stride.tolist())

        if isinstance(detection_head, IKeypoint):
            stride = 256  # 2x min stride
            detection_head.stride = torch.tensor([stride / x.shape[-2] for x in self.forward(torch.zeros(1, n_channels, stride, stride))])  # forward
            check_anchor_order(detection_head)
            detection_head.anchors /= detection_head.stride.view(-1, 1, 1)
            self.stride = detection_head.stride
            self._initialize_biases_kpt()  # only run once
            # print('Strides: %s' % detection_head.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced=False

            if self.traced:
                if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect) or isinstance(m,
                                                                                                              IKeypoint):
                    break

            if profile:
                c = isinstance(m, (Detect, IDetect, IAuxDetect, IBin))
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=n_classes) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.n_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_aux_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=n_classes) + 1.
        m = self.model[-1]  # Detect() module
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.n_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b2.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b2.data[:, 5:] += math.log(0.6 / (m.n_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)

    def _initialize_biases_bin(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=n_classes) + 1.
        m = self.model[-1]  # Bin() module
        bc = m.bin_count
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            old = b[:, (0,1,2,bc+3)].data
            obj_idx = 2*bc+4
            b[:, :obj_idx].data += math.log(0.6 / (bc + 1 - 0.99))
            b[:, obj_idx].data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, (obj_idx+1):].data += math.log(0.6 / (m.n_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            b[:, (0,1,2,bc+3)].data = old
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_biases_kpt(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=n_classes) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.n_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))


    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                #print(f" fuse_repvgg_block")
                m.fuse_repvgg_block()
            elif isinstance(m, RepConv_OREPA):
                #print(f" switch_to_deploy")
                m.switch_to_deploy()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, (IDetect, IAuxDetect)):
                m.fuse()
                m.forward = m.fuseforward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)
