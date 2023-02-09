from torch import nn
from typing import List
from .common import Conv, RobustConv, RobustConv2, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC, SPP, SPPF, \
    SPPCSPC, GhostSPPCSPC, Focus, Stem, GhostStem, Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC, \
    RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC, Res, ResCSPA, ResCSPB, ResCSPC, RepRes, \
    RepResCSPA, RepResCSPB, RepResCSPC, ResX, ResXCSPA, ResXCSPB, ResXCSPC, RepResX, RepResXCSPA, RepResXCSPB, \
    RepResXCSPC, Ghost, GhostCSPA, GhostCSPB, GhostCSPC, SwinTransformerBlock, STCSPA, STCSPB, STCSPC, \
    SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC, Concat, Chuncat, Shortcut, Foldcut, ReOrg, Contract, Expand
from .experimental import MixConv2d, CrossConv
from .detection_modules import Detect, IDetect, IKeypoint, IAuxDetect, IBin
from ..utils.general import make_divisible
import logging
import yaml
from typing import Union, Dict, Tuple
from copy import deepcopy

logger = logging.getLogger("models")

def load_architecture_from_config(yaml_dict: dict, n_input_channels: List[int])->Tuple[nn.Module, list]:  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))

    anchors: List[List[int]] = yaml_dict['anchors']
    n_classes: int = yaml_dict['nc']
    depth_multiple: float = yaml_dict['depth_multiple']
    width_multiple: float = yaml_dict['width_multiple']

    n_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    n_outputs_per_cell = n_anchors * (n_classes + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, n_output_channels = [], [], n_input_channels[-1]  # layers, savelist, input_channels out
    for i, (f, n, module_class, args) in enumerate(yaml_dict['backbone'] + yaml_dict['head']):  # from, number, module, args
        module_class = eval(module_class) if isinstance(module_class, str) else module_class  # eval strings

        for j, a in enumerate(args):
            if isinstance(a, str):

                try:
                    args[j] = yaml_dict[a]
                except KeyError:
                    pass

                try:
                    args[j] = eval(a)
                except NameError:
                    pass


        n = max(round(n * depth_multiple), 1) if n > 1 else n  # depth gain
        if module_class in [nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC,
                 SPP, SPPF, SPPCSPC, GhostSPPCSPC, MixConv2d, Focus, Stem, GhostStem, CrossConv,
                 Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                 RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                 Res, ResCSPA, ResCSPB, ResCSPC,
                 RepRes, RepResCSPA, RepResCSPB, RepResCSPC,
                 ResX, ResXCSPA, ResXCSPB, ResXCSPC,
                 RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC,
                 Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
                 SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
                 SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC]:
            c1, n_output_channels = n_input_channels[f], args[0]

            if n_output_channels != n_outputs_per_cell:  # if not output
                n_output_channels = make_divisible(n_output_channels * width_multiple, 8)

            args = [c1, n_output_channels, *args[1:]]
            if module_class in [DownC, SPPCSPC, GhostSPPCSPC,
                     BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                     RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                     ResCSPA, ResCSPB, ResCSPC,
                     RepResCSPA, RepResCSPB, RepResCSPC,
                     ResXCSPA, ResXCSPB, ResXCSPC,
                     RepResXCSPA, RepResXCSPB, RepResXCSPC,
                     GhostCSPA, GhostCSPB, GhostCSPC,
                     STCSPA, STCSPB, STCSPC,
                     ST2CSPA, ST2CSPB, ST2CSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif module_class is nn.BatchNorm2d:
            args = [n_input_channels[f]]
        elif module_class is Concat:
            n_output_channels = sum([n_input_channels[x] for x in f])
        elif module_class is Chuncat:
            n_output_channels = sum([n_input_channels[x] for x in f])
        elif module_class is Shortcut:
            n_output_channels = n_input_channels[f[0]]
        elif module_class is Foldcut:
            n_output_channels = n_input_channels[f] // 2

        elif module_class in [Detect, IDetect, IAuxDetect, IBin, IKeypoint]:
            args.append([n_input_channels[x] for x in f])

            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)

        elif module_class is ReOrg:
            n_output_channels = n_input_channels[f] * 4
        elif module_class is Contract:
            n_output_channels = n_input_channels[f] * args[0] ** 2
        elif module_class is Expand:
            n_output_channels = n_input_channels[f] // args[0] ** 2
        else:
            n_output_channels = n_input_channels[f]

        m_ = nn.Sequential(*[module_class(*args) for _ in range(n)]) if n > 1 else module_class(*args)  # module
        t = str(module_class)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            n_input_channels = []
        n_input_channels.append(n_output_channels)
    return nn.Sequential(*layers), sorted(save)


def parse_config(cfg: Union[Dict[str, object], str]) -> Dict[str, object]:
    cfg = deepcopy(cfg)
    if isinstance(cfg, dict):
        yaml_content = cfg  # model dict
    else:  # is *.yaml
        with open(cfg) as f:
            yaml_content = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
    return yaml_content

def update_cfg_with_model_parameters(cfg: Dict[str, object],
                                     n_channels: int,
                                     n_classes: int,
                                     anchors: List[List[int]]):
    cfg = deepcopy(cfg)
    # Define model
    cfg['ch'] = cfg.get('ch', n_channels)  # input channels
    if n_classes and n_classes != cfg['nc']:
        logger.info(f"Overriding model.yaml nc={cfg['nc']} with nc={n_classes}")
        cfg['nc'] = n_classes  # override yaml value
    if anchors:
        logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
        cfg['anchors'] = anchors  # override yaml value
    return cfg