"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn 

from src.core import YAMLConfig


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    # The dummy input must match the config: pos-embed/anchors are precomputed for
    # eval_spatial_size [h, w], and RGB-D backbones take 4 input channels.
    h, w = cfg.yaml_cfg.get('eval_spatial_size', [640, 640])
    in_chans = next(m for m in model.model.backbone.modules() if isinstance(m, nn.Conv2d)).in_channels
    data = torch.rand(1, in_chans, h, w)
    size = torch.tensor([[w, h]])  # orig_target_sizes is [W, H]
    outputs = model(data, size)
    # Deploy-mode postprocessor returns 5 tensors when the model has pose heads, else 3.
    output_names = ['labels', 'boxes', 'scores', 'rotations', 'translations'][:len(outputs)]

    dynamic_axes = {
        'images': {0: 'N', },
        'orig_target_sizes': {0: 'N'}
    }

    torch.onnx.export(
        model, 
        (data, size), 
        args.output_file,
        input_names=['images', 'orig_target_sizes'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=16, 
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx 
        import onnxsim
        # test_input_shapes only feeds onnxsim's check; overwrite_input_shapes (what the
        # deprecated input_shapes maps to) would pin the dynamic batch axis N to 1.
        test_input_shapes = {'images': list(data.shape), 'orig_target_sizes': list(size.shape)}
        onnx_model_simplify, check = onnxsim.simplify(args.output_file, test_input_shapes=test_input_shapes)
        onnx.save(onnx_model_simplify, args.output_file)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--output_file', '-o', type=str, default='model.onnx')
    parser.add_argument('--check',  action='store_true', default=False,)
    parser.add_argument('--simplify',  action='store_true', default=False,)

    args = parser.parse_args()

    main(args)
