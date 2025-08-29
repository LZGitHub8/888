import os
import sys
import mmcv
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette

#sys.path.insert(0, 'E:/semantic segmentation/PVT3-new design/PVT-2/segmentation')

#import pvt
#from align_resize import AlignResize

def main():
    config = 'E://semantic segmentation//sam2-main//sam2//configs//sam2.1_training//sam2.1_hiera_b+_MOSE_finetune.yaml'
    checkpoint = 'E:/semantic segmentation/sam2-main/training/sam2_logs/sam2.1_hiera_b+_MOSE_finetune/checkpoints/checkpoint.pt'
    img_path = 'E:/semantic segmentation/sam2-main/demo/00037.jpg'
    # 初始化前打印配置检查
    from mmcv import Config
    cfg = Config.fromfile(config)
    print(cfg.pretty_text)  # 确认参数结构正确

    # 初始化模型
    model = init_segmentor(config, checkpoint, device='cuda:0')

    # 推理并保存结果
    result = inference_segmentor(model, img_path)
    model.show_result(
        img_path,
        result,
        palette=get_palette('ade20k'),
        out_file='output.png',
        opacity=0.5
    )


if __name__ == '__main__':
    main()