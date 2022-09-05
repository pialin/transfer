import glob
import os.path
from argparse import Namespace

from train import main,  parse_opt

root_dir = "exps/fd_search"

for r_model_cfg in glob.iglob(os.path.join(root_dir, "models", "*.yaml")):
    r_model_name = os.path.splitext(os.path.basename(r_model_cfg))[0]

    # main(parse_opt(known=True, namespace=Namespace(
    #     weights=None,
    #     cfg=r_model_cfg,
    #     # data=os.path.join(root_dir, "data.yaml"),
    #     data="data/coco.yaml",
    #     hyp="data/hyps/hyp.scratch-low.yaml",
    #     epochs=1,
    #     batch_size=64,
    #     imgsz=320,
    #     device="cpu",
    #     workers=1,
    #     single_cls=True,
    #     name=r_model_name,
    # )))

    opt_dict = {
        "weights": None,
        "cfg": r_model_cfg,
        # "data": os.path.join(root_dir, "data.yaml"),
        "data": "data/coco.yaml",
        "hyp": "data/hyps/hyp.scratch-high.yaml",
        "epochs": 2,
        "batch_size": 8,
        "imgsz": 320,
        "device": "cpu",
        "single_cls": True,
        "workers": 2,
        "name": r_model_name,
    }

    arg_str = ""

    for r_key, r_value in opt_dict.items():
        arg_str += f" --{r_key.replace('_', '-')} {str(r_value) if r_value is not True else ''}"

    train_cmd = f"python train.py {arg_str}"
    ret_code = os.system(train_cmd)

    if ret_code != 0:
        print(f"{train_cmd} return error code {ret_code}!")
        break


    # main(parse_opt(known=True, namespace=Namespace(**opt_dict)))

# parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
# parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
# parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
# parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
# parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
# parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
# parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
# parser.add_argument('--rect', action='store_true', help='rectangular training')
# parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
# parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
# parser.add_argument('--noval', action='store_true', help='only validate final epoch')
# parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
# parser.add_argument('--noplots', action='store_true', help='save no plot files')
# parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
# parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
# parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
# parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
# parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
# parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
# parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
# parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
# parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
# parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
# parser.add_argument('--name', default='exp', help='save to project/name')
# parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
# parser.add_argument('--quad', action='store_true', help='quad dataloader')
# parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
# parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
# parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
# parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
# parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
# parser.add_argument('--seed', type=int, default=0, help='Global training seed')
# parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
#
# # Weights & Biases arguments
# parser.add_argument('--entity', default=None, help='W&B: Entity')
# parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
# parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
# parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')
