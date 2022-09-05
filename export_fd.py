import glob
import os.path
from argparse import Namespace

from export import main, parse_opt

root_dir = os.path.join("runs", "train")

for r_weight_path in glob.iglob(os.path.join(root_dir, "*", os.path.join("weights", "best.pt"))):

    r_model_name = r_weight_path.split(os.path.sep)[-3]

    for r_image_size in ("256 320", ): #(224, 288), (192, 240)):

        opt_dict = {
            "data": os.path.join("exps/fd_search/calibrate_data.yaml"),
            "weights": r_weight_path,
            "imgsz": r_image_size,
            "batch_size": 1,
            "device": 0,
            # "train": True,
            # "keras": False,
            "int8": True,
            "simplify": True,
            "include": "tflite",
            "tflite_name": r_model_name
        }

        arg_str = ""

        for r_key, r_value in opt_dict.items():
            arg_str += f" --{r_key.replace('_', '-')} {str(r_value) if r_value is not True else ''}"

        export_cmd = f"python export.py {arg_str}"
        ret_code = os.system(export_cmd)

        if ret_code != 0:
            print(f"{export_cmd} return error code {ret_code}!")
            continue


        # main(parse_opt(namespace=Namespace(
        #     data=os.path.join("exps/fd_search/calibrate_data.yaml"),
        #     weights=r_weight_path,
        #     imgsz=r_image_size,
        #     batch_size=1,
        #     device="cpu",
        #     train=True,
        #     keras=True,
        #     int8=True,
        #     simplify=True,
        #     include=["tflite"],
        #     tflite_name=r_model_name
        # )))

# parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
# parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
# parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
# parser.add_argument('--batch-size', type=int, default=1, help='batch size')
# parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
# parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
# parser.add_argument('--train', action='store_true', help='model.train() mode')
# parser.add_argument('--keras', action='store_true', help='TF: use Keras')
# parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
# parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
# parser.add_argument('--dynamic', action='store_true', help='ONNX/TF/TensorRT: dynamic axes')
# parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
# parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
# parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log')
# parser.add_argument('--workspace', type=int, default=4, help='TensorRT: workspace size (GB)')
# parser.add_argument('--nms', action='store_true', help='TF: add NMS to model')
# parser.add_argument('--agnostic-nms', action='store_true', help='TF: add agnostic NMS to model')
# parser.add_argument('--topk-per-class', type=int, default=100, help='TF.js NMS: topk per class to keep')
# parser.add_argument('--topk-all', type=int, default=100, help='TF.js NMS: topk for all classes to keep')
# parser.add_argument('--iou-thres', type=float, default=0.45, help='TF.js NMS: IoU threshold')
# parser.add_argument('--conf-thres', type=float, default=0.25, help='TF.js NMS: confidence threshold')
# parser.add_argument('--include',
#                     nargs='+',
#                     default=['torchscript'],
#                     help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs')
