import torch

from imageio import imread, imsave
from skimage.transform import resize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from models import DispNetS
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--gt-dataset-dir", default='.', type=str, help="GT Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'JPEG', 'PNG', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument('--disp-transformer', action='store_true', help='use DPT Transformer network for depthnet')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_log = np.mean(np.abs(np.log(gt) - np.log(pred)))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    abs_diff = np.mean(np.abs(gt - pred))

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return np.array([abs_diff, abs_rel, sq_rel, rmse, rmse_log, abs_log, a1, a2, a3])

@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return
    if args.disp_transformer:
        print ("Using Transformer DPT for depth net")
        from dpt.models import DPTDepthModel
        disp_net = DPTDepthModel(
            path=args.pretrained,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        ).to(device)
    else:
        disp_net = DispNetS().to(device)
    # weights = torch.load(args.pretrained)
    # disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    gt_dataset_dir = Path(args.gt_dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sum([list(dataset_dir.walkfiles('*.{}'.format(ext))) for ext in args.img_exts], [])
        gt_test_files = sum([list(gt_dataset_dir.walkfiles('*.{}'.format(ext))) for ext in args.img_exts], [])
  

    test_files = sorted(test_files, key=lambda i: int(i.stem))
    gt_test_files = sorted(gt_test_files, key=lambda i: int(i.stem))

    print('{} files to test'.format(len(test_files)))
    print ("gt test files: ", len(gt_test_files))

    results = np.zeros(9)
    for file, gt_file in tqdm(zip(test_files, gt_test_files)):
        img = imread(file)
        gt_img = imread(gt_file)
        
        h,w,_ = img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = resize(img, (args.img_height, args.img_width))
        img = np.transpose(img, (2, 0, 1))

        tensor_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        tensor_img = ((tensor_img - 0.5)/0.5).to(device)

        if args.disp_transformer:
            depth = disp_net(tensor_img)
            output = 1/depth
            print ("transformer depth: ", torch.max(depth), torch.min(depth))
        else:
            output = disp_net(tensor_img)[0]
            depth = 100/output
            print ("cnn: depth: ", torch.max(depth), torch.min(depth))

        scale_factor = np.median(gt_img)/np.median(depth.detach().cpu().numpy())
        # print ("scale factor: ", scale_factor)
        err = compute_errors(gt_img/scale_factor, depth.detach().cpu().numpy())
        # print ("curr err: ", err)
        results = results + err

        file_path, file_ext = file.relpath(args.dataset_dir).splitext()
        file_name = '-'.join(file_path.splitall()[1:])

        new_gt = gt_img/np.max(gt_img)
        imsave(output_dir/'{}_gt{}'.format(file_name, file_ext), new_gt*255)
        from skimage import color

        if args.output_disp:
            # disp = (255-255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
            disp = output.squeeze().detach().cpu().numpy()
            disp = disp/np.max(disp)
            disp = 255 - disp * 255
            disp = disp.astype(np.uint8)
            import cv2
            disp = cv2.equalizeHist(disp)
            # disp = color.rgb2gray(np.transpose(disp, (1,2,0)))
            imsave(output_dir/'{}_disp{}'.format(file_name, file_ext), disp)
        if args.output_depth:
            # depth = 1/output
            # depth = (255*tensor2array(depth, max_value=None, colormap='bone')).astype(np.uint8)
            depth = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
            imsave(output_dir/'{}_depth{}'.format(file_name, file_ext), np.transpose(depth, (1,2,0)))
            
    print ("mean err (abs_diff, abs_rel, sq_rel, rmse, rmse_log, abs_log, a1, a2, a3): ", results/len(gt_test_files))


if __name__ == '__main__':
    main()
