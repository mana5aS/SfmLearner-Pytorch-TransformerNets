import torch

from imageio import imread, imsave
from skimage.transform import resize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import os

from utils import tensor2array
from models import PoseExpNet
from inverse_warp import pose_vec2mat
import h5py
from scipy.spatial.transform import Rotation

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
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return ATE/snippet_length, RE/snippet_length

@torch.no_grad()
def main():
    args = parser.parse_args()
    weights = torch.load(args.pretrained)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
    # print ("seq length: ", seq_length)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    
    dataset_dir = Path(args.dataset_dir)

    f = h5py.File(filename, "r")
    traj_id = dataset_dir.split('/')[-2].strip()
    print ("traj: ", traj_id)
    quats = f[traj_id]['groundtruth']['attitude'][()]
    trans = f[traj_id]['groundtruth']['position'][()]
    ang_vel = f[traj_id]['groundtruth']['angular_velocity'][()]
    imu_angles = f[traj_id]['imu']['gyroscope'][()]
    gps_pos = f[traj_id]['gps']['position'][()]
    f.close()

    print (imu_angles.shape, quats.shape, ang_vel.shape)
    print (imu_angles[0]*180/np.pi, Rotation.from_quat(quats[0]).as_euler('xyz', degrees=True), ang_vel[0])
    print (imu_angles[1]*180/np.pi, Rotation.from_quat(quats[1]).as_euler('xyz', degrees=True), ang_vel[1])
    print (imu_angles[2]*180/np.pi, Rotation.from_quat(quats[2]).as_euler('xyz', degrees=True), ang_vel[2])
    print (imu_angles[3]*180/np.pi, Rotation.from_quat(quats[3]).as_euler('xyz', degrees=True), ang_vel[3])

    print (imu_angles[1]*180/np.pi - imu_angles[0]*180/np.pi, Rotation.from_quat(quats[1]).as_euler('zyx', degrees=True) - Rotation.from_quat(quats[0]).as_euler('ZYX', degrees=True))

    xxx
    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sum([list(dataset_dir.walkfiles('*.{}'.format(ext))) for ext in args.img_exts], [])
    print('{} files to test'.format(len(test_files)))

    sorted_test_files = sorted(test_files, key=lambda i: int(i.stem))

    idx = 0
    errors = np.zeros((len(sorted_test_files[0:-2]), 2))
    for ref1, target, ref2 in zip(sorted_test_files[0:-2], sorted_test_files[1:-1], sorted_test_files[2:]):
        ref1_img = imread(ref1)
        ref2_img = imread(ref2)
        tgt_img = imread(target)
        
        h,w,_ = tgt_img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            tgt_img = resize(tgt_img, (args.img_height, args.img_width))

        ref1_img = np.transpose(ref1_img, (2, 0, 1))
        ref2_img = np.transpose(ref2_img, (2, 0, 1))
        tgt_img = np.transpose(tgt_img, (2, 0, 1))

        ref1_tensor_img = torch.from_numpy(ref1_img.astype(np.float32)).unsqueeze(0)
        ref1_tensor_img = ((ref1_tensor_img/255.0 - 0.5)/0.5).to(device)

        ref2_tensor_img = torch.from_numpy(ref2_img.astype(np.float32)).unsqueeze(0)
        ref2_tensor_img = ((ref2_tensor_img/255.0 - 0.5)/0.5).to(device)

        tgt_tensor_img = torch.from_numpy(tgt_img.astype(np.float32)).unsqueeze(0)
        tgt_tensor_img = ((tgt_tensor_img/255.0 - 0.5)/0.5).to(device)

        # print (tgt_tensor_img.shape, ref1_tensor_img.shape, ref2_tensor_img.shape)
        _, poses = pose_net(tgt_tensor_img, [ref1_tensor_img, ref2_tensor_img])
        poses = poses.detach().cpu()[0]
        # print (poses.shape)

        poses = torch.cat([poses[:1], torch.zeros(1,6).float(), poses[1:]])
        inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)

        rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        # print (rot_matrices.shape)
        r = Rotation.from_matrix(rot_matrices)
        # print ("first pred rot angles: ", r.as_euler('zyx', degrees=True))

        tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:,:3] @ transform_matrices
        final_poses[:,:,-1:] += first_inv_transform[:,-1:]
        # print (final_poses, final_poses.shape)

        r = Rotation.from_matrix(final_poses[1, :, :3])
        # print ("pred rot angles f: ", r.as_euler('zyx', degrees=True))

        r = Rotation.from_matrix(final_poses[2, :, :3] @ np.linalg.inv(final_poses[1, :, :3]))
        # print ("pred rot angles s: ", r.as_euler('zyx', degrees=True))

        r = Rotation.from_quat(quats[4*idx])
        R1 = r.as_matrix()
        r = Rotation.from_quat(quats[4*idx+4])
        R2 = r.as_matrix()
        R_rel1 = R2 @ np.linalg.inv(R1)
        r = Rotation.from_matrix(R_rel1)
        # print ("gt rot angles 1: ", r.as_euler('zyx', degrees=True))

        r = Rotation.from_quat(quats[4*idx+4])
        R1 = r.as_matrix()
        r = Rotation.from_quat(quats[4*idx+8])
        R2 = r.as_matrix()
        R_rel2 = R2 @ np.linalg.inv(R1)
        r = Rotation.from_matrix(R_rel2)
        # print ("gt rot angles 2: ", r.as_euler('zyx', degrees=True))

        final_poses_gt = np.zeros_like(final_poses)
        final_poses_gt[0, :, :3] = np.eye(3)
        final_poses_gt[1, :, :3] = R_rel1
        final_poses_gt[2, :, :3] = R_rel2
        final_poses_gt[1, :, -1] = trans[4*idx+4]-trans[4*idx]
        final_poses_gt[2, :, -1] = trans[4*idx+8]-trans[4*idx+4]
     
        ATE, RE  = compute_pose_error(final_poses_gt, final_poses)
        # print (ATE, RE)

        errors[idx] = ATE, RE 
        idx+=1

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    print ("overall mean errors (ATE, RE): ", mean_errors, " std devs: ", std_errors)

if __name__ == '__main__':
    filename = "/home/jupyter/MidAir/Kite_training/sunny/sensor_records.hdf5"
    np.set_printoptions(precision=5, suppress=True)
    main()
