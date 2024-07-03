import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
import matplotlib.pyplot as plt

def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    fin_str = "img_auc,"+run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc,"+run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap,"+run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap,"+run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./outputs/results.txt",'a+') as file:
        file.write(fin_str)


def save_visualization(images, gt_images, out_masks, in_masks, output_dir, idx):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 원본 이미지
    axes[0].imshow(images.permute(1, 2, 0).cpu().detach().numpy())
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Ground Truth 이미지
    axes[1].imshow(gt_images.permute(1, 2, 0).cpu().detach().numpy())
    axes[1].set_title('Ground Truth Image')
    axes[1].axis('off')

    # 출력 마스크
    axes[2].imshow(out_masks.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[2].set_title('Output Mask')
    axes[2].axis('off')

    # 입력 마스크
    axes[3].imshow(in_masks.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[3].set_title('Input Mask')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'image_{idx:03d}.png'))
    plt.close()

    

def test(obj_names, mvtec_path, checkpoint_path, base_model_name):
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    
    for obj_name in obj_names:
        img_dim = 256
        run_name = base_model_name+"_"+obj_name+'_'

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path,run_name+".pckl"), map_location='cuda:0'))
        model.cuda()
        model.eval()

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+"_seg.pckl"), map_location='cuda:0'))
        model_seg.cuda()
        model_seg.eval()

        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        output_dir = './outputs/' + obj_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i_batch, sample_batched in enumerate(dataloader):
            gray_batch = sample_batched["image"].cuda()
            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            gray_rec = model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            t_mask = out_mask_sm[:, 1:, :, :]
            save_visualization(gray_rec[0], gray_batch[0], t_mask[0], true_mask[0], output_dir, i_batch)

            out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()
            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1, padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)

            anomaly_score_prediction.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1

            print(mask_cnt)

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)
        print(obj_name)
        print("AUC Image:  " +str(auroc))
        print("AP Image:  " +str(ap))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("AP Pixel:  " +str(ap_pixel))
        print("==============================")

    print(run_name)
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
    print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))

    write_results_to_file(run_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=False, default=0)
    parser.add_argument('--base_model_name', action='store', type=str, required=False, default="DRAEM_seg_large_ae_large_0.0001_800_bs8")
    parser.add_argument('--data_path', action='store', type=str, required=False, default="../datasets\\MVTec\\")
    parser.add_argument('--checkpoint_path', action='store', type=str, required=False, default="checkpoints\\")

    args = parser.parse_args()

    obj_list = ['bottle',
                # 'capsule',
                #  'carpet',
                #  'leather',
                #  'pill',
                #  'transistor',
                #  'tile',
                #  'cable',
                #  'zipper',
                #  'toothbrush',
                #  'metal_nut',
                #  'hazelnut',
                #  'screw',
                #  'grid',
                #  'wood'
                 ]

    with torch.cuda.device(args.gpu_id):
        test(obj_list,args.data_path, args.checkpoint_path, args.base_model_name)
