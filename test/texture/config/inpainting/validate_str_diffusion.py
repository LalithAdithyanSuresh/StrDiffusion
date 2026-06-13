import os
import sys

# Set custom cache/compile directories to avoid permission issues in /home/cks
os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/cks/.cache/torch_extensions"
os.environ["MPLCONFIGDIR"] = "/tmp/cks/.cache/matplotlib"
os.environ["HF_HOME"] = "/tmp/cks/.cache/huggingface"

import argparse
import random
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add script directory to sys.path to allow importing local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import options as option
from models import create_model
import str_utils as str_util
import utils as util

def calc_psnr(gt, pre):
    # gt and pre are numpy arrays of shape (H, W, 3) in range 0-255
    mse = np.mean((gt.astype(np.float64) - pre.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def generate_comparison_grid(gt_np, mask_np, pred_resized_np):
    # gt_np: (H, W, 3) RGB numpy array
    # mask_np: (H, W, 1) float numpy array (range 0.0 to 1.0, where 1.0 is masked)
    # pred_resized_np: (H, W, 3) RGB numpy array (blended result)
    
    # 1. Ground Truth (GT)
    img_gt = gt_np.copy()
    
    # 2. Mask + GT (GT with a semi-transparent red overlay on the mask)
    bool_mask = mask_np[:, :, 0] > 0.5
    overlay = gt_np.copy()
    overlay[bool_mask] = [255, 0, 0]
    img_mask_gt = (0.7 * gt_np + 0.3 * overlay).clip(0, 255).astype(np.uint8)
    
    # 3. Prediction (blended inpainted result)
    img_pred = pred_resized_np.copy()
    
    # 4. Diff (Red-Green error map over grayscale GT)
    # Pure numpy luma formula to convert RGB to grayscale
    gray_gt = (0.299 * gt_np[:, :, 0] + 0.587 * gt_np[:, :, 1] + 0.114 * gt_np[:, :, 2]).astype(np.uint8)
    gray_gt_3ch = np.stack([gray_gt, gray_gt, gray_gt], axis=-1)
    
    # Absolute difference error across RGB channels
    err = np.mean(np.abs(gt_np.astype(np.float64) - pred_resized_np.astype(np.float64)), axis=-1)
    # Scale error to make differences more visible
    scaled_err = np.clip(err * 3.0, 0, 255)
    
    # Red-Green colormap: pure red for high error, pure green for zero error
    diff_color = np.zeros_like(gt_np)
    diff_color[:, :, 0] = scaled_err # Red
    diff_color[:, :, 1] = 255.0 - scaled_err # Green
    diff_color[:, :, 2] = 0 # Blue
    
    # Blend structure (40%) with error color (60%)
    img_diff = (0.4 * gray_gt_3ch + 0.6 * diff_color).clip(0, 255).astype(np.uint8)
    
    # Concatenate horizontally
    grid = np.concatenate([img_gt, img_mask_gt, img_pred, img_diff], axis=1)
    return grid

def image_to_edge(image, sigma=2.0):
    from skimage.feature import canny
    from skimage.color import rgb2gray
    
    # Convert tensor (3, H, W) to PIL and then grayscale numpy
    to_pil = transforms.ToPILImage()
    pil_img = to_pil(image)
    gray_image = rgb2gray(np.array(pil_img))
    
    # Get edge map and gray image as tensors
    to_tensor = transforms.ToTensor()
    edge = to_tensor(Image.fromarray(canny(gray_image, sigma=sigma)))
    gray_image = to_tensor(Image.fromarray(gray_image))
    
    return edge, gray_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/tmp/cks/SEM-Net/datasets/places365/test_256', help='Directory with source images')
    parser.add_argument('--mask_dir', type=str, default='/tmp/cks/SEM-Net/datasets/testing_mask_dataset', help='Directory with masks')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--opt', type=str, default=os.path.join(script_dir, 'options/test/ir-sde.yml'), help='Path to option YMAL file')
    parser.add_argument('--texture_ckpt', type=str, default='checkpoint/t.pth', help='Path to texture denoising model')
    parser.add_argument('--structure_ckpt', type=str, default='checkpoint/s.pth', help='Path to structure denoising model')
    parser.add_argument('--discriminator_ckpt', type=str, default='checkpoint/dis.pth', help='Path to discriminator model')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()

    # Load options YMAL config
    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    # Override checkpoint paths and GPU IDs programmatically
    opt['gpu_ids'] = [args.gpu_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Using GPU Device: {args.gpu_id}")

    opt['path']['pretrain_model_G'] = args.texture_ckpt
    opt['path']['pretrain_model_Gs'] = args.structure_ckpt
    opt['path']['pretrain_model_D'] = args.discriminator_ckpt

    # Set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Create model
    print("[*] Creating StrDiffusion model...")
    model = create_model(opt)
    device = model.device

    # Initialize SDE runners
    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)
       
    S_sde = str_util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    S_sde.set_model(model.models)

    # Targets to evaluate
    targets = [
        # (category, image_id, mask_id)
        ('LARGE', 'Places365_test_00032703', '08398'),
        ('LARGE', 'Places365_test_00081016', '08986'),
        ('LARGE', 'Places365_test_00014790', '08180'),
        ('LARGE', 'Places365_test_00217740', '10650'),
        ('MEDIUM', 'Places365_test_00018734', '04228'),
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    for cat in ['LARGE', 'MEDIUM']:
        os.makedirs(os.path.join(args.output_dir, cat), exist_ok=True)

    print("\n--- Starting StrDiffusion Validation on 5 Target Images ---")
    results = []

    for cat, img_id, mask_id in targets:
        print(f"\nProcessing {img_id} with mask {mask_id} ({cat})...")

        # Locate source image
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            p = os.path.join(args.image_dir, f"{img_id}{ext}")
            if os.path.exists(p):
                img_path = p
                break
        
        if not img_path:
            print(f"Error: Source image {img_id} not found in {args.image_dir}. Skipping.")
            continue

        # Locate mask
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            p = os.path.join(args.mask_dir, f"{mask_id}{ext}")
            if os.path.exists(p):
                mask_path = p
                break

        if not mask_path:
            print(f"Error: Mask image {mask_id} not found in {args.mask_dir}. Skipping.")
            continue

        # Load and preprocess image
        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size

        # Resize original image to 256x256
        img_256 = pil_img.resize((256, 256), Image.Resampling.BILINEAR)
        img_np = np.array(img_256).astype(np.float32) / 255.0
        
        # RGB representation for internal model usage (HWC RGB)
        img_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img_np, (2, 0, 1)))).float()

        # Generate edge and gray maps using Canny
        GT_edge, GT_gray = image_to_edge(img_tensor, sigma=2.0)

        # Batch inputs and move to device
        Y_GT = img_tensor.unsqueeze(0).to(device)
        X_GT = GT_gray.unsqueeze(0).to(device)
        X_LQ = GT_edge.unsqueeze(0).to(device)

        # Load mask: 0.0 is masked (missing region), 1.0 is unmasked (background)
        # Convert to grayscale first and threshold to avoid PIL conversion/dithering artifacts
        mask_pil = Image.open(mask_path).convert('L').resize((256, 256), Image.Resampling.NEAREST)
        mask_np_bin = (np.array(mask_pil) > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np_bin).unsqueeze(0).unsqueeze(0).to(device)
        mask_tensor = 1.0 - mask_tensor # Invert so 0 = masked, 1 = unmasked

        # Prepare SDE states
        noisy_state = sde.noise_state(Y_GT * mask_tensor)
        noisy_states = S_sde.noise_state(X_LQ * mask_tensor)

        model.feed_data(noisy_state, Y_GT * mask_tensor, Y_GT, mask_tensor, S_sde, X_GT, X_LQ * mask_tensor)

        print("[*] Running texture-guided structure reverse SDE inference...")
        tic = time.time()
        model.test(sde, save_states=False, save_dir=None, GT=Y_GT, mask=mask_tensor, S_sde=S_sde, S_GT=X_GT, S_LQ=noisy_states, dis=model.dis)
        toc = time.time()
        print(f"Inference completed in {toc - tic:.2f} seconds.")

        # Postprocess output
        visuals = model.get_current_visuals()
        SR_img = visuals["Output"] # Output tensor (3, 256, 256)
        
        # Convert output to RGB numpy array
        output_bgr = util.tensor2img(SR_img.squeeze()) # returns BGR HWC uint8
        output_rgb = output_bgr[:, :, [2, 1, 0]]

        # Resize output prediction back to original image size
        pred_pil = Image.fromarray(output_rgb).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        pred_np = np.array(pred_pil)

        # Merge with ground truth using the original mask
        gt_np = np.array(pil_img)
        # Load mask in original resolution: 1.0 is masked, 0.0 is unmasked
        # Convert to grayscale first and threshold to avoid PIL conversion/dithering artifacts
        mask_pil_orig = Image.open(mask_path).convert('L').resize((orig_w, orig_h), Image.Resampling.NEAREST)
        mask_np = (np.array(mask_pil_orig) > 127).astype(np.float32)
        mask_np = mask_np[:, :, None] # Shape (H, W, 1)

        # Blend: pred inside mask, gt outside mask
        merged_np = (pred_np * mask_np + gt_np * (1.0 - mask_np)).clip(0, 255).astype(np.uint8)

        # Calculate PSNR
        psnr = calc_psnr(gt_np, merged_np)
        print(f"Blended image generated. PSNR: {psnr:.2f}")

        # Save output
        save_name = f"{img_id}_{mask_id}_{psnr:.2f}.png"
        save_path = os.path.join(args.output_dir, cat, save_name)
        Image.fromarray(merged_np).save(save_path)
        print(f"Saved output to: {save_path}")

        # Save comparison grid (GT, Mask+GT, Pred, Diff)
        grid_np = generate_comparison_grid(gt_np, mask_np, merged_np)
        grid_name = f"{img_id}_{mask_id}_comparison.png"
        grid_path = os.path.join(args.output_dir, cat, grid_name)
        Image.fromarray(grid_np).save(grid_path)
        print(f"Saved comparison grid to: {grid_path}")

        results.append({
            'Category': cat,
            'Image ID': img_id,
            'Mask ID': mask_id,
            'PSNR': f"{psnr:.2f}"
        })

    print("\n" + "="*60)
    print("STRDIFFUSION MODEL EVALUATION SUMMARY")
    print("="*60)
    for r in results:
        print(f"[{r['Category']}] Image: {r['Image ID']} | Mask: {r['Mask ID']} | PSNR: {r['PSNR']}")
    print("="*60)

if __name__ == '__main__':
    main()
