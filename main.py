import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from utils import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
    display_images
)

def main():
    """Main function to run the image denoising pipeline."""
    
    # --- 1. Setup Paths ---
    IMAGE_NAME = "WIN_20250412_01_44_59_Pro.jpg"  # User should replace this
    INPUT_PATH = os.path.join("images", IMAGE_NAME)
    OUTPUT_DIR = "output"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Project structure is ready.")
    print("Please add an image to the 'images' directory and update IMAGE_NAME.")

    # --- 2. Load Image ---
    original_image = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print(f"Error: Could not load image from {INPUT_PATH}")
        print("Please make sure the image exists and the path is correct.")
        return

    print("Successfully loaded original image.")

    # --- 3. Add Noise ---
    print("Adding noise to the image...")
    noisy_gaussian = add_gaussian_noise(original_image, sigma=25)
    noisy_salt_pepper = add_salt_and_pepper_noise(original_image, amount=0.05)

    # Save the noisy images
    gaussian_path = os.path.join(OUTPUT_DIR, "noisy_gaussian.jpg")
    salt_pepper_path = os.path.join(OUTPUT_DIR, "noisy_salt_pepper.jpg")
    cv2.imwrite(gaussian_path, noisy_gaussian)
    cv2.imwrite(salt_pepper_path, noisy_salt_pepper)
    print(f"Saved Gaussian noisy image to: {gaussian_path}")
    print(f"Saved Salt & Pepper noisy image to: {salt_pepper_path}")

    # --- 4. Apply Filters ---
    print("Applying filters to noisy images...")

    # Store all denoised results for later evaluation
    denoised_results = {}

    # 4.1 Median Filter
    denoised_median_on_gaussian = cv2.medianBlur(noisy_gaussian, 5)
    denoised_median_on_salt_pepper = cv2.medianBlur(noisy_salt_pepper, 5)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "denoised_median_on_gaussian.jpg"), denoised_median_on_gaussian)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "denoised_median_on_salt_pepper.jpg"), denoised_median_on_salt_pepper)
    denoised_results['median_on_gaussian'] = denoised_median_on_gaussian
    denoised_results['median_on_salt_pepper'] = denoised_median_on_salt_pepper

    # 4.2 Gaussian Filter
    denoised_gaussian_on_gaussian = cv2.GaussianBlur(noisy_gaussian, (5, 5), 0)
    denoised_gaussian_on_salt_pepper = cv2.GaussianBlur(noisy_salt_pepper, (5, 5), 0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "denoised_gaussian_on_gaussian.jpg"), denoised_gaussian_on_gaussian)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "denoised_gaussian_on_salt_pepper.jpg"), denoised_gaussian_on_salt_pepper)
    denoised_results['gaussian_on_gaussian'] = denoised_gaussian_on_gaussian
    denoised_results['gaussian_on_salt_pepper'] = denoised_gaussian_on_salt_pepper

    # 4.3 Custom Averaging Kernel Filter
    averaging_kernel = np.ones((5, 5), np.float32) / 25
    denoised_kernel_on_gaussian = cv2.filter2D(noisy_gaussian, -1, averaging_kernel)
    denoised_kernel_on_salt_pepper = cv2.filter2D(noisy_salt_pepper, -1, averaging_kernel)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "denoised_kernel_on_gaussian.jpg"), denoised_kernel_on_gaussian)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "denoised_kernel_on_salt_pepper.jpg"), denoised_kernel_on_salt_pepper)
    denoised_results['kernel_on_gaussian'] = denoised_kernel_on_gaussian
    denoised_results['kernel_on_salt_pepper'] = denoised_kernel_on_salt_pepper

    # 4.4 Bilateral Filter
    denoised_bilateral_on_gaussian = cv2.bilateralFilter(noisy_gaussian, 9, 75, 75)
    denoised_bilateral_on_salt_pepper = cv2.bilateralFilter(noisy_salt_pepper, 9, 75, 75)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "denoised_bilateral_on_gaussian.jpg"), denoised_bilateral_on_gaussian)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "denoised_bilateral_on_salt_pepper.jpg"), denoised_bilateral_on_salt_pepper)
    denoised_results['bilateral_on_gaussian'] = denoised_bilateral_on_gaussian
    denoised_results['bilateral_on_salt_pepper'] = denoised_bilateral_on_salt_pepper

    print("All filters applied and results saved to the 'output' directory.")

    # --- 5. Evaluate and Compare ---
    print("Evaluating denoised images (PSNR & SSIM) and creating comparison plot...")

    # Calculate metrics for each denoised image compared to original
    metrics = {}
    for key, denoised_img in denoised_results.items():
        psnr_score = calculate_psnr(original_image, denoised_img)
        ssim_score = calculate_ssim(original_image, denoised_img)
        metrics[key] = {"psnr": psnr_score, "ssim": ssim_score}

    # Pretty names for table
    filt_map = {
        "median": "Median",
        "gaussian": "Gaussian",
        "kernel": "Averaging Kernel",
        "bilateral": "Bilateral",
    }
    noise_map = {
        "gaussian": "Gaussian",
        "salt_pepper": "Salt & Pepper",
    }

    # Print formatted table
    print("\nFilter Results (compared to original):")
    print(f"{'Filter':<24}{'Noise':<16}{'PSNR (dB)':>12}{'SSIM':>10}")
    print("-" * 64)
    for key, vals in metrics.items():
        try:
            f, n = key.split("_on_")
        except ValueError:
            f, n = key, ""
        f_name = filt_map.get(f, f)
        n_name = noise_map.get(n, n)
        print(f"{f_name:<24}{n_name:<16}{vals['psnr']:>12.2f}{vals['ssim']:>10.4f}")

    # Create visualization plot (3x5 grid)
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    # Row 1: Original, Noisy Gaussian, Noisy Salt & Pepper
    axes[0, 0].imshow(original_image, cmap='gray'); axes[0, 0].set_title("Original"); axes[0, 0].axis('off')
    axes[0, 1].imshow(noisy_gaussian, cmap='gray'); axes[0, 1].set_title("Noisy Gaussian"); axes[0, 1].axis('off')
    axes[0, 2].imshow(noisy_salt_pepper, cmap='gray'); axes[0, 2].set_title("Noisy Salt & Pepper"); axes[0, 2].axis('off')
    for c in range(3, 5):
        axes[0, c].axis('off')

    # Row 2: Filters on Gaussian noise
    row2_items = [
        ("median_on_gaussian", "Median on Gaussian"),
        ("gaussian_on_gaussian", "Gaussian on Gaussian"),
        ("kernel_on_gaussian", "Kernel on Gaussian"),
        ("bilateral_on_gaussian", "Bilateral on Gaussian"),
    ]
    for idx, (key, title) in enumerate(row2_items):
        img = denoised_results[key]
        psnr_val = metrics[key]["psnr"]
        axes[1, idx].imshow(img, cmap='gray')
        axes[1, idx].set_title(f"{title}\nPSNR: {psnr_val:.2f} dB")
        axes[1, idx].axis('off')
    for c in range(len(row2_items), 5):
        axes[1, c].axis('off')

    # Row 3: Filters on Salt & Pepper noise
    row3_items = [
        ("median_on_salt_pepper", "Median on Salt & Pepper"),
        ("gaussian_on_salt_pepper", "Gaussian on Salt & Pepper"),
        ("kernel_on_salt_pepper", "Kernel on Salt & Pepper"),
        ("bilateral_on_salt_pepper", "Bilateral on Salt & Pepper"),
    ]
    for idx, (key, title) in enumerate(row3_items):
        img = denoised_results[key]
        psnr_val = metrics[key]["psnr"]
        axes[2, idx].imshow(img, cmap='gray')
        axes[2, idx].set_title(f"{title}\nPSNR: {psnr_val:.2f} dB")
        axes[2, idx].axis('off')
    for c in range(len(row3_items), 5):
        axes[2, c].axis('off')

    plt.tight_layout()
    comparison_path = os.path.join(OUTPUT_DIR, "comparison_results.png")
    plt.savefig(comparison_path, dpi=150)
    print(f"Saved comparison figure to: {comparison_path}")

    # --- 6. Display Results (TODO for next step) ---
    

if __name__ == "__main__":
    main()