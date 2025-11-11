import numpy as np
import cv2
from typing import List, Tuple
from skimage.metrics import structural_similarity

def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
    """Adds Gaussian noise to a grayscale image."""
    # Ensure input is a numpy array with expected dtype
    image_float: np.ndarray = image.astype(np.float32)

    # Generate Gaussian noise with specified mean and sigma
    noise: np.ndarray = np.random.normal(loc=mean, scale=sigma, size=image.shape).astype(np.float32)

    # Add noise to the image
    noisy_image: np.ndarray = image_float + noise

    # Clip to valid pixel range and convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

def add_salt_and_pepper_noise(image: np.ndarray, amount: float = 0.05) -> np.ndarray:
    """Adds Salt and Pepper noise to a grayscale image."""
    # Create a copy to avoid modifying the original image
    noisy: np.ndarray = image.copy()

    # Guard against invalid amount values
    amount = max(0.0, float(amount))
    if amount == 0.0:
        return noisy

    # Determine number of pixels to alter
    total_pixels: int = noisy.size
    num_salt: int = int(np.ceil(amount * total_pixels / 2))
    num_pepper: int = int(np.floor(amount * total_pixels / 2))

    # Image dimensions (assumes grayscale 2D). If color, apply on entire array flatten-wise.
    if noisy.ndim == 2:
        h, w = noisy.shape
        # Salt (set to white: 255)
        coords_salt = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt))
        noisy[coords_salt] = 255

        # Pepper (set to black: 0)
        coords_pepper = (np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper))
        noisy[coords_pepper] = 0
    else:
        # For non-grayscale, operate on flattened indices and map back
        flat_noisy = noisy.reshape(-1)
        idx_salt = np.random.randint(0, total_pixels, num_salt)
        idx_pepper = np.random.randint(0, total_pixels, num_pepper)
        flat_noisy[idx_salt] = 255
        flat_noisy[idx_pepper] = 0
        noisy = flat_noisy.reshape(noisy.shape)

    return noisy

def calculate_mse(original_image: np.ndarray, processed_image: np.ndarray) -> float:
    """Calculates the Mean Squared Error (MSE) between two images."""
    # Convert to float64 to avoid overflow/underflow during subtraction
    a = original_image.astype(np.float64)
    b = processed_image.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float(mse)

def calculate_psnr(original_image: np.ndarray, processed_image: np.ndarray, max_pixel: int = 255) -> float:
    """Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Uses the formula:
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    where MAX_I is the maximum possible pixel value of the image (default 255).
    """
    mse = calculate_mse(original_image, processed_image)
    if mse == 0:
        # Images are identical; PSNR is infinite
        return float('inf')
    psnr = 20 * np.log10(float(max_pixel)) - 10 * np.log10(mse)
    return float(psnr)

def calculate_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
    """Calculates Structural Similarity Index (SSIM) between two images.

    SSIM considers luminance, contrast, and structure. Returns a value in [0, 1].
    """
    score, _ = structural_similarity(original, compressed, data_range=255, full=True)
    return float(score)

def display_images(images: List[np.ndarray], titles: List[str], main_title: str) -> None:
    """Displays a list of images in a single window using matplotlib."""
    pass