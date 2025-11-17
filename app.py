import os
import cv2
import numpy as np
import streamlit as st
from utils import add_gaussian_noise, add_salt_and_pepper_noise, calculate_psnr, calculate_ssim

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def list_images(dir_path: str):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(dir_path):
        return []
    return [f for f in os.listdir(dir_path) if f.lower().endswith(exts)]

def load_image_from_path(path: str):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def run_pipeline(original_image: np.ndarray, params: dict):
    noisy_gaussian = add_gaussian_noise(original_image, sigma=params["gauss_sigma"]) 
    noisy_salt_pepper = add_salt_and_pepper_noise(original_image, amount=params["sp_amount"]) 
    denoised_results = {}
    k_med = params["median_ksize"] if params["median_ksize"] % 2 == 1 else params["median_ksize"] + 1
    denoised_results["median_on_gaussian"] = cv2.medianBlur(noisy_gaussian, k_med)
    denoised_results["median_on_salt_pepper"] = cv2.medianBlur(noisy_salt_pepper, k_med)
    k_g = params["gauss_ksize"] if params["gauss_ksize"] % 2 == 1 else params["gauss_ksize"] + 1
    denoised_results["gaussian_on_gaussian"] = cv2.GaussianBlur(noisy_gaussian, (k_g, k_g), params["gauss_sigma_blur"]) 
    denoised_results["gaussian_on_salt_pepper"] = cv2.GaussianBlur(noisy_salt_pepper, (k_g, k_g), params["gauss_sigma_blur"]) 
    k_avg = params["avg_ksize"] if params["avg_ksize"] % 2 == 1 else params["avg_ksize"] + 1
    averaging_kernel = np.ones((k_avg, k_avg), np.float32) / float(k_avg * k_avg)
    denoised_results["kernel_on_gaussian"] = cv2.filter2D(noisy_gaussian, -1, averaging_kernel)
    denoised_results["kernel_on_salt_pepper"] = cv2.filter2D(noisy_salt_pepper, -1, averaging_kernel)
    denoised_results["bilateral_on_gaussian"] = cv2.bilateralFilter(noisy_gaussian, params["bilateral_d"], params["bilateral_sigma_color"], params["bilateral_sigma_space"]) 
    denoised_results["bilateral_on_salt_pepper"] = cv2.bilateralFilter(noisy_salt_pepper, params["bilateral_d"], params["bilateral_sigma_color"], params["bilateral_sigma_space"]) 
    metrics = {}
    for key, img in denoised_results.items():
        metrics[key] = {"psnr": calculate_psnr(original_image, img), "ssim": calculate_ssim(original_image, img)}
    return noisy_gaussian, noisy_salt_pepper, denoised_results, metrics

st.set_page_config(page_title="Noise Reduction", layout="wide")
st.title("Noise Reduction")

col_src, col_params = st.columns([1, 1])
with col_src:
    src_choice = st.radio("Image source", ["Folder", "Upload"], index=0)
    selected_image = None
    if src_choice == "Folder":
        files = list_images("images")
        selected_image = st.selectbox("Select image", files) if files else None
        if selected_image:
            original = load_image_from_path(os.path.join("images", selected_image))
        else:
            original = None
    else:
        uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]) 
        if uploaded is not None:
            data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            original = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        else:
            original = None

with col_params:
    gauss_sigma = st.slider("Gaussian noise sigma", 0.0, 75.0, 25.0, 1.0)
    sp_amount = st.slider("Salt & Pepper amount", 0.0, 0.20, 0.05, 0.01)
    median_ksize = st.slider("Median kernel size", 3, 15, 5, 2)
    gauss_ksize = st.slider("Gaussian blur kernel size", 3, 15, 5, 2)
    gauss_sigma_blur = st.slider("Gaussian blur sigma", 0.0, 5.0, 0.0, 0.1)
    avg_ksize = st.slider("Averaging kernel size", 3, 15, 5, 2)
    bilateral_d = st.slider("Bilateral d", 1, 15, 9, 1)
    bilateral_sigma_color = st.slider("Bilateral sigmaColor", 1, 150, 75, 1)
    bilateral_sigma_space = st.slider("Bilateral sigmaSpace", 1, 150, 75, 1)
    run = st.button("Run")

if original is None:
    st.info("Add an image to the images folder or upload one.")
else:
    st.image(original, caption="Original", use_container_width=True, clamp=True)
    if run:
        params = {
            "gauss_sigma": float(gauss_sigma),
            "sp_amount": float(sp_amount),
            "median_ksize": int(median_ksize),
            "gauss_ksize": int(gauss_ksize),
            "gauss_sigma_blur": float(gauss_sigma_blur),
            "avg_ksize": int(avg_ksize),
            "bilateral_d": int(bilateral_d),
            "bilateral_sigma_color": int(bilateral_sigma_color),
            "bilateral_sigma_space": int(bilateral_sigma_space),
        }
        noisy_gaussian, noisy_salt_pepper, denoised_results, metrics = run_pipeline(original, params)
        st.subheader("Noisy")
        c1, c2 = st.columns(2)
        c1.image(noisy_gaussian, caption="Noisy Gaussian", use_container_width=True, clamp=True)
        c2.image(noisy_salt_pepper, caption="Noisy Salt & Pepper", use_container_width=True, clamp=True)
        st.subheader("Denoised")
        r1c = st.columns(4)
        items_g = [
            ("median_on_gaussian", "Median on Gaussian"),
            ("gaussian_on_gaussian", "Gaussian on Gaussian"),
            ("kernel_on_gaussian", "Kernel on Gaussian"),
            ("bilateral_on_gaussian", "Bilateral on Gaussian"),
        ]
        for i, (key, title) in enumerate(items_g):
            r1c[i].image(denoised_results[key], caption=f"{title} | PSNR {metrics[key]['psnr']:.2f} | SSIM {metrics[key]['ssim']:.4f}", use_container_width=True, clamp=True)
        r2c = st.columns(4)
        items_s = [
            ("median_on_salt_pepper", "Median on Salt & Pepper"),
            ("gaussian_on_salt_pepper", "Gaussian on Salt & Pepper"),
            ("kernel_on_salt_pepper", "Kernel on Salt & Pepper"),
            ("bilateral_on_salt_pepper", "Bilateral on Salt & Pepper"),
        ]
        for i, (key, title) in enumerate(items_s):
            r2c[i].image(denoised_results[key], caption=f"{title} | PSNR {metrics[key]['psnr']:.2f} | SSIM {metrics[key]['ssim']:.4f}", use_container_width=True, clamp=True)
        if selected_image:
            base = os.path.splitext(selected_image)[0]
        else:
            base = "uploaded"
        paths = {}
        for name, img in {"noisy_gaussian": noisy_gaussian, "noisy_salt_pepper": noisy_salt_pepper}.items():
            p = os.path.join(OUTPUT_DIR, f"{base}_{name}.png")
            cv2.imwrite(p, img)
            paths[name] = p
        for name, img in denoised_results.items():
            p = os.path.join(OUTPUT_DIR, f"{base}_{name}.png")
            cv2.imwrite(p, img)
            paths[name] = p
        st.subheader("Download")
        dl_cols = st.columns(4)
        keys = list(paths.keys())
        for i, key in enumerate(keys):
            with open(paths[key], "rb") as f:
                data = f.read()
            dl_cols[i % 4].download_button(label=key, data=data, file_name=os.path.basename(paths[key]))