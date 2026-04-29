import numpy as np
import matplotlib.pyplot as plt

import torch
import cv2

def normalize_depth(depth_image):
    min_depth = np.min(depth_image)
    max_depth = np.max(depth_image)
    
    if max_depth - min_depth == 0:
        print("Warning: Depth image has no variation. Returning zero array.")
        return np.zeros_like(depth_image, dtype=np.float32)
    
    normalized_depth = (depth_image - min_depth) / (max_depth - min_depth)
    return normalized_depth

def viridis_cmap(gray: np.ndarray) -> np.ndarray:
    colored = plt.cm.viridis(plt.Normalize()(gray.squeeze()))[..., :-1]
    return colored.astype(np.float32)

def turbo_cmap(gray: np.ndarray) -> np.ndarray:
    colored = plt.cm.turbo(plt.Normalize()(gray.squeeze()))[..., :-1]
    return colored.astype(np.float32)

def to8b(img):
    if img.max() > 1.0 or img.min() < 0.0:
        img = (img - img.min()) / (img.max() - img.min())
    img = img * 255.0
    img = img.astype(np.uint8)
    return img


def save_depth_npy(depth_map, filename):
    np.save(filename, depth_map)


def save_rgb_png(rgb_image, filename):
    rgb_image = to8b(rgb_image)
    
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(filename, bgr_image)


def save_depth_png(depth_map, filename, max_depth=2000):
    depth_mm = np.rint(depth_map).astype(np.uint16)
    cv2.imwrite(filename, depth_mm, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    
    
    

def read_depth_png(filename, max_depth=2000):
    depth_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    
    depth_image = depth_image.astype(np.float32)

    depth_image = (depth_image / (2**16 - 1)) * max_depth
    depth_image[depth_image > max_depth] = 0
    
    return depth_image


def draw_label_at_mask_center(rgb_image, mask, label_text, font_scale=1.0, 
                            font_thickness=2, text_color=(255, 255, 255), 
                            bg_color=None, font=cv2.FONT_HERSHEY_SIMPLEX):
    result_image = rgb_image.copy()
    
    y_coords, x_coords = np.where(mask == 1)
    
    if len(y_coords) == 0:
        print("Warning: No pixels with value 1 found in mask")
        return result_image
    
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, font, font_scale, font_thickness
    )
    
    text_x = center_x - text_width // 2
    text_y = center_y + text_height // 2
    
    if bg_color is not None:
        padding = 2
        cv2.rectangle(
            result_image,
            (text_x - padding, text_y - text_height - padding),
            (text_x + text_width + padding, text_y + baseline + padding),
            bg_color,
            -1
        )
    
    cv2.putText(
        result_image,
        label_text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        font_thickness
    )
    
    return result_image


def concatenate_images_horizontal(img1_path, img2_path, gap_width=20, output_path=None):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"Error: Could not load images from {img1_path} or {img2_path}")
        return None
    
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    
    max_height = max(h1, h2)
    
    if h1 != max_height:
        aspect_ratio = w1 / h1
        new_width = int(max_height * aspect_ratio)
        img1 = cv2.resize(img1, (new_width, max_height))
        w1 = new_width
    
    if h2 != max_height:
        aspect_ratio = w2 / h2
        new_width = int(max_height * aspect_ratio)
        img2 = cv2.resize(img2, (new_width, max_height))
        w2 = new_width
    
    total_width = w1 + gap_width + w2
    concatenated = np.ones((max_height, total_width, 3), dtype=np.uint8) * 0
    
    concatenated[:, :w1] = img1
    concatenated[:, w1+gap_width:w1+gap_width+w2] = img2
    
    if output_path:
        cv2.imwrite(output_path, concatenated)
        print(f"Concatenated image saved to: {output_path}")
    
    return concatenated