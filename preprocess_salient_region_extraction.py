import cv2
import numpy as np
from skimage import filters
from scipy.ndimage import center_of_mass
import os
from tqdm import tqdm
import gc
import argparse

# Limit OpenCV threads to avoid excessive CPU usage
cv2.setNumThreads(2)

# Original entropy map computation
def compute_entropy_map_original(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    entropy_map = filters.rank.entropy(gray_image, np.ones((5, 5)))  # Compute entropy map
    return entropy_map

# Function to threshold and identify high-entropy regions
def identify_high_entropy_regions(entropy_map, entropy_threshold=4):
    return entropy_map > entropy_threshold  # Binary mask of high-entropy regions

# Function to compute the centroid of high-entropy regions
def compute_centroid_of_high_entropy(mask):
    return center_of_mass(mask)  # Center of mass of high-entropy region

# Function to enhance entropy map for visualization
def enhance_entropy_map_for_visualization(entropy_map):
    normalized_entropy_map = cv2.normalize(entropy_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized_entropy_map = np.uint8(normalized_entropy_map)  # Convert to 8-bit grayscale
    enhanced_entropy_map = cv2.equalizeHist(normalized_entropy_map)
    return enhanced_entropy_map

def extract_high_entropy_region(image, entropy_map, centroid, entropy_threshold=1.0, min_size=224, max_iterations=1000):
    """
    Extracts a high-entropy region around a given centroid with constraints on minimum size and aspect ratio.

    Args:
        image (np.ndarray): The original image.
        entropy_map (np.ndarray): The entropy map of the image.
        centroid (tuple): The (y, x) coordinates of the centroid.
        entropy_threshold (float): Threshold for entropy expansion.
        min_size (int): Minimum size for the bounding box.
        max_iterations (int): Maximum number of iterations for circular expansion.

    Returns:
        np.ndarray: The extracted region of the image.
    """
    rows, cols = entropy_map.shape
    x_center, y_center = int(centroid[1]), int(centroid[0])  # Reverse order (center_of_mass returns y, x)
    aspect_ratio = cols / rows  # Aspect ratio of the original image

    # Set initial bounding box with minimum size centered at the centroid
    left = max(0, x_center - min_size // 2)
    right = min(cols, x_center + min_size // 2)
    top = max(0, y_center - min_size // 2)
    bottom = min(rows, y_center + min_size // 2)

    # Flags to check if each side can continue expanding
    expand_top, expand_right, expand_bottom, expand_left = True, True, True, True
    expansion_step = 2  # Expand each side by 2 pixels per iteration

    # Safety counter to avoid infinite loops
    iterations = 0

    # Perform circular expansion until each side stops due to low-entropy, boundaries, or reaching max iterations
    while (expand_top or expand_right or expand_bottom or expand_left) and iterations < max_iterations:
        iterations += 1

        # Expand top
        if expand_top and top > 0:
            top_region = entropy_map[max(0, top - expansion_step):top, left:right]
            if top_region.size > 0 and np.mean(top_region) > entropy_threshold:
                top = max(0, top - expansion_step)
            else:
                expand_top = False
        elif top <= 0:  # Stop if reached the top boundary
            expand_top = False

        # Expand right
        if expand_right and right < cols:
            right_region = entropy_map[top:bottom, right:min(cols, right + expansion_step)]
            if right_region.size > 0 and np.mean(right_region) > entropy_threshold:
                right = min(cols, right + expansion_step)
            else:
                expand_right = False
        elif right >= cols:  # Stop if reached the right boundary
            expand_right = False

        # Expand bottom
        if expand_bottom and bottom < rows:
            bottom_region = entropy_map[bottom:min(rows, bottom + expansion_step), left:right]
            if bottom_region.size > 0 and np.mean(bottom_region) > entropy_threshold:
                bottom = min(rows, bottom + expansion_step)
            else:
                expand_bottom = False
        elif bottom >= rows:  # Stop if reached the bottom boundary
            expand_bottom = False

        # Expand left
        if expand_left and left > 0:
            left_region = entropy_map[top:bottom, max(0, left - expansion_step):left]
            if left_region.size > 0 and np.mean(left_region) > entropy_threshold:
                left = max(0, left - expansion_step)
            else:
                expand_left = False
        elif left <= 0:  # Stop if reached the left boundary
            expand_left = False

    # Check if the maximum number of iterations was reached
    if iterations >= max_iterations:
        print("Warning: Maximum iterations reached in circular expansion. Check entropy map or parameters.")

    # Adjust bounding box to maintain aspect ratio
    crop_width = right - left
    crop_height = bottom - top

    if crop_width / crop_height > aspect_ratio:
        # Adjust height based on width and aspect ratio
        required_height = int(crop_width / aspect_ratio)
        height_diff = required_height - crop_height
        top = max(0, top - height_diff // 2)
        bottom = min(rows, bottom + height_diff // 2)
    else:
        # Adjust width based on height and aspect ratio
        required_width = int(crop_height * aspect_ratio)
        width_diff = required_width - crop_width
        left = max(0, left - width_diff // 2)
        right = min(cols, right + width_diff // 2)

    # Ensure the bounding box stays within image boundaries
    top = max(0, top)
    bottom = min(rows, bottom)
    left = max(0, left)
    right = min(cols, right)

    # Extract the region from the original image
    extracted_region = image[top:bottom, left:right]
    return extracted_region

# Main function to process all images in a directory
def process_directory(input_dir, output_dir, initial_entropy_threshold=5.5, min_entropy_threshold=0.5, min_size=224, dresscode=False, save_process=False):
    os.makedirs(output_dir, exist_ok=True)
    
    process_dir = os.path.join(output_dir, 'process')
    if save_process:
        os.makedirs(process_dir, exist_ok=True)

    if dresscode:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('_1.png', '_1.jpg', '_1.jpeg'))]
    else:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    batch_to_save = []

    for idx, filename in enumerate(tqdm(image_files, desc="Processing images", unit="image")):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        # Step 1: Compute entropy map 
        entropy_map = compute_entropy_map_original(image).astype(np.uint8)

        # Step 2: Adjust entropy threshold dynamically if no high-entropy region is detected
        entropy_threshold = initial_entropy_threshold
        high_entropy_mask = identify_high_entropy_regions(entropy_map, entropy_threshold)
        while not np.any(high_entropy_mask) and entropy_threshold > min_entropy_threshold:
            entropy_threshold -= 0.5  # Decrease threshold incrementally
            high_entropy_mask = identify_high_entropy_regions(entropy_map, entropy_threshold)

        # Step 3: If no high-entropy region is found after adjustments, skip further processing
        if not np.any(high_entropy_mask):
            del entropy_map, high_entropy_mask  # Free memory
            gc.collect()
            continue

        # Step 4: Compute centroid of high entropy regions
        centroid = compute_centroid_of_high_entropy(high_entropy_mask)

        # Step 5: Extract high-entropy region with minimum size enforcement
        extracted_region = extract_high_entropy_region(image, entropy_map, centroid, entropy_threshold=0.8, min_size=min_size)

        # Save salient region
        out_dir = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
        cv2.imwrite(out_dir, extracted_region)

        # Save process images if enabled
        if save_process:
            resized_original = cv2.resize(image, (386, 512))
            # resized_entropy = cv2.resize(entropy_map, (386, 512))  #  entropy map
            enhanced_entropy = cv2.resize(enhance_entropy_map_for_visualization(entropy_map), (386, 512))  # Enhanced for visualization
            resized_extracted = cv2.resize(extracted_region, (386, 512))

            # Use enhanced entropy map for concatenated visualization
            concatenated = np.hstack((resized_original, cv2.cvtColor(enhanced_entropy, cv2.COLOR_GRAY2BGR), resized_extracted))
            process_output_path = os.path.join(process_dir, f"{os.path.splitext(filename)[0]}_process.png")

            # Save batch
            batch_to_save.append({'path': process_output_path, 'data': concatenated})

            # Save batch to disk every 10 images
            if idx % 10 == 0 or idx == len(image_files) - 1:
                for item in batch_to_save:
                    cv2.imwrite(item['path'], item['data'])
                batch_to_save = []  # Clear batch after saving

        # Explicitly delete large variables to free memory
        del image, entropy_map, high_entropy_mask, centroid, extracted_region
        gc.collect()


def check_and_process(input_dir, output_dir, dresscode=False, save_process=False):
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"[!] Input directory not found: {input_dir}")
        return
    
    # Count image files
    if dresscode:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('_1.png', '_1.jpg', '_1.jpeg'))]
    else:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        

    if not image_files:
        print(f"[!] No image files found in: {input_dir}")
    else:
        print(f"Found {len(image_files)} image files in: {input_dir}")
        # Run the process
        process_directory(input_dir, output_dir, dresscode=dresscode, save_process=save_process)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Salient Region Extraction for VITON and DressCode datasets')
    parser.add_argument('--path_to_datasets', type=str, required=True, help='Root path to the datasets, e.g., ./DATA')
    parser.add_argument('--save_process', action='store_true', help='Whether to save the mid-process images (default: False)')
    args = parser.parse_args()

    path_to_datasets = args.path_to_datasets
    save_process = args.save_process
    print('Preprocess Salient Region Extraction..')

    # VITON-HD train
    print('[1/5] VITON-HD train data..')
    check_and_process(
        input_dir=os.path.join(path_to_datasets, 'zalando-hd-resized', 'train', 'cloth'),
        output_dir=os.path.join(path_to_datasets, 'zalando-hd-resized', 'train', 'cloth_sr'),
        save_process=save_process
    )

    # VITON-HD test
    print('[2/5] VITON-HD test data..')
    check_and_process(
        input_dir=os.path.join(path_to_datasets, 'zalando-hd-resized', 'test', 'cloth'),
        output_dir=os.path.join(path_to_datasets, 'zalando-hd-resized', 'test', 'cloth_sr'),
        save_process=save_process
    )

    # DressCode - dresses
    print('[3/5] DressCode dresses subset..')
    check_and_process(
        input_dir=os.path.join(path_to_datasets, 'DressCode', 'dresses', 'images'),
        output_dir=os.path.join(path_to_datasets, 'DressCode', 'dresses', 'cloth_sr'),
        dresscode=True,
        save_process=save_process
    )

    # DressCode - upper_body
    print('[4/5] DressCode upper subset..')
    check_and_process(
        input_dir=os.path.join(path_to_datasets, 'DressCode', 'upper_body', 'images'),
        output_dir=os.path.join(path_to_datasets, 'DressCode', 'upper_body', 'cloth_sr'),
        dresscode=True,
        save_process=save_process
    )

    # DressCode - lower_body
    print('[5/5] DressCode lower subset..')
    check_and_process(
        input_dir=os.path.join(path_to_datasets, 'DressCode', 'lower_body', 'images'),
        output_dir=os.path.join(path_to_datasets, 'DressCode', 'lower_body', 'cloth_sr'),
        dresscode=True,
        save_process=save_process
    )