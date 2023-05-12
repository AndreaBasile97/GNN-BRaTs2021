
import nibabel as nib
import numpy as np
import os

def rescale_normalize_nii_image(image):
    # Get the image data as a NumPy array
    img_data = image.get_fdata()
    # Calculate the 99.5th percentile value
    percentile_99_5 = np.percentile(img_data, 99.5)
    # Rescale intensities by dividing them by the 99.5th percentile value
    img_data_rescaled = img_data / percentile_99_5
    # Clip the values to the range [0, 1]
    img_data_rescaled = np.clip(img_data_rescaled, 0, 1)
    # Create a new NIfTI image with the rescaled data
    mean = np.mean(img_data_rescaled)
    std_dev = np.std(img_data_rescaled)
    img_data_normalized = (img_data_rescaled - mean) / std_dev
    # Create a new NIfTI image with the rescaled and normalized data
    normalized_img = nib.Nifti1Image(img_data_normalized, image.affine)
    return normalized_img


def crop_nii(image_data, img):
    target_shape = (240, 240, 155)
    # Get the original shape of the image data
    original_shape = image_data.shape
    # Calculate the start and end indices for each dimension
    start_indices = tuple((original_shape[i] - target_shape[i]) // 2 for i in range(3))
    end_indices = tuple(start_indices[i] + target_shape[i] for i in range(3))
    # Crop the image data using NumPy slicing
    cropped_image_data = image_data[
        start_indices[0]:end_indices[0],
        start_indices[1]:end_indices[1],
        start_indices[2]:end_indices[2]
    ]    
    # Use the original image's affine transformation matrix
    affine = img.affine
    # Create a new Nifti1Image object with the cropped image data
    cropped_nifti_image = nib.Nifti1Image(cropped_image_data, affine)
    return cropped_nifti_image


def preprocess(brain_images_path):
    print('starting preprocessing...')
    subdirectories = [d for d in os.listdir(brain_images_path) if os.path.isdir(os.path.join(brain_images_path, d))]
    for subdir in subdirectories:
        subdir_path = os.path.join(brain_images_path, subdir)
        print(subdir)
        os.makedirs(f'../preprocessed_dataset/{subdir}')
        for filename in os.listdir(subdir_path):
            try:
                scan_modality = (filename.split("_")[2]).split(".")[0]
                if scan_modality in ['flair','t1','t2','t1ce']:
                    scan = nib.load(f"{brain_images_path}/{subdir}/{filename}")
                    preprocessed_image = rescale_normalize_nii_image(scan)
                    cropped_scan = crop_nii(preprocessed_image.get_fdata(), scan)
                    nib.save(cropped_scan, f"../preprocessed_dataset/{subdir}/{filename}")
                    print(f"{filename} correctly preprocessed")
            except:
                raise Exception(f"{filename} is not a valid file for processing")