
import nibabel as nib
import numpy as np
import os
import numpy as np


class DatasetPreprocesser:

    def __init__(self, brats_raw_dataset_path, save_path):
        self.brats_path = brats_raw_dataset_path
        self.save_path = save_path


    def rescale_normalize_nii_image(self, image):
        img_data = image.get_fdata()
        percentile_99_5 = np.percentile(img_data, 99.5)
        img_data_rescaled = img_data / percentile_99_5
        img_data_rescaled = np.clip(img_data_rescaled, 0, 1)
        mean = np.mean(img_data_rescaled)
        std_dev = np.std(img_data_rescaled)
        img_data_normalized = (img_data_rescaled - mean) / std_dev
        normalized_img = nib.Nifti1Image(img_data_normalized, image.affine)
        return normalized_img


    def crop_nii(self, image_data, nifti_image):
        non_zero_coords = np.nonzero(image_data)
        min_coords = np.min(non_zero_coords, axis=1)
        max_coords = np.max(non_zero_coords, axis=1)
        cropped_image_data = image_data[min_coords[0]:max_coords[0]+1,
                                        min_coords[1]:max_coords[1]+1,
                                        min_coords[2]:max_coords[2]+1]
        cropped_image = nib.Nifti1Image(cropped_image_data, nifti_image.affine, nifti_image.header)
        return cropped_image, min_coords, max_coords  # Return the min and max coordinates


    def pad_images(dataset_path):
        patients = os.listdir(dataset_path)

        for patient in patients:
            patient_path = os.path.join(dataset_path, patient)
            
            # Load all images
            img_t1 = nib.load(os.path.join(patient_path, f'{patient}_t1.nii.gz'))
            img_t2 = nib.load(os.path.join(patient_path, f'{patient}_t2.nii.gz'))
            img_t1ce = nib.load(os.path.join(patient_path, f'{patient}_t1ce.nii.gz'))
            img_flair = nib.load(os.path.join(patient_path, f'{patient}_flair.nii.gz'))
            img_seg = nib.load(os.path.join(patient_path, f'{patient}_seg.nii.gz'))
            images = [img_t1, img_t2, img_t1ce, img_flair, img_seg]
            data = [img.get_fdata() for img in images]

            # Find the maximum shape
            max_shape = np.max([img.shape for img in data], axis=0)

            # Pad images to the maximum shape
            padded_data = []
            for img in data:
                pad_width = [(0, max_s - s) for s, max_s in zip(img.shape, max_shape)]
                padded_img = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)
                padded_data.append(padded_img)

            # Save padded images
            for img, img_data, img_name in zip(images, padded_data, ['t1', 't2', 't1ce', 'flair', 'seg']):
                new_img = nib.Nifti1Image(img_data, img.affine, img.header)
                
                nib.save(new_img, os.path.join(patient_path, f'{patient}_{img_name}.nii.gz'))

    def preprocess(self):
        
        print('starting preprocessing...')
        subdirectories = [d for d in os.listdir(self.brats_path) if os.path.isdir(os.path.join(self.brats_path, d))]
        for subdir in subdirectories:
            subdir_path = os.path.join(self.brats_path, subdir)
            print(subdir)
            os.makedirs(os.path.join(self.save_path, subdir), exist_ok=True)
            for filename in os.listdir(subdir_path):
                try:
                    scan_modality = (filename.split("_")[2]).split(".")[0]
                    if scan_modality in ['flair','t1','t2','t1ce']:
                        scan = nib.load(os.path.join(subdir_path, filename))
                        cropped_scan, min_coords, max_coords = self.crop_nii(scan.get_fdata(), scan)
                        preprocessed_image = self.rescale_normalize_nii_image(cropped_scan)

                        # Load and crop the segmentation mask using the same coordinates
                        seg_filename = filename.replace(scan_modality, 'seg')  # Assuming the segmentation file has the same name but with 'seg' instead of the modality
                        seg_scan = nib.load(os.path.join(subdir_path, seg_filename))
                        cropped_seg_data = seg_scan.get_fdata()[min_coords[0]:max_coords[0]+1,
                                                                min_coords[1]:max_coords[1]+1,
                                                                min_coords[2]:max_coords[2]+1]
                        cropped_seg = nib.Nifti1Image(cropped_seg_data, seg_scan.affine, seg_scan.header)

                        # Save the preprocessed image and the cropped segmentation mask
                        nib.save(preprocessed_image, os.path.join(self.save_path, subdir, filename))
                        nib.save(cropped_seg, os.path.join(self.save_path, subdir, seg_filename))  # Save the cropped segmentation mask in the same folder
                        print(f"{filename} and corresponding segmentation mask correctly preprocessed")
                except:
                    raise Exception(f"{filename} is not a valid file for processing")
        print('Preprocessing terminated.')
        self.pad_images(self.brats_path)

