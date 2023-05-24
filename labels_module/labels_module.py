import os
import nibabel as nib
import pickle
import numpy as np
from training.utilities import get_patient_ids

class LabelsModule:
    def __init__(self, dataset_path, labels_path):
        self.dataset_path = dataset_path
        self.labels_path = labels_path


    def generate_labels(self, segmented_data, ground_truth_data):
        labels = np.zeros_like(segmented_data, dtype=np.uint8)

        # Define labels and thresholds for each class
        labels_map = {
            'core_tumor': 1,
            'enhancing_tumor': 2,
            'peritumoral_tissue': 4
        }
        thresholds = {
            'core_tumor': 0.5,
            'enhancing_tumor': 0.5,
            'peritumoral_tissue': 0.5
        }

        # Find the SLIC segments intersecting with the tumor region
        tumor_mask = (ground_truth_data != 0)
        intersecting_segments = np.unique(segmented_data[tumor_mask])

        # Assign labels only to segments intersecting with the tumor region
        for i in intersecting_segments:
            superpixel_mask = (segmented_data == i)
            superpixel_sum = np.sum(superpixel_mask)

            if superpixel_sum == 0:
                continue

            for voxel_value in labels_map.values():
                intersection = np.sum(superpixel_mask & (ground_truth_data == voxel_value))
                intersection_ratio = intersection / superpixel_sum

                if intersection_ratio > thresholds[list(labels_map.keys())[list(labels_map.values()).index(voxel_value)]]:
                    labels[superpixel_mask] = voxel_value
                    break
        return labels


    # Generating labels for the entire dataset
    def generate_all_labels(self):
        subdirectories = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        for subdir in subdirectories:
            subdir_path = os.path.join(self.dataset_path, subdir)
            try:
                id, flag = get_patient_ids([f"{self.dataset_path}/{subdir}"])
                for filename in os.listdir(subdir_path):
                    scan_modality = (filename.split("_")[2]).split(".")[0]
                    if scan_modality in ['SLIC']:
                        scan = nib.load(f"{self.dataset_path}/{subdir}/{filename}").get_fdata()
                    if scan_modality in ['seg']:
                        seg = nib.load(f"{self.dataset_path}/{subdir}/{filename}").get_fdata()
                seg = np.round(seg).astype(int)

                labels_generated = self.generate_labels(scan, seg)

                # Verify that the generated labels are not all 0.
                if len(np.unique(labels_generated)) <= 1:
                    print(f'Error, patient {id[0]} has all labels equal to 0 or does not have any.')

                print(f"labels generated for {id[0]}")
                with open(f"{self.labels_path}/labels_{id[0]}.pkl", "wb") as f:
                    pickle.dump(labels_generated, f)
            except Exception as e:
                print(e)
