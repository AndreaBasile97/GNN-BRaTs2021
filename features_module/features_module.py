import numpy as np
import pickle
import nibabel as nib
import os
from training.utilities import get_patient_ids

class FeaturesModule:
    def __init__(self, dataset_path, save_path):
        self.dataset_path = dataset_path
        self.save_path = save_path


    def extract_features(self, slic_data, scan_list):
        percentiles = [10, 25, 50, 75, 90]
        features = {}

        for cluster in np.unique(slic_data):
            feature_vector = []
            coordinate = np.where(slic_data == cluster)
            
            for scan in scan_list:
                supervoxel = scan[coordinate]
                modality_features = [np.percentile(supervoxel, p) for p in percentiles]
                feature_vector.extend(modality_features)
                features[cluster] = feature_vector
        
        return features     


    # Generating features for the entire dataset
    def generate_features(self):
        list_feature_completed = []
        features_completed = [d for d in os.listdir(self.save_path)]
        for f in features_completed:
            id_f = f.split('_')[1].split('.')[0]
            list_feature_completed.append(id_f)

        i = 1
        subdirectories = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        for subdir in subdirectories:
            if not subdir == 'features':
                subdir_path = os.path.join(self.dataset_path, subdir)
                scans = []
                try:
                    id, flag = get_patient_ids([f"{self.dataset_path}/{subdir}"])
                    i = i + 1
                    if id[0] not in list_feature_completed:
                        print(f"Processing {id[0]}: slic taken from {self.dataset_path} | flair, t1, t2, t1ce from {self.dataset_path}")
                        for filename in os.listdir(subdir_path):
                            try:
                                substring = filename.split("_")[2]
                                modality = substring.split(".")[0]
                                if modality in ['SLIC']:
                                    slic = nib.load(f"{self.dataset_path}/{subdir}/{filename}")
                                elif modality in ['flair','t1','t2','t1ce']:
                                    scans.append(nib.load(f"{self.dataset_path}/{subdir}/{filename}").get_fdata())
                            except:
                                pass

                        features = self.extract_features(slic.get_fdata(), scans)

                        with open(f"{self.save_path}/features_{id[0]}.pkl", "wb") as f:
                            pickle.dump(features, f)
                except Exception as e:
                    print(e)
