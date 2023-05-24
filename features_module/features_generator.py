import numpy as np
import pickle
import nibabel as nib
import os

def extract_features(slic_data, scan_list):
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


def get_patient_ids(paths):
    ids = []
    for path in paths:
        splitted_path = path.split("/")
        ids.append(splitted_path[-1].split("_")[1])
    if all(elem == ids[0] for elem in ids):
        return ids, True
    else:
        return ids, False


def generate_features(new_dataset, features_path):

    list_feature_completed = []
    features_completed = [d for d in os.listdir(features_path)]
    for f in features_completed:
        id_f = f.split('_')[1].split('.')[0]
        list_feature_completed.append(id_f)

    i = 1
    subdirectories = [d for d in os.listdir(new_dataset) if os.path.isdir(os.path.join(new_dataset, d))]
    for subdir in subdirectories:
        if not subdir == 'features':
            subdir_path = os.path.join(new_dataset, subdir)
            scans = []
            try:
                id, flag = get_patient_ids([f"{new_dataset}{subdir}"])
                i = i + 1
                if id[0] not in list_feature_completed:
                    print(f"Processing {id[0]}: slic taken from {new_dataset} | flair, t1, t2, t1ce from {new_dataset}")
                    for filename in os.listdir(subdir_path):
                        try:
                            substring = filename.split("_")[2]
                            modality = substring.split(".")[0]
                            if modality in ['SLIC']:
                                slic = nib.load(f"{new_dataset}{subdir}/{filename}")
                            elif modality in ['flair','t1','t2','t1ce']:
                                scans.append(nib.load(f"{new_dataset}{subdir}/{filename}").get_fdata())
                        except:
                            pass

                    features = extract_features(slic.get_fdata(), scans)

                    with open(f"{features_path}features_{id[0]}.pkl", "wb") as f:
                        pickle.dump(features, f)
            except Exception as e:
                print(e)


generate_features('/ext/PREPROCESSED_DATASET_NEW/', '/ext/tesi_BraTS2021/features/')