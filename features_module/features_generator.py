import numpy as np
import pickle
import nibabel as nib
import os

def compute_supervoxel_percentile(SLIC, *scans):
    supervoxel_percentile = {}
    
    unique_labels = np.unique(SLIC)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude the zero label

    # Create a mask that contains only the unique labels in SLIC
    mask = np.isin(SLIC, unique_labels)

    # Loop over each scan
    for scan in scans:
        # Use the mask to extract the intensities for each label from the scan
        intensities = [scan[mask & (SLIC == label)] for label in unique_labels]

        # Compute the percentile for each set of intensities
        scan_percentiles = {int(label): [np.percentile(values, 10), np.percentile(values, 25), np.percentile(values, 50), np.percentile(values, 75), np.percentile(values, 90)] for label, values in zip(unique_labels, intensities)}
        
        # If the label is already in the dictionary, concatenate the percentiles. If not, add it to the dictionary.
        for label, percentiles in scan_percentiles.items():
            if label in supervoxel_percentile:
                supervoxel_percentile[label].extend(percentiles)
            else:
                supervoxel_percentile[label] = percentiles

    return supervoxel_percentile


def get_patient_ids(paths):
    ids = []
    for path in paths:
        splitted_path = path.split("/")
        ids.append(splitted_path[-1].split("_")[1])
    if all(elem == ids[0] for elem in ids):
        return ids, True
    else:
        return ids, False


def generate_features(old_dataset, new_dataset, features_path):

    list_feature_completed = []
    features_completed = [d for d in os.listdir(features_path)]
    for f in features_completed:
        id_f = f.split('_')[1].split('.')[0]
        list_feature_completed.append(id_f)

    i = 1
    subdirectories = [d for d in os.listdir(old_dataset) if os.path.isdir(os.path.join(old_dataset, d))]
    for subdir in subdirectories:
        if not subdir == 'features':
            subdir_path = os.path.join(old_dataset, subdir)
            scans = []
            id, flag = get_patient_ids([f"{old_dataset}{subdir}"])
            i = i + 1
            if id[0] not in list_feature_completed:
                print(f"Processing {id[0]}: slic taken from {old_dataset} | flair, t1, t2, t1ce from {new_dataset}")
                for filename in os.listdir(subdir_path):
                    try:
                        substring = filename.split("_")[2]
                        modality = substring.split(".")[0]
                        if modality in ['SLIC']:
                            slic = nib.load(f"{old_dataset}{subdir}/{filename}")
                        elif modality in ['flair','t1','t2','t1ce']:
                            scans.append(nib.load(f"{new_dataset}{subdir}/{filename}").get_fdata())
                    except:
                        pass

                features = compute_supervoxel_percentile(slic.get_fdata(), scans[0], scans[1], scans[2], scans[3])

                with open(f"{features_path}features_{id[0]}.pkl", "wb") as f:
                    pickle.dump(features, f)


generate_features('../datasets/old_dataset/', '../datasets/preprocessed_dataset/', './new_features/')