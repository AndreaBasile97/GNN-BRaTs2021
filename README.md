# GNN-BRaTs2021 🧠🔬

Welcome to the GNN-BRaTs2021 project, the focus of my Master's thesis in Computer Science. This project aims to leverage Graph Neural Networks (GNNs) for brain tumor segmentation using the BRaTS-2021 dataset. The comprehensive approach involves preprocessing the data, generating graphs for each patient's brain based on four MRI modalities (T1, T1-ce, T2, and Flair), and mapping the classified graphs to 3D MRI scans for tumor segmentation into Necrotic, Edema, and Enhanced Tumor regions.

## Project Goals:

1. **Preprocessing:** Utilize the 'preprocess_dataset.py' script to prepare the BRaTS-2021 dataset for further analysis.

2. **Graph Generation:** Produce informative graphs for each patient's brain by integrating data from the four modalities of 3D MRI.

3. **Tumor Segmentation:** Map the classified graph to the 3D MRI scan, resulting in a segmented tumor with distinct regions (Necrotic, Edema, and Enhanced Tumor).

4. **GNN Implementation:** Implement various Graph Neural Networks, including:
   - Graph Attention Network
   - Graph Convolutional Network
   - ChebNet

5. **Evaluation Metrics:** Assess the performance of GNNs using Dice-Score and HD95 scores, considering both Node-wise and Voxel-wise metrics. Take into account training time and available computational resources.

## How to Use:

1. **Installation:** Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```

2. **Preprocessing:** Ensure the BRaTS2021 dataset is in the root directory and execute the 'preprocess_dataset.py' script.

3. **Training:** Train the GNN by using the 'training.py' script, and remember to adjust the paths accordingly.

4. **Testing:** Evaluate the GNN using the saved model and the test dataset.

## Results:

The output of the project will be a 3D MRI of the patient's brain with highlighted tumor zones, providing valuable insights into tumor segmentation.

Feel free to explore and contribute to this exciting project! 🚀🧑‍💻