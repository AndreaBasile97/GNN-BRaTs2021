# GNN-BRaTs2021

This project refers to my thesis of Master degree in Computer Science.

The goal is to:

1. preprocess the BraTS-2021 dataset
2. produce graphs for each patient brain including informations from the four modalities of 3D MRI (T1, T1-ce, T2 and Flair). 
3. Map the classified graph to the 3D Mri scan in order to produce a segmented tumor divided in his main three regions (Necrotic, Edema and Enh. Tumor)


These graphs will be the input for different GNN such as:

- Graph Attention Network
- Graph Convolutional Network
- ChebNet

All these GNN are evaluated using Dice-Score and HD95 score (Node Wisely and Voxel Wisely) also taking into account the training time and computational resources available.

The output will be a 3D MRI of the patient brain but with the highlighted tumor zones.
