## This is the code of the paper "Unsupervised 3D CNN-based Spatial-Temporal Tensor Model for Infrared Small Target Detection".
# Algorithm Introduction
![image](https://github.com/ELOESZHANG/Infrared-Small-Target-Detection/blob/main/img_demo/architecture.png)

We first convert the infrared image sequence into an infrared tensor in the spatial-temporal domain and construct a 3D CNN (3D Low-Rank Net) to extract sufficient spatial-temporal motion information. Next, we embed the 3D Low-Rank Net into the sparse regularization-based spatial-temporal tensor optimization model, which can effectively enhance the intrinsic features of the targets by introducing low-rank and sparse priors. Finally, the proposed model's unsupervised optimization strategy eliminates the need for labeled datasets. To the best of our knowledge, our approach is the first deep learning-based unsupervised method for detecting small targets in infrared sequences. Therefore, a new sequence-based infrared small target detection method is provided to address the problems listed before and effectively enhance detection performance. The following is a summary of the main contributions of this paper:

1. For detecting small targets in infrared sequences, we propose a novel detection model based on unsupervised deep learning and low-rank sparse decomposition for the first time.

2. To fully use both the spatial and temporal features of the infrared sequences, we construct a sparse regularization-based spatial-temporal tensor optimization model and devise a 3D CNN (3D Low-Rank Net) to be inserted into the model.

3. By injecting low-rank and sparse priors into the loss function, we can efficiently enrich target features and obtain accurate detection results after learning tensor factor parameters and net parameters unsupervisedly.

4. We compare our method with current state-of-the-art CNN-based and traditional methods on public infrared sequences. The experiment results demonstrate that our model has optimal detection performance.

# Qualitative Results
![image](https://github.com/ELOESZHANG/Infrared-Small-Target-Detection/blob/main/img_demo/fig_1.png)
![image](https://github.com/ELOESZHANG/Infrared-Small-Target-Detection/blob/main/img_demo/fig_2.png)
![image](https://github.com/ELOESZHANG/Infrared-Small-Target-Detection/blob/main/img_demo/fig_3.png)

# Usage
run `main_code/main.py` to test your datasets.

# Reference
Part of the code is borrowed from the work of the following authors:

1.https://github.com/YeRen123455/Infrared-Small-Target-Detection

2.https://github.com/Tianfang-Zhang/AGPCNet

3.Luo Y, Zhao X L, Meng D, et al. HLRTF: Hierarchical Low-Rank Tensor Factorization for Inverse Problems in Multi-Dimensional Imaging[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 19303-19312.
