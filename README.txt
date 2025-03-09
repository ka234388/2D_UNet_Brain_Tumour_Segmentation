# 2D_UNet_Brain_Tumour_Segmentation
**Overview** 
This Assignment implements a 2D U-Net-based Brain Tumor Segmentation model trained on the BraTS dataset. The model segments MRI images into different tumor sub-regions using deep learning techniques. A 5-fold cross-validation strategy is applied to evaluate model performance.
Dataset
The dataset used is from the Brain Tumor Image Segmentation (BraTS) challenge, available https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2. Download the Task01_BrainTumour.tar file and extract it.
The dataset structure should look like:
Task01_BrainTumour/
├── dataset.json
├── imagesTr/      # Training images
├── labelsTr/      # Training labels
└── imagesTs/      # Test images (not used)

**How To Run**
git clone https://github.com/your-repo/BrainTumorSegmentation.git
cd BrainTumorSegmentation

Step 1: To Run the code: Install pip install torch medpy nibabel numpy scikit-learn tqdm matplotlib 
Step 2: Mount the Drive
Step 3: Extract the tar file from the drive location or Attached the Dataset from the drive or upload it from you local and Run the cell one by one to execute the Assignement.
Step 4: Run The Task 1 cell for the MRI Segmentation mask visualization
Step 5: Run The Task 2 cells for the 5 fold function executiona nd finally it will give the output of final segmentation. 
Step 6: For better results add the Image size, Epoch and Batch sizes as needed.

**Challenges & Future Work**
 Memory Limitations: Unable to train full dataset due to GPU constraints
 False Positives: Some extra segmented regions detected
 Blurry Tumor Boundaries: Model needs better boundary refinement
 Future Work: Upgrade to 3D U-Net for better spatial consistency
