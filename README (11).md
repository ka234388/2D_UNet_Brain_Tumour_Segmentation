# Brain Tumor Segmentation Using 2D U-Net

## Overview

This project implements a deep learning approach to automatically identify and segment brain tumors from MRI scans. Using a 2D U-Net neural network architecture trained on the BraTS dataset, the model learns to detect different regions of brain tumors including the enhancing tumor, tumor core, and the overall tumor area. The work evaluates performance across multiple training configurations and provides insights into effective strategies for medical image segmentation tasks.

## Why This Matters

Brain tumor segmentation is one of the most important applications of deep learning in medical imaging. Manually identifying tumor boundaries in MRI scans is time-consuming, requires extensive medical expertise, and can vary significantly between different radiologists. An automated segmentation system helps oncologists make faster, more consistent treatment decisions.

With an accurate segmentation model, doctors can better plan surgical approaches, determine radiation therapy targets, and assess tumor severity. Since tumor sub-regions (like enhancing vs. non-enhancing areas) have different characteristics and prognosis implications, segmenting them separately provides crucial information. This work tackles the challenge of building a reliable automated system that can work on standard medical imaging hardware, making it practical for real clinical workflows.

## The Approach

The foundation of this project is the U-Net architecture, a convolutional neural network specifically designed for image segmentation tasks. Unlike typical classification networks that output a single prediction, U-Net processes images pixel-by-pixel, assigning each pixel to one of five classes: background, necrotic tissue, edema (swelling), enhancing tumor, or tumor core.

The network works through an encoder-decoder design. The encoder progressively downsamples the input MRI slice, extracting increasingly abstract features at each level (starting with 64 feature maps and growing to 512). The decoder then upsamples these features back to the original image resolution, using skip connections that preserve fine details from the encoding path. This symmetric structure with skip connections is what makes U-Net effective for medical image segmentation.

For this project, 3D MRI volumes were converted into individual 2D axial slices for more efficient processing on limited GPU memory. Each slice was normalized to a standard range and processed independently through the network. The model outputs a segmentation map where each pixel receives a probability score for each class, and the final prediction selects the class with the highest probability.

## How It Was Done

### Dataset and Preparation

The work uses the BraTS (Brain Tumor Segmentation) dataset, specifically the Task01_BrainTumour subset, which contains real MRI scans from patients with brain tumors along with expert segmentation masks. Due to GPU memory constraints on a T4 GPU, the project trained on only 10% of the dataset (approximately 30-50 images) rather than the full dataset. Each 3D volume was sliced into 2D images, and pixel intensities were normalized between 0 and 1.

### Training Strategy

The model was trained using a 5-fold cross-validation approach, which provides a more reliable estimate of how well the model generalizes. Each fold splits the data so that the model is trained on approximately 80% and validated on 20%. This process was repeated five times with different data splits, and the final performance represents an average across all folds.

Several training configurations were tested to understand the impact of dataset size and training duration:

- **Configuration 1**: 50 images, 15 epochs → Achieved average Dice score of 0.8275
- **Configuration 2**: 50 images, 3 epochs → Average Dice score of 0.7845 (underfitting)
- **Configuration 3**: 30 images, 10 epochs → Average Dice score of 0.8263
- **Configuration 4**: 48 images, 5 epochs → Average Dice score of 0.8120

The model was trained using Cross-Entropy Loss (a standard loss function for multi-class segmentation) and the Adam optimizer with a learning rate of 0.0001. Batch size was kept small (8-12 images) due to memory limitations. After each epoch, the model achieving the best performance on the validation set was saved, preserving the best-performing version.

### Evaluation Metrics

Two metrics were used to evaluate segmentation quality:

**Dice Similarity Coefficient**: This metric measures the overlap between the predicted segmentation and the ground truth. It ranges from 0 (no overlap) to 1 (perfect match). A Dice score of 0.82 means the predicted tumor region overlaps with the true tumor region about 82% of the time.

**Hausdorff Distance (95%)**: This metric captures how well the predicted tumor boundaries align with the true boundaries, measured in millimeters. Lower values indicate better boundary precision. The 95% threshold makes the metric more robust to outliers.

## Results and Findings

### Performance Highlights

The best configuration (50 images, 10 epochs) achieved an average Whole Tumor Dice score of **0.8275** across all five folds, with consistent performance across all folds (ranging from 0.8064 to 0.8490). This indicates the model learned to reliably identify tumor regions despite being trained on only a small subset of the full dataset.

The boundary alignment was reasonable, with an average Hausdorff Distance of 3.5 mm for the whole tumor region, suggesting the model captures tumor locations effectively. Different epochs showed different trade-offs: more epochs improved overall scores but required more computational time, while fewer epochs sometimes led to underfitting.

### What Worked Well

The model successfully segments the main tumor regions, and tumor sub-regions were clearly distinguishable in the output. The 5-fold cross-validation demonstrated stable performance across different data splits, indicating the model isn't simply memorizing specific training images. The architecture's skip connections effectively preserved fine spatial details, particularly visible in the tumor core segmentation.

### Where Improvement Is Needed

With limited training data and epochs, the model sometimes overfitted to specific tumor characteristics. For example, Enhancing Tumor Dice scores were consistently 1.0 (perfect), which seems unrealistically high and suggests the model isn't generalizing well for this sub-region. The model also generated false positives—segmenting regions outside the actual tumor, particularly at the bottom edges of some images. Tumor boundaries appeared somewhat blurry compared to the crisp ground truth, indicating the model could benefit from more advanced training techniques.

Some folds performed better than others (Fold 3 sometimes achieved Dice of 0.86, while other times it dropped to 0.74), showing that certain data splits were more challenging. This variation highlights that the limited dataset size creates some instability in training.

## How This Is Useful

This project demonstrates that even with limited computational resources and reduced dataset sizes, a U-Net model can achieve reasonable performance on brain tumor segmentation. For research purposes, this shows that 2D approaches can work effectively, though 3D methods would likely improve spatial consistency.

The memory optimization techniques used here (garbage collection and CUDA cache clearing) make it possible to train on standard GPUs without expensive high-end hardware. This makes the approach accessible to researchers without large computational budgets.

The consistent cross-fold performance suggests that with more training data and longer training schedules, this architecture would likely achieve clinical-grade segmentation. The findings also provide baseline metrics that can inform future improvements, such as using Dice Loss instead of Cross-Entropy, applying post-processing filters to reduce false positives, or upgrading to 3D U-Net for volumetric consistency.

## Getting Started

### Requirements

You'll need Python 3.8+ and the following libraries:

```
torch>=2.0.0
torchvision>=0.15.0
nibabel>=4.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
scipy>=1.7.0
```

### Installation

**For Google Colab (Recommended):**

```python
# Run these in your Colab notebook
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install nibabel numpy pandas scikit-learn matplotlib scipy

# Verify GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

**For Local Machine:**

```bash
# Create conda environment
conda create -n brain-tumor-seg python=3.9
conda activate brain-tumor-seg

# Install dependencies
pip install -r requirements.txt
```

### Running the Code

1. Download the BraTS dataset from https://www.med.upenn.edu/cbica/brats/
2. Extract it to a `data/` folder
3. Open `brain_tumor_segmentation.ipynb` in Jupyter
4. Run cells sequentially to load data, train the model, and visualize results

## Project Structure

```
├── brain_tumor_segmentation.ipynb    # Main notebook with full pipeline
├── requirements.txt                   # Python dependencies
├── data/
│   └── BraTS/                        # Downloaded dataset
├── models/
│   └── best_model.pth                # Saved trained model
└── README.md                          # This file
```

## Key Takeaways

This assignment demonstrated that U-Net is an effective architecture for medical image segmentation, even with limited training data. The main challenges encountered were memory constraints (limiting dataset size to 10%), overfitting on some sub-regions, and false positives in segmentation output. The most successful configuration used moderate training data (50 images) with sufficient epochs (10) to allow the model to learn meaningful features.

For practical clinical deployment, the next steps would involve training on the full dataset, implementing advanced loss functions like Focal Loss or combined Dice+CE Loss, adding post-processing to reduce false positives, and likely migrating to 3D U-Net for improved spatial consistency. The current results provide a solid foundation and demonstrate the feasibility of this approach.

## References

- Ronneberger et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- BraTS Dataset: https://www.med.upenn.edu/cbica/brats/
- PyTorch Documentation: https://pytorch.org/

---

**Author**: Karthika Ramasamy  
**Course**: CAP 5516 - Computer Vision  
**University**: University of Central Florida  
**Date**: 2025
