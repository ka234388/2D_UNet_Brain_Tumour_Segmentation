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

Several training configurations were tested to understand the impact of dataset size and training duration. The loss function used was Cross-Entropy Loss, optimized with the Adam optimizer using a learning rate of 0.0001. Batch sizes were kept small (8-12 images) due to memory limitations. After each epoch, the model achieving the best performance on the validation set was saved, preserving the best-performing version.

### Evaluation Metrics

Two metrics were used to evaluate segmentation quality:

**Dice Similarity Coefficient (DSC)**: This metric measures the overlap between the predicted segmentation and the ground truth. It ranges from 0 (no overlap) to 1 (perfect match). A Dice score of 0.82 means the predicted tumor region overlaps with the true tumor region about 82% of the time.

**Hausdorff Distance (95%)**: This metric captures how well the predicted tumor boundaries align with the true boundaries, measured in millimeters. Lower values indicate better boundary precision. The 95% threshold makes the metric more robust to outliers.

## Experimental Results

### Configuration 1: 50 Images, 10 Epochs, 5-Fold Cross-Validation

This was the best-performing configuration. Training on 50 images with 10 epochs per fold provided a good balance between learning sufficient features and avoiding overfitting given the limited dataset.

| Fold | Whole Tumor Dice | Hausdorff Distance (mm) | Epoch |
|------|-----------------|------------------------|-------|
| 1    | 0.8064          | 3.8244                 | 10    |
| 2    | 0.8180          | 2.8331                 | 10    |
| 3    | 0.8488          | 3.6713                 | 10    |
| 4    | 0.8490          | 3.7658                 | 10    |
| 5    | 0.8151          | 3.3968                 | 10    |
| **Average** | **0.8275** | **3.4983** | **10** |

**Key Observations:**
- Consistently high Dice scores across all folds, ranging from 0.8064 to 0.8490
- The model showed stable performance across different data splits, indicating good generalization
- Hausdorff Distance values around 3.5 mm suggest reasonable boundary alignment
- Clear separation between different tumor regions, particularly in whole tumor (WT) and tumor core (TC) areas

**Limitations:**
- Enhancing Tumor (ET) Dice scores were consistently 1.0, suggesting potential overfitting rather than genuine generalization
- Some segmentation outputs lacked sharp boundaries, reducing fine detail accuracy
- Limited by the small training dataset size

### Configuration 2: 50 Images, 3 Epochs, 5-Fold Cross-Validation

Training with only 3 epochs resulted in underfitting, as the model didn't have sufficient iterations to learn tumor boundaries effectively.

| Fold | Whole Tumor Dice | Hausdorff Distance (mm) | Epoch |
|------|-----------------|------------------------|-------|
| 1    | 0.8046          | 3.9352                 | 3     |
| 2    | 0.7808          | 3.3614                 | 2     |
| 3    | 0.6930          | 8.2192                 | 2     |
| 4    | 0.8194          | 3.6775                 | 3     |
| 5    | 0.8249          | 3.6960                 | 3     |
| **Total** | **3.9227** | **22.8893** | **13** |

**Key Observations:**
- Lower average Dice score (0.78) compared to the 10-epoch configuration
- High variability across folds—Fold 3 performed particularly poorly (0.6930)
- Significantly higher average Hausdorff Distance (22.89 mm total), indicating poor boundary alignment
- Clear false positives (over-segmentation) with extra regions outside tumor boundaries
- Blurry or spread-out segmentation without clear definitions

### Configuration 3: 30 Images, 10 Epochs, 5-Fold Cross-Validation

Training on a smaller dataset (30 images) but with 10 epochs showed that even reduced data can achieve reasonable performance with sufficient training iterations.

| Fold | Whole Tumor Dice | Hausdorff Distance (mm) | Epoch |
|------|-----------------|------------------------|-------|
| 1    | 0.8488          | 2.9350                 | 10    |
| 2    | 0.7808          | 5.0951                 | 10    |
| 3    | 0.8620          | 3.6684                 | 10    |
| 4    | 0.8140          | 2.6140                 | 10    |
| 5    | 0.8259          | 2.8305                 | 10    |
| **Average** | **0.8263** | **3.4286** | **10** |

**Key Observations:**
- Achieved similar average Dice score (0.8263) as the 50-image configuration
- Better boundary alignment overall with lower average HD95
- Most consistent HD95 values, suggesting stable boundary predictions
- Demonstrates that more training iterations can partially compensate for smaller dataset size

**Limitations:**
- Fold 2 showed higher HD95 (5.10), indicating inconsistent boundary matching across folds
- False positives still present, particularly in extracranial regions

### Configuration 4: 48 Images, 5 Epochs, 5-Fold Cross-Validation

Training with more images but fewer epochs showed the trade-off between dataset size and training duration.

| Fold | Whole Tumor Dice | Epoch | Hausdorff Distance (mm) |
|------|-----------------|-------|------------------------|
| 1    | 0.8419          | 5     | 4.1451                 |
| 2    | 0.8423          | 5     | 3.9402                 |
| 3    | 0.7434          | 5     | 7.8494                 |
| 4    | 0.8205          | 5     | 2.6861                 |
| 5    | 0.8420          | 5     | 3.6170                 |
| **Average** | **0.8180** | **5** | **4.4476** |

**Key Observations:**
- Good average Dice score (0.8180) but lower than 10-epoch configurations
- More images provided some benefit, but fewer training epochs limited learning
- Highly variable performance—Fold 3 performed poorly (Dice: 0.7434, HD95: 7.85)
- Better boundary alignment in Folds 4 and 5, suggesting certain data splits were more challenging

**Limitations:**
- Insufficient epochs prevented the model from fully learning tumor boundaries
- High HD95 variance indicates instability in boundary prediction across folds
- Still exhibited false positives in the segmentation output

## Performance Summary and Comparison

Based on all four configurations tested:

| Configuration | Avg Dice | Avg HD95 | Best Performance | Issues |
|---|---|---|---|---|
| 50 img, 10 epochs | 0.8275 | 3.50 | ✓ Best Overall | Overfitting on ET |
| 50 img, 3 epochs | 0.7823 | 5.74 | ✗ Underfitting | High variance, Fold 3 failed |
| 30 img, 10 epochs | 0.8263 | 3.43 | ✓ Nearly as good | Fewer images but stable |
| 48 img, 5 epochs | 0.8180 | 4.45 | Moderate | Too few epochs |

**Key Findings:**

The 50-image, 10-epoch configuration emerged as the best choice, achieving the highest Dice score (0.8275) with reasonable boundary alignment (3.50 mm HD95). Training with 10 epochs provided sufficient iterations for the model to learn meaningful features, while 50 images contained enough diversity to avoid severe overfitting despite representing only 10% of the full BraTS dataset.

## What Worked Well and What Didn't

### Strengths of This Approach

The model successfully segments the main tumor regions, and different tumor sub-regions were clearly distinguishable in the output. The 5-fold cross-validation demonstrated stable performance across different data splits, indicating the model wasn't simply memorizing specific training images. The architecture's skip connections effectively preserved fine spatial details, particularly visible in the tumor core segmentation. U-Net's encoder-decoder structure proved well-suited for this task even with limited data.

### Areas Needing Improvement

Enhancing Tumor (ET) Dice scores were consistently 1.0 across configurations, which is unrealistically high and indicates the model wasn't generalizing well for this sub-region. The model generated false positives—segmenting regions outside the actual tumor, particularly in the extracranial region at the bottom of images. Tumor boundaries appeared somewhat blurry compared to the crisp ground truth masks, indicating the model could benefit from more advanced training techniques like Focal Loss or combined Dice + Cross-Entropy loss functions.

The limited training dataset (10% of BraTS) and GPU memory constraints significantly impacted what could be achieved. Some folds showed notably worse performance than others (Fold 3 sometimes dropped to 0.74 Dice while others reached 0.86), demonstrating that dataset diversity remained a limiting factor.


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
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
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
│       ├── imagesTr/                 # Training MRI volumes
│       └── labelsTr/                 # Training segmentation masks
├── models/
│   └── best_model.pth                # Saved trained model
└── README.md                          # This file
```

## Key Takeaways

This assignment demonstrated that U-Net is an effective architecture for medical image segmentation, even with limited training data. The main challenges encountered were memory constraints (limiting dataset size to 10%), overfitting on some sub-regions, and false positives in segmentation output. 

The most successful configuration used moderate training data (50 images) with 10 epochs of training, achieving a Dice score of 0.8275. This configuration struck a balance between learning sufficient tumor characteristics and avoiding severe overfitting. Reducing epochs to 3 caused significant underfitting and high variance across folds, while training with fewer images showed that increased epochs could partially compensate.

For practical clinical deployment, the next steps would involve training on the full dataset, implementing advanced loss functions like Focal Loss or combined Dice+CE Loss to reduce false positives, applying post-processing morphological operations, and likely migrating to 3D U-Net for improved spatial consistency. The current results provide a solid foundation and demonstrate the feasibility of this approach for medical image segmentation in resource-constrained environments.

## References

- Ronneberger et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- BraTS Dataset: https://www.med.upenn.edu/cbica/brats/
- PyTorch Documentation: https://pytorch.org/

**University**: University of Central Florida  
**Date**: 2025
