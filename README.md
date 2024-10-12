
# Road Segmentation Using Transfer Learning with VGG16 and Fully Convolutional Networks

## Table of Contents

- [Abstract](#abstract)
- [Introduction](#introduction)
- [Approach Overview](#approach-overview)
  - [Dataset Description](#dataset-description)
    - [Classes and Color Coding](#classes-and-color-coding)
  - [Dataset Preprocessing](#dataset-preprocessing)
    - [Data Separation and Organization](#data-separation-and-organization)
    - [Normalization and Resizing](#normalization-and-resizing)
    - [Data Augmentation](#data-augmentation)
  - [Model and Training Settings](#model-and-training-settings)
    - [Feature Extraction with VGG16](#feature-extraction-with-vgg16)
    - [Fully Convolutional Network (FCN) for Classification](#fully-convolutional-network-fcn-for-classification)
    - [Training Configuration](#training-configuration)
    - [Training Process](#training-process)
  - [Inference Pipeline](#inference-pipeline)
    - [Prediction Process](#prediction-process)
    - [Performance Metrics](#performance-metrics)
- [Results](#results)
  - [Training Performance](#training-performance)
  - [Evaluation Metrics](#evaluation-metrics)
    - [Pixel-wise Accuracy](#pixel-wise-accuracy)
    - [Dice Coefficient](#dice-coefficient)
    - [F1 Beta Score](#f1-beta-score)
  - [Inference Speed](#inference-speed)
  - [Visualization of Segmentation Results](#visualization-of-segmentation-results)
- [Conclusion, Discussions, and Future Scopes](#conclusion-discussions-and-future-scopes)
  - [Conclusion](#conclusion-1)
  - [Discussion](#discussion)
    - [Pros](#pros)
    - [Cons](#cons)
  - [Future Scopes](#future-scopes)
- [Appendix](#appendix)
  - [Project File Structure](#project-file-structure)
  - [Detailed Function Descriptions](#detailed-function-descriptions)
- [References](#references)

## Abstract

This report presents a comprehensive approach to segmenting Indian road images into 27 distinct classes using transfer learning with the VGG16 network and Fully Convolutional Networks (FCN). The dataset comprises images of Indian roads, each annotated with pixel-wise labels corresponding to various road elements such as roads, sidewalks, vehicles, and traffic signs. The segmentation task leverages a pre-trained VGG16 model for feature extraction, which is integrated with an FCN for classification. The report details the dataset preparation, model architecture, training procedure, evaluation metrics, and results, along with a discussion on the challenges faced and potential future improvements. GitHub repository: [https://github.com/who-else-but-arjun/image_segmentation](https://github.com/who-else-but-arjun/image_segmentation).

## Introduction

Road segmentation is a fundamental task in computer vision with significant applications in autonomous driving, traffic management, and urban planning. Accurate segmentation of road scenes enables machines to understand and navigate complex environments by distinguishing between various elements such as roads, sidewalks, vehicles, pedestrians, and traffic signs. This project focuses on segmenting images of Indian roads into 27 distinct classes, each representing a different road element, using a combination of transfer learning and Fully Convolutional Networks (FCN).

The structure of a binary mask is crucial for effective segmentation. It provides a clear and concise representation of the spatial distribution of different classes within an image. During the training phase, models learn to predict these masks by minimizing the difference between the predicted mask and the ground truth mask using loss functions such as cross-entropy or Dice loss. This process involves the model capturing intricate patterns and features from the input images that correspond to different road elements.

**Primary objectives of this project:**
- Develop an efficient image segmentation model capable of classifying each pixel of an Indian road image into one of 27 classes.
- Leverage transfer learning by utilizing a pre-trained VGG16 network for feature extraction, thereby improving the model's performance and reducing training time.
- Implement and evaluate the performance of the FCN in conjunction with VGG16 on the segmentation task.
- Assess the model's performance using metrics such as pixel-wise accuracy, Dice Coefficient, and F1 Beta Score.

## Approach Overview

The approach to road segmentation in this project involves several key stages: dataset preprocessing, model architecture design, training, and inference. Transfer learning is employed to utilize the powerful feature extraction capabilities of the VGG16 network, which is integrated with an FCN to perform pixel-wise classification.

### Dataset Description

The dataset used for this project consists of images depicting Indian roads, along with their corresponding segmentation masks. The dataset is organized into three main folders:

- **Train**: Contains the training images in JPEG format.
- **Labels**: Contains the corresponding segmentation masks in PNG format.
- **Test**: Contains images designated for testing the trained model.

#### Classes and Color Coding

There are 27 distinct classes representing various road elements. Each class is assigned a unique color for easy visualization and differentiation in the segmentation masks. The classes and their corresponding RGB color codes are detailed in the table below.

| **Class ID** | **Class Name**           | **Adjusted Mean Dice Coefficient** |
|--------------|--------------------------|-------------------------------------|
| 0            | Road                     | 0.93                                |
| 1            | Parking                  | NaN                                 |
| 2            | Sidewalk                 | 0.96                                |
| 3            | Rail Track               | 0.025                               |
| 4            | Person                   | 0.85                                |
| 5            | Rider                    | 0.835                               |
| 6            | Motorcycle               | 0.28                                |
| 7            | Bicycle                  | NaN                                 |
| 8            | Auto Rickshaw            | 0.265                               |
| 9            | Car                      | 0.31                                |
| 10           | Truck                    | 0.25                                |
| 11           | Bus                      | 0.855                               |
| 12           | Caravan                  | 0.995                               |
| 13           | Curb                     | 0.05                                |
| 14           | Wall                     | 0.465                               |
| 15           | Fence                    | 0.665                               |
| 16           | Guard Rail               | 0.935                               |
| 17           | Billboard                | 0.214                               |
| 18           | Traffic Sign             | 0.9300                              |
| 19           | Traffic Light            | NaN                                 |
| 20           | Pole                     | 0.0816                              |
| 21           | obs-str-bar-fallback     | NaN                                 |
| 22           | Building                 | 0.2204                              |
| 23           | Bridge                   | NaN                                 |
| 24           | Vegetation               | 0.860                               |
| 25           | Sky                      | 0.885                               |
| 255          | Unlabeled                | NaN                                 |

### Dataset Preprocessing

Effective preprocessing is crucial for ensuring that the model receives high-quality input data. The preprocessing steps undertaken include data separation, normalization, resizing, and augmentation.

#### Data Separation and Organization

The `dataset.py` script is responsible for organizing the dataset by separating images and labels into their respective folders. It ensures that each image in the `train/images` folder has a corresponding label in the `train/labels` folder, maintaining the same order and naming convention with different file formats (JPEG for images and PNG for labels).

#### Normalization and Resizing

All images are normalized to have pixel values within a standard range, typically [0, 1], to facilitate faster convergence during training. Additionally, images are resized to uniform dimensions to maintain consistency across batches. Two different sizes were experimented with:

- **224 × 312 pixels**
- **416 × 608 pixels**

#### Data Augmentation

To increase the diversity of the training data and prevent overfitting, several augmentation techniques were applied using the `augmentation.py` script:

- **Flipping**: Horizontal and vertical flips to simulate different camera angles.
- **Rotating**: Random rotations to account for varied perspectives.
- **Blurring**: Applying Gaussian blur to mimic motion or focus variations.
- **Contrast and Brightness Adjustments**: Varying contrast and brightness to enhance model robustness against lighting changes.

### Model and Training Settings

The model architecture combines a pre-trained VGG16 network with a Fully Convolutional Network (FCN) for pixel-wise classification. Transfer learning is utilized to leverage the feature extraction capabilities of VGG16, which is fine-tuned during training to adapt to the specific segmentation task.

#### Feature Extraction with VGG16

The VGG16 network, pre-trained on the ImageNet dataset, serves as the encoder in the FCN architecture. The fully connected layers of VGG16 are removed, and the convolutional layers are retained to extract hierarchical features from the input images. This pre-trained network provides a robust foundation for feature extraction, reducing the need for extensive training from scratch.

#### Fully Convolutional Network (FCN) for Classification

The FCN replaces the fully connected layers of traditional CNNs with convolutional layers that output spatial maps, enabling pixel-wise classification. The `model.py` script defines the `fcn_8_vgg` function, which constructs the FCN-8 architecture by integrating the VGG16 encoder with upsampling layers and skip connections for finer segmentation details. This architecture allows the model to retain spatial information while performing deep feature extraction.

#### Training Configuration

The training process was orchestrated using the `train.py` script with the following settings:

- **Batch Size**: Initially set to 32, later reduced to 16 to accommodate hardware limitations.
- **Image Sizes**: 224 × 312 pixels and 416 × 608 pixels.
- **Epochs**: 5 epochs due to computational constraints.
- **Optimizer**: Adam optimizer with default learning rate settings.
- **Loss Function**: Categorical Cross-Entropy loss.
- **Data Split**: 80% training and 20% validation split.
- **Hardware**: Training was conducted on a CPU (Intel i5 13th Gen), resulting in longer training times.

#### Training Process

The training process involves the following steps:

1. **Data Loading**: The `dataset.py` script separates images and labels into training and validation sets.
2. **Batch Generation**: The `functions.py` script handles batch generation, loading images and labels, applying normalization, resizing, and augmentation.
3. **Model Initialization**: The `fcn_8_vgg` model is initialized with the specified number of classes and input dimensions.
4. **Compilation**: The model is compiled with the Adam optimizer and categorical cross-entropy loss.
5. **Training Loop**: The `train.py` script runs the training loop for the defined number of epochs, adjusting the batch size as necessary.
6. **Checkpointing**: Model weights are saved periodically to allow for recovery and evaluation.

Training on a CPU resulted in a total training time of approximately 10 hours for 5 epochs. The model achieved a pixel-wise accuracy of 75%, indicating that 75% of the pixels were correctly classified.

### Inference Pipeline

The inference pipeline involves loading the trained model and applying it to test images to generate segmented outputs.

#### Prediction Process

The `predict.py` script facilitates the prediction of segmentation masks on test images. The steps involved are:

1. **Loading the Model**: The saved model weights are loaded.
2. **Preprocessing**: Test images are preprocessed in the same manner as training images, including normalization and resizing.
3. **Prediction**: The model predicts segmentation masks for the input images.
4. **Post-processing**: The binary masks are converted into color-coded segmented images using the predefined class colors.
5. **Saving Results**: The segmented images are saved in the designated output directory.

#### Performance Metrics

The inference pipeline achieved an average prediction time of approximately 300 milliseconds per image. The model size was optimized to ensure efficient deployment and quick inference times.

## Results

### Training Performance

The model was trained for five epochs, achieving a pixel-wise accuracy of 75% on the validation set. The training process was constrained by the use of a CPU, which limited the number of epochs and increased the overall training time.

### Evaluation Metrics

#### Pixel-wise Accuracy

Pixel-wise accuracy measures the proportion of correctly classified pixels out of the total number of pixels. An accuracy of 75% indicates that three-quarters of the pixels in the validation set were correctly classified.

#### Dice Coefficient

The Dice Coefficient is a statistical measure of similarity between the predicted segmentation and the ground truth. It is defined as:

\[
\text{Dice Coefficient} = \frac{2 \times TP}{2 \times TP + FP + FN}
\]

Where:
- \( TP \) = True Positives
- \( FP \) = False Positives
- \( FN \) = False Negatives

The mean Dice Coefficient across all classes was calculated to assess the model's ability to accurately segment different road elements. The mean Dice Coefficient was found to be **0.5097**.

However, some classes yielded NaN values for the Dice Coefficient. This occurrence is attributed to the absence of true positives, false positives, and false negatives for those specific classes, leading to division by zero in the Dice formula. Consequently, careful interpretation of the Dice Coefficient is necessary, particularly in cases where class imbalance exists.

#### F1 Beta Score

The F1 Beta Score is the harmonic mean of precision and recall, providing a balance between the two. It is particularly useful for evaluating segmentation tasks where class imbalance may exist.

\[
\text{F1 Beta Score} = \frac{(1 + \beta^2) \times \text{Precision} \times \text{Recall}}{(\beta^2 \times \text{Precision}) + \text{Recall}}
\]

In this project, the F1 Beta Score was calculated for each class to evaluate the balance between precision and recall in the segmentation results.

### Inference Speed

The model demonstrated an average inference time of 300 milliseconds per image, making it suitable for real-time applications where rapid segmentation is required.

### Visualization of Segmentation Results

Segmented images were generated by mapping the predicted binary masks to their corresponding class colors. Below is an example of the original image alongside its segmented output:

![Sample Predicted Images of Segmentation](images/segmentation_result.png)

*Figure: Sample Predicted Images of Segmentation*

## Conclusion, Discussions, and Future Scopes

### Conclusion

This project successfully implemented an image segmentation model for Indian road images using transfer learning with VGG16 and a Fully Convolutional Network (FCN). The model achieved a pixel-wise accuracy of 75% after five epochs of training on a CPU. The approach demonstrated the effectiveness of leveraging pre-trained networks for feature extraction in segmentation tasks, enabling accurate classification of various road elements despite computational constraints.

### Discussion

#### Pros

- **Transfer Learning**: Utilizing a pre-trained VGG16 model significantly enhanced feature extraction capabilities, leading to better segmentation performance with limited training data.
- **Data Augmentation**: Applying augmentation techniques improved the model's generalization by introducing variability in the training data.

#### Cons

- **Hardware Limitations**: Training on a CPU resulted in prolonged training times and restricted the number of epochs, potentially limiting the model's performance.
- **Limited Epochs**: Due to computational constraints, only five epochs were feasible, which may not be sufficient for the model to fully converge.
- **Class Imbalance**: Some classes had fewer examples, leading to lower performance metrics for those classes.

### Future Scopes

- **Hardware Upgrade**: Transitioning to GPU-based training would allow for faster training times and the ability to train for more epochs, potentially improving model accuracy.
- **Model Enhancements**: Exploring more advanced architectures such as U-Net, DeepLab, or Attention-UNet could lead to better segmentation performance.
- **Expanding the Dataset**: Increasing the size and diversity of the dataset would help the model generalize better to different road conditions and environments.
- **Real-time Deployment**: Optimizing the model for real-time applications by reducing inference time and model size without compromising accuracy.
- **Class Balancing**: Implementing techniques to address class imbalance, such as oversampling minority classes or using class weighting in the loss function, could improve performance for underrepresented classes.

## Appendix

### Project File Structure

The project is organized into several directories and scripts, each responsible for different aspects of the segmentation pipeline.

```plaintext
project/
│
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       └── images/
│
├── models/
│   ├── VGG16.py
│   ├── model.py
│   └── utils.py
│
├── scripts/
│   ├── train.py
│   ├── predict.py
│   ├── functions.py
│   ├── augmentation.py
│   ├── dataset.py
│   └── evaluation.ipynb
│
├── checkpoints/
│   └── model_weights.h5
│
├── helper.py
└── README.md
```

### Detailed Function Descriptions

#### VGG16.py

- **Functionality**: Implements the VGG16 network architecture for feature extraction.
- **Key Components**:
  - Loading pre-trained weights from ImageNet.
  - Modifying the architecture to remove fully connected layers.
  - Outputting feature maps for the FCN.

#### model.py

- **Functionality**: Defines the FCN architecture by integrating VGG16 with additional convolutional layers.
- **Key Components**:
  - `fcn_8_vgg` function: Constructs the FCN-8 architecture, adding upsampling layers and skip connections for finer segmentation.

#### utils.py

- **Functionality**: Provides utility functions for building and processing the segmentation model.
- **Key Components**:
  - Merging feature maps from different layers.
  - Implementing upsampling operations to match the input image size.

#### train.py

- **Functionality**: Handles the training loop for the segmentation model.
- **Key Components**:
  - Loading training and validation data.
  - Compiling the model with appropriate loss functions and optimizers.
  - Saving model checkpoints.
  - Monitoring training progress.

#### predict.py

- **Functionality**: Facilitates the prediction and visualization of segmentation masks on test data.
- **Key Components**:
  - Loading the trained model.
  - Predicting segmentation masks for new images.
  - Mapping binary masks to colored segmented images based on class IDs and colors.

#### functions.py

- **Functionality**: Contains essential functions for data handling and batch generation.
- **Key Components**:
  - Loading and preprocessing images and labels.
  - Implementing the batch generator for training.

#### augmentation.py

- **Functionality**: Implements data augmentation techniques to enhance training data diversity.
- **Key Components**:
  - Functions for flipping, rotating, blurring, and adjusting contrast and brightness of images.

#### dataset.py

- **Functionality**: Organizes and prepares the dataset by separating images and labels.
- **Key Components**:
  - Extracting image and label pairs.
  - Ensuring correspondence between training and validation sets.

## References

- Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2015.
- Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." *arXiv preprint arXiv:1409.1556* (2014).
- Chen, Liang-Chieh, et al. "Semantic image segmentation with deep convolutional nets and fully connected CRFs." *arXiv preprint arXiv:1412.7062* (2014).
- Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *International Conference on Medical Image Computing and Computer-Assisted Intervention*. Springer, Cham, 2015.

