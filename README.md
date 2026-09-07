# Roman Numeral Classification with Data-Centric AI

A computer vision course project focused on improving handwritten Roman numeral classification through **automated data quality analysis, data cleaning, and image augmentation**.

The project explores how improving the quality and diversity of training data can enhance classification performance for handwritten Roman numerals.

---

## Project Overview

The task is to classify handwritten Roman numerals into 10 classes:

`I, II, III, IV, V, VI, VII, VIII, IX, X`

The original dataset contained approximately **4,400 handwritten images** collected from a Kaggle course dataset.

The project focused primarily on improving the training data through:

- Automated image quality inspection
- Detection of mislabeled and near-duplicate samples
- Low-information image filtering
- Image augmentation
- Dataset balancing
- CNN-based exploratory data diagnosis
- ResNet50-based model training and evaluation

The course required the final training and validation dataset to contain fewer than **12,000 images**, and model performance was evaluated using **F1-score**.

---

## Workflow

```text
Raw Image Dataset
        │
        ▼
Data Quality Analysis
    (CleanVision)
        │
        ├── Odd aspect-ratio detection
        ├── Near-duplicate detection
        ├── Grayscale processing
        ├── Lighting inspection
        ├── Black / white image detection
        └── Low-information inspection
        │
        ▼
Automated Data Cleaning
        │
        ▼
Low-Information Filtering
(Custom Image Features)
        │
        ▼
Data Augmentation
   (Albumentations)
        │
        ├── Shift / Scale / Rotation
        ├── Brightness / Contrast
        ├── Gaussian Noise
        └── Random Resized Crop
        │
        ▼
Training Dataset Optimization
        │
        ▼
ResNet50-based Training
        │
        ▼
F1-score Evaluation
```

---

## Data Quality Analysis

We used **CleanVision** to automatically identify potential image-quality problems in the dataset.

The preprocessing pipeline included several stages.

### 1. Odd Aspect Ratio

Images with abnormal width-to-height ratios were detected and removed.

- Removed: **1 image**

### 2. Near-Duplicate Detection

CleanVision was used to identify visually similar samples.

A custom noise-based comparison method was then used to determine which image should be removed from each near-duplicate pair.

- Removed: **44 near-duplicate images**
- Among them, **42 were found to have incorrect labels**

After additional lighting processing, another set of near-duplicate images was detected:

- Removed: **7 additional images**
- All 7 contained incorrect labels

Overall, this stage removed **51 near-duplicate samples**, of which **49 were incorrectly labeled**.

### 3. Grayscale Processing

All images were converted to grayscale to provide a more consistent representation for handwritten numeral recognition.

### 4. Lighting Processing

Images with abnormal lighting conditions were adjusted to improve consistency across the dataset.

### 5. Black / White Image Detection

Images containing only black or white pixels were treated as invalid samples.

- Removed: **1 completely blank image**

---

## Low-Information Image Filtering

One major challenge was handling images containing very little useful information.

Initial attempts included:

- Contrast enhancement
- Image cropping
- Image resizing
- Direct CleanVision low-information filtering

However, these approaches removed too many potentially useful samples or introduced additional near-duplicate detections.

Instead, we developed a custom filtering approach based on image characteristics such as:

- Number of connected components
- Total foreground pixels
- Number of small connected regions

This approach was designed to detect images dominated by scattered dots, broken strokes, or extremely limited visual information.

The final filtering strategy removed:

**111 low-information or highly noisy images**

while preserving more representative handwritten numeral samples.

---

## Data Augmentation

After data cleaning, **Albumentations** was used to increase the diversity of the training dataset.

The goal was to maximize the amount of training data while remaining within the course limit of 12,000 images.

### Shift, Scale, and Rotation

```python
A.ShiftScaleRotate(
    shift_limit=0.20,
    scale_limit=0.20,
    rotate_limit=5
)
```

This transformation simulated variations in position, size, and writing angle.

### Random Brightness and Contrast

```python
A.RandomBrightnessContrast(
    brightness_limit=0.20,
    contrast_limit=0.20
)
```

This simulated different lighting and image contrast conditions.

### Gaussian Noise

```python
A.GaussNoise(
    var_limit=(30.0, 70.0)
)
```

Noise was added to improve robustness against low-quality or noisy images.

### Random Resized Crop

```python
A.RandomResizedCrop(
    height=64,
    width=64,
    scale=(0.8, 1.0),
    ratio=(0.75, 1.33)
)
```

This simulated differences in cropping, image scale, and handwriting position.

---

## Dataset Optimization

After data cleaning, approximately:

```text
3,203 images
```

remained in the cleaned dataset.

Through image augmentation, the dataset was expanded to:

```text
3,203 → 11,992 images
```

This allowed us to use nearly the maximum amount of training data permitted by the course while increasing image diversity.

---

## Model Evaluation

The final classification pipeline used a **ResNet50-based model** to evaluate the quality of the processed dataset.

The goal of the project was not only to train a classifier, but also to investigate how different data-processing strategies affected model performance.

We compared four major experimental settings.

| Method | Evaluation Score |
| --- | ---: |
| Original dataset without additional processing | 0.61634 |
| Exploratory Model 1 approach | 0.68864 |
| CleanVision cleaning + data augmentation | 0.73041 |
| CleanVision cleaning + incorrect-label filtering + augmentation | **0.76967** |

### Final Improvement

```text
0.61634 → 0.76967
```

The final data-processing strategy achieved the best performance.

This result suggests that improving dataset quality and diversity can substantially improve classification performance without relying only on changes to the final classifier.

---

## Exploratory CNN Models

In addition to the final CleanVision-based pipeline, we proposed and partially tested a multi-stage CNN-based data diagnosis framework.

These models were designed primarily for **data quality analysis rather than final classification**.

### Model 1 — Class-Specific Roman Numeral Diagnosis

A separate binary CNN classifier was proposed for each Roman numeral class.

The objective was to estimate a confidence score for each image and use it to identify potentially mislabeled samples.

Each sample was ranked according to its confidence level:

```text
Rank 1 → Highest-confidence samples
Rank 2
Rank 3
Rank 4
Rank 5 → Lowest-confidence samples
```

Low-confidence images could then be inspected as potential labeling or image-quality problems.

---

### Model 2 — Uppercase / Lowercase Confidence Correction

During Model 1 experiments, lowercase Roman numerals sometimes received lower confidence scores because uppercase samples dominated the dataset.

Model 2 was therefore proposed to distinguish uppercase and lowercase representations and reduce this imbalance.

The goal was to improve the fairness of confidence-based sample filtering.

---

### Model 3 — Unified Roman Numeral Diagnosis

Model 3 was designed to combine the outputs of the previous models and classify all Roman numerals from I to X.

The proposed model used:

- Categorical Cross-Entropy
- Adam optimizer
- Batch size of 32
- Early stopping
- Multi-class probability output

---

## Computational Limitation

The exploratory CNN framework was **not fully completed** due to computational limitations.

The complete design would have required:

```text
Model 1 : 10 class-specific models
Model 2 : 10 uppercase/lowercase models
Model 3 :  1 unified model
-------------------------------
Total   : 21 models
```

The original plan required considerably more training iterations, but only limited experiments could be completed within the available computing resources.

Therefore, the three-model framework should be regarded as an **exploratory research direction**, while the final reported result was obtained using the CleanVision-based data-cleaning and augmentation pipeline.

---

## Repository Structure

```text
roman-numeral-classification/
│
├── notebooks/
│   ├── main_experiment.ipynb
│   └── exploratory_experiments.ipynb
│
├── models/
│   └── best_model.weights.h5
│
├── report/
│   ├── final_report.pdf
│   └── final_presentation.pdf
│
├── README.md
└── requirements.txt
```

### `notebooks/main_experiment.ipynb`

Contains the main experimental pipeline, including:

- CleanVision analysis
- Data cleaning
- Low-information filtering
- Albumentations augmentation
- Dataset preparation
- Model training
- Prediction and evaluation

### `notebooks/exploratory_experiments.ipynb`

Contains exploratory experiments related to:

- Confidence-based image analysis
- Class-specific CNN models
- Label-error detection
- Uppercase / lowercase classification
- Alternative preprocessing strategies

### `models/best_model.weights.h5`

Saved weights from the best model checkpoint obtained during the project.

---

## Reports

For detailed methodology and experimental discussion:

- [Final Written Report](./report/final_report.pdf)
- [Final Presentation](./report/final_presentation.pdf)

The written report contains the complete description of the preprocessing experiments, exploratory model designs, results, limitations, and future improvements.

---

## Tech Stack

- Python
- TensorFlow
- Keras
- KerasCV
- CleanVision
- Albumentations
- NumPy
- Pandas
- Matplotlib
- Pillow
- OpenCV
- Jupyter Notebook

---

## Key Takeaways

This project provided practical experience with a **data-centric computer vision workflow**.

The main lessons from the project were:

1. Image quality can significantly affect downstream model performance.
2. Automated data-quality tools can help identify duplicate, noisy, and potentially mislabeled samples.
3. Automatically removing every detected anomaly is not always effective; filtering strategies need to consider the characteristics of the dataset.
4. Data augmentation can improve both dataset diversity and model robustness.
5. Iterative comparison of preprocessing strategies can provide substantial improvements even without redesigning the final classification model.

The final evaluation score improved from:

```text
0.61634 → 0.76967
```

through data cleaning, incorrect-label filtering, and augmentation.

---

## Limitations

This project was developed under several constraints:

- Training and validation data were limited to fewer than 12,000 images.
- Computing resources limited the number of CNN experiments that could be completed.
- The proposed 21-model diagnosis framework was only partially tested.
- The final result was evaluated under the course-specific dataset and evaluation setting.

Therefore, the result should not be interpreted as state-of-the-art Roman numeral recognition performance.

Instead, the primary contribution of this project is the exploration of **data quality improvement and data-centric optimization for computer vision**.

---

## Future Improvements

Future work could include:

- Completing the proposed multi-stage CNN diagnosis framework
- Improving positive and negative sample construction
- Systematically testing augmentation parameter combinations
- Applying Grid Search or Random Search to augmentation settings
- Designing class-specific augmentation strategies
- Adding per-class Precision, Recall, and F1-score analysis
- Generating confusion matrices for error analysis
- Refactoring preprocessing into reusable modules
- Building a standalone inference pipeline
- Deploying the classifier through a lightweight API

---

## Project Background

This project was developed collaboratively as a final project for the course:

**Practical and Innovative Analytics in Data Science**

The work focused on improving handwritten Roman numeral classification through automated data-quality analysis, data cleaning, image augmentation, and exploratory CNN-based data diagnosis.

The final reported score improved from **0.61634 to 0.76967**.
