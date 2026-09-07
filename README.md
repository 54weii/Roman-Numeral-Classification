# Roman Numeral Classification with Data-Centric AI

A computer vision course project focused on improving handwritten Roman numeral classification through **data quality analysis, data cleaning, and image augmentation**.

Instead of modifying the model architecture, the main objective of this project was to improve model performance by improving the training dataset under a fixed evaluation pipeline.

---

## Project Overview

The task is to classify handwritten Roman numerals into 10 classes:

`I, II, III, IV, V, VI, VII, VIII, IX, X`

The original dataset contained approximately 4,400 handwritten Roman numeral images.

Since the model architecture was fixed by the course benchmark, our work mainly focused on improving the **data pipeline**, including:

- Data quality inspection
- Duplicate and low-quality sample detection
- Data cleaning
- Image augmentation
- Training / validation split construction
- Model training and checkpointing
- Prediction and error analysis

The final course evaluation used **Macro F1 Score**.

---

## Workflow

```text
Raw Image Dataset
        │
        ▼
Data Quality Analysis
    (CleanVision)
        │
        ├── Near-duplicate detection
        ├── Low-information inspection
        ├── Image size / aspect-ratio inspection
        ├── Lighting inspection
        └── Grayscale inspection
        │
        ▼
Data Cleaning
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
Train / Validation Split
        │
        ▼
Fixed ResNet50-based Model
        │
        ▼
Model Evaluation
        │
        ▼
Prediction / Submission
```

---

## Data Quality Analysis

We used **CleanVision** to inspect image-quality issues in the training dataset.

The analysis included:

- Low-information images
- Near-duplicate images
- Odd image sizes
- Odd aspect ratios
- Lighting issues
- Grayscale images
- Blurry or dark images

Instead of automatically removing every image flagged by CleanVision, we manually inspected several categories because some detected issues were caused by the characteristics of handwritten Roman numeral images rather than actual data errors.

This iterative inspection helped us determine which samples should be removed and which should be retained.

---

## Data Augmentation

To improve the diversity of the training data, we used **Albumentations** with several image transformations.

```python
A.ShiftScaleRotate(
    shift_limit=0.20,
    scale_limit=0.20,
    rotate_limit=5,
    p=1
)

A.RandomBrightnessContrast(
    brightness_limit=0.20,
    contrast_limit=0.20,
    p=1
)

A.GaussNoise(
    var_limit=(30.0, 70.0),
    p=1
)

A.RandomResizedCrop(
    height=64,
    width=64,
    scale=(0.8, 1.0),
    ratio=(0.75, 1.33),
    p=1
)
```

These transformations were used to simulate variations in:

- Position
- Scale
- Rotation
- Lighting
- Image noise
- Handwriting appearance

---

## Final Dataset

The final dataset used by the fixed evaluation pipeline contained:

| Split | Number of Images |
| --- | ---: |
| Training | 11,024 |
| Validation | 963 |
| Testing | 500 |
| Number of Classes | 10 |

The course required the combined training and validation dataset to contain fewer than **12,000 images**.

---

## Model

The model architecture was fixed by the course benchmark and was not modified during the final evaluation.

The evaluation model used a truncated **ResNet50-based backbone** implemented with KerasCV.

### Model Configuration

- Input size: `32 × 32 × 3`
- Backbone: ResNet50-based feature extractor
- Global Average Pooling
- Dense output layer with 10 classes
- Optimizer: Adam
- Learning rate: `1e-4`
- Loss: Categorical Cross-Entropy
- Training epochs: 75
- Model checkpoint based on validation accuracy
- Learning-rate reduction on plateau

The fixed-model constraint shifted the focus of this project from model architecture design toward **data-centric model improvement**.

---

## Results

We compared several versions of the data-processing pipeline.

| Experiment | Evaluation Score |
| --- | ---: |
| Original baseline | 0.6163 |
| Initial processing approach | 0.6886 |
| CleanVision + data augmentation | 0.7304 |
| Final pipeline | **0.7697** |

### Improvement

```text
0.6163  →  0.7697
```

The final pipeline improved the course evaluation score by approximately **0.153**, showing that dataset quality and augmentation had a substantial impact even when the model architecture was fixed.

---

## Experimental Exploration

Besides the final pipeline, we also investigated several alternative approaches during development.

These experiments included:

- Confidence-based sample inspection
- Class-specific CNN models
- Additional data-quality filtering
- Different augmentation strategies
- Removal of low-information samples
- Inspection of potentially mislabeled samples

Not every experiment was included in the final pipeline, but these trials helped us better understand the dataset and guided later improvements.

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
│   └── final_presentation.pdf
│
├── README.md
└── requirements.txt
```

### `notebooks/main_experiment.ipynb`

Contains the main workflow used for the final project, including:

- Data quality analysis
- Data cleaning
- Data augmentation
- Dataset construction
- Fixed-model training
- Prediction and submission generation

### `notebooks/exploratory_experiments.ipynb`

Contains additional experiments performed during development, including alternative preprocessing and model-analysis approaches.

### `models/best_model.weights.h5`

Saved weights from the best model checkpoint based on validation accuracy.

### `report/final_presentation.pdf`

Final project presentation containing the complete experimental process and project results.

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

This project demonstrated that model performance is not determined only by model architecture.

Under a fixed model setting, we improved performance by focusing on:

1. Identifying data-quality problems
2. Inspecting problematic samples instead of blindly removing them
3. Increasing data diversity through augmentation
4. Iteratively comparing different preprocessing strategies

The project provided practical experience with a complete computer vision workflow from raw data inspection to model evaluation.

---

## Limitations

This project was developed under a course-specific setting with:

- A fixed model architecture
- A dataset-size constraint
- A predefined evaluation pipeline

Therefore, the final score should not be interpreted as state-of-the-art performance for Roman numeral recognition.

The original notebooks also preserve parts of the experimental development process and contain environment-specific file paths that may need to be modified before running the code on another machine.

---

## Future Improvements

Possible future improvements include:

- Refactoring preprocessing code into reusable Python modules
- Adding per-class Precision, Recall, and F1-score evaluation
- Generating a confusion matrix
- Improving class-imbalance handling
- Comparing augmentation strategies systematically
- Creating a standalone inference script
- Building a simple API for image classification
- Deploying the model as a lightweight application

---

## Project Background

This project was developed collaboratively as a final project for the course:

**Practical and Innovative Analytics in Data Science**

The repository preserves both the final implementation and selected exploratory experiments to document the development process and the effect of data-centric improvements on model performance.
