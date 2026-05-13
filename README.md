# Pneumonia Detection from Chest X-Ray Images

Binary image classification using deep learning to detect pneumonia from chest X-ray images.

**Course:** COMP 20037 - Artificial Intelligence and Deep Learning

---

## Overview

This project implements a deep learning model to classify chest X-ray images as either:
- **NORMAL**: Healthy lungs
- **PNEUMONIA**: Pneumonia detected

The model uses a convolutional neural network (CNN) architecture trained on the Kaggle Chest X-Ray Pneumonia dataset.

---

## Dataset

**Source:** [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Directory Structure
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/almalkydev/Pneumonia-Detection.git
cd Pneumonia-Detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download the Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract it to the project directory, or use Kaggle CLI:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip
```

### Step 4: Run the Application
```bash
python app.py
```

---

## Requirements

All dependencies are listed in `requirements.txt`:

- **TensorFlow** 2.15.0 - Deep learning framework
- **NumPy** 1.26.4 - Numerical computing
- **Matplotlib** 3.8.4 - Data visualization
- **scikit-learn** 1.4.2 - ML metrics and utilities
- **Seaborn** 0.13.2 - Statistical visualization
- **Pillow** 10.3.0 - Image processing

---

## Model Architecture

The project uses:
- Convolutional Neural Networks (CNN) with multiple layers
- Transfer learning with MobileNetV2
- Image augmentation for better generalization
- Early stopping to prevent overfitting

---

## Key Features

- ✅ Image preprocessing and normalization
- ✅ Data augmentation
- ✅ Model training with callbacks
- ✅ Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
- ✅ Confusion matrix visualization
- ✅ Classification reports

---

## Project Structure

```
Pneumonia-Detection/
├── app.py              # Main application and training script
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore file
└── README.md          # This file

```

---

## Usage Example

The main script (`app.py`) will:
1. Load and preprocess chest X-ray images
2. Create data generators for training and validation
3. Train the CNN model
4. Evaluate on test data
5. Generate classification reports and visualizations

---

## Performance Metrics

The model is evaluated using:
- **Accuracy** - Overall correctness
- **Precision** - True positives / All positives
- **Recall** - True positives / All actual positives
- **F1-Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Classification breakdown

---

## Notes

- Default image size: 128×128 pixels
- Batch size: 32 images per batch
- Maximum epochs: 20 (with early stopping)
- Random seed: 42 (for reproducibility)

---

## Author

Created for COMP 20037 - Artificial Intelligence and Deep Learning course

---

## License

This project uses the Kaggle Chest X-Ray Pneumonia dataset. Please refer to the dataset's license for usage terms.
