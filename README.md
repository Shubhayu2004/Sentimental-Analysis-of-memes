# Sentimental Notebook

This project is a multimodal deep learning pipeline for sentiment and emotion analysis on memes, combining image and text features to predict multiple sentiment-related labels.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## Overview

This notebook builds a model that predicts four sentiment-related labels (`humour`, `sarcasm`, `offensive`, `motivational`) from memes using both their images and associated text. The workflow includes data cleaning, image preprocessing, text vectorization, model building (with transfer learning and LSTM), training, and evaluation.

---

## Dataset

- **Source:** `memotion_dataset_7k`
- **Files Used:**
  - `labels.csv` (contains image names, text, and sentiment labels)
  - `images/` (folder with meme images)

---

## Setup

### Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

### Installation

```sh
pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn
```

---

## Data Preprocessing

### 1. **Label Encoding**

All categorical sentiment labels are mapped to integers:

- `humour`: `not_funny`=0, `funny`=1, `very_funny`=2, `hilarious`=3
- `sarcasm`: `not_sarcastic`=0, `general`=1, `twisted_meaning`=2, `very_twisted`=3
- `offensive`: `not_offensive`=0, `slight`/`slightly_offensive`=1, `very_offensive`=2, `hateful_offensive`=3
- `motivational`: `not_motivational`=0, `motivational`=1
- `overall_sentiment`: `very_negative`=0, `negative`=1, `neutral`=2, `positive`=3, `very_positive`=4

Any stray values like `'slight'` or `'slightly_offensive'` are mapped to `1`.

### 2. **Missing Values**

- All rows with missing values are dropped.

### 3. **Image Preprocessing**

- Images are loaded, resized to 100x100, and normalized to [0, 1].
- Only images that exist are loaded; the DataFrame is filtered to match successfully loaded images.

### 4. **Text Preprocessing**

- Text is lowercased, numbers and punctuation are removed, and `.com` substrings are stripped.
- Text is vectorized using Keras `TextVectorization` with a vocabulary size of 100,000 and sequence length of 50.

### 5. **Data Splitting**

- Data is split into training and test sets (80/20), ensuring alignment between images, text, and labels.

---

## Model Architecture

### 1. **Image Model**

- Uses transfer learning with ResNet50 and VGG16 (pretrained on ImageNet, frozen).
- Outputs are concatenated and passed through additional layers.

### 2. **Text Model**

- Embedding layer followed by two Bidirectional LSTM layers and dense layers.

### 3. **Combined Model**

- Image and text features are concatenated.
- Dense layers and dropout for regularization.
- Final output: 4 regression heads (for each sentiment label).

---

## Training

- Loss: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)
- Callbacks: ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
- Data augmentation is applied to images (random flip and rotation).

---

## Evaluation

- Plots for training/validation loss and MAE.
- MAE is reported for each label.
- Classification reports and confusion matrices are generated for each label.

---

## Troubleshooting

- **Label dtype errors:**  
  Ensure all label columns are numeric. Any stray string values (like `'slight'`) are mapped to integers before conversion.
- **Image/text alignment:**  
  Only rows with successfully loaded images are used for training/testing.
- **Git LFS:**  
  If using Git LFS for images, ensure all images are tracked after running `git lfs track "*.jpg"` and re-add any untracked files.

---

## Acknowledgements

- [Memotion Dataset](https://www.kaggle.com/datasets/abhinavwalia95/memotion-dataset-7k)
- TensorFlow and Keras documentation

---

## Example Usage

```python
# To run the notebook, open sentimental_notebook.ipynb in VS Code or Jupyter and execute cells sequentially.
```

---