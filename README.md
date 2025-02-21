# Benchmarking machine learning for bowel sound pattern classification – from tabular features to pretrained models

This is the official code repository associated with the paper:

📄 **Benchmarking machine learning for bowel sound pattern classification – from tabular features to pretrained models**
✍️ *Zahra Mansour, Verena Uslar, Dirk Weyhe, Danilo Hollosi and Nils Strodthoff.*
📅 **Currently under review**

This repository contains a **Bowel sounds classification pipeline** that supports both **Deep Learning** and **Machine Learning** models for audio classification. The pipeline is built using **PyTorch, Hugging Face Transformers, and Scikit-Learn**.


##  **Features**
 **Supports multiple model architectures**:
   - **Deep Learning Models:** VGG, ResNet, AlexNet, CNN-LSTM
   - **Pre-trained models:** Wav2Vec, HuBERT
   - **Machine Learning Models:** SVM, XGBoost, KNN, Decision Tree (DTC), CatBoost

 **Automatic feature extraction**:
   - Extracts **spectrogram, log-mel, MFCC, or raw waveform** for deep learning.
   - Supports **GeMAPS & ComParE** feature extraction for machine learning.

 **Stratified Data Splitting**:
   - Ensures class distribution remains balanced across **train**, **validation**, and **test** sets.

 **Configurable settings using `config.yaml`**:
   - **Easily change dataset paths, model type, and hyperparameters** without modifying the code.

 **Automatic Model Training & Evaluation**:
   - Computes **accuracy, F1-score, confusion matrix, and AUC (Area Under Curve)** for evaluation.

---

## **Usage**
- **Install dependencies** 
```sh
pip install torch torchvision torchaudio transformers pandas scikit-learn pyyaml
```
- **For Machine and deep learning models:** after choosing the features and models by updating the config file, Simply run:
```sh
python main.py
```
- **For finetuning pretrained models:**
```sh
python pre_trained_Wav2Vec.py
python pre_trained_HuBERT.py
```
This will:
- Load the dataset from config.yaml
- Split the dataset into train/validation/test sets

- Extract features
- Train the specified model
- Evaluate model performance

## **Data Preparation**

Your dataset should be in **CSV format** with the following columns:

| Column Name  | Description |
|-------------|------------|
| `path`      | File path to the audio file in .wav formate |
| `label`     | Class label for the audio sample (for bowel sound patterns: SB, MB, CRS, HS, and Silence period labelled NONE) |
| `patent_id` | Identifier for subject grouping  |

### **Example CSV File**
Below is an example of how your dataset should look:

#### **Bowel Sound patterns Classification (`BS_segments.csv`)**
| path         | label  | patent_id |
|-------------|--------|-----------|
| /101SG_PT_segment_1.wav | SB| 101 |
| /102SG_PT_segment_1.wav | MB | 102 |
| /103SG_PT_segment_1.wav| NONE | 103 |
