import torch
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, load_dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import torchaudio
from Straified_split import split_stratified
import os
import yaml

# Load Configuration File
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()

# Load and Preprocess Dataset
def preprocess_audio(batch):

    audio_path = os.path.join(config["audio_dir"], batch['path'])  # Append directory to path
    audio, sr = torchaudio.load(audio_path)
    batch['speech'] = audio
    return batch

# Load dataset from CSV
df = pd.read_csv(config["dataset_csv"])

# 2. Perform Stratified Split
X_train, y_train, X_val, y_val, X_test, y_test = split_stratified(df, subset_ratios=[0.7, 0.15, 0.15])


# Convert Pandas DataFrames to Hugging Face Dataset
def dataset_from_dataframe(train_df, val_df, test_df):
    return DatasetDict({
        'train': Dataset.from_pandas(pd.concat([train_df, y_train], axis=1)),
        'validation': Dataset.from_pandas(pd.concat([val_df, y_val], axis=1)),
        'test': Dataset.from_pandas(pd.concat([test_df, y_test], axis=1))
    })


dataset = dataset_from_dataframe(X_train, X_val, X_test)

# 2. Apply Audio Preprocessing
dataset = dataset.map(preprocess_audio)

# Initialize Feature Extractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding=True)


# 4. Extract Features
def extract_features(batch):

    batch['input_values'] = \
    feature_extractor(batch['speech'], sampling_rate=16000, return_tensors='pt', padding=True).input_values[0]
    return batch


dataset = dataset.map(extract_features)

# Assign Class Labels
unique_labels = list(set(dataset['train']['label']))
class_labels = ClassLabel(names=unique_labels)

for split in dataset:
    dataset[split] = dataset[split].cast_column('label', class_labels)

# Number of Classes
num_classes = class_labels.num_classes
print("Number of unique classes:", num_classes)

# 5. Load Pretrained Wav2Vec2 Model
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    'facebook/wav2vec2-base',
    num_labels=num_classes
)


# 6. Define Evaluation Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


def compute_auc_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    if probabilities.shape[1] == 2:  # Binary case
        positive_class_probs = probabilities[:, 1]
        overall_auc = roc_auc_score(labels, positive_class_probs)
        return {"overall_AUC": overall_auc}

    per_class_auc = {}
    for i in range(probabilities.shape[1]):
        auc = roc_auc_score(labels == i, probabilities[:, i])
        per_class_auc[f"AUC_class_{i}"] = auc

    overall_auc = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
    per_class_auc["overall_AUC"] = overall_auc
    return per_class_auc


# 7. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    save_steps=500,
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_AUC"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=feature_extractor,
    compute_metrics=compute_auc_metrics
)

# 8. Train Model
trainer.train()

# 9. Evaluate on Test Set
evaluation_results = trainer.evaluate(eval_dataset=dataset['test'])

# Save the Model
model.save_pretrained("./fine_tuned_model")

print("Evaluation results:", evaluation_results)
