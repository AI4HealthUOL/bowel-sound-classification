import torch
import yaml
import torchvision.transforms as transforms
import pandas as pd
from data_loader import load_and_split_data, get_dataloaders
from DLmodels import create_model, train_model, evaluate_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from ML_models import read_data, extract_features, train_and_evaluate, param_grids
from sklearn.preprocessing import StandardScaler
from Split_the_data import split_wave_data_with_stratified_groups


# Load Configuration File
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


config = load_config()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset path and parameters from config
csv_file = config["dataset_csv"]
audio_dir = config["audio_dir"]
subset_ratios = [config["train_ratio"], config["val_ratio"], config["test_ratio"]]

# Load model type from config
model_type = config["model_type"]

# 1. Split the data into train, test, and validation
train_csv, val_csv, test_csv, label_encoder = load_and_split_data(csv_file, subset_ratios)

## **For Deep Learning Models**
if model_type in ['vgg', 'resnet', 'alexnet', 'cnn-lstm']:
    # Define transforms for deep learning models
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # 2. Extract spectrogram features
    data_type = config["feature_type"]  # Use feature type from config
    dataloaders = get_dataloaders(train_csv, val_csv, test_csv, feature=data_type, transform=True)

    # 3. Build the deep learning model
    model = create_model(model_type, num_classes=len(label_encoder.classes_))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 4. Train the deep learning model
    model = train_model(model, criterion, optimizer, dataloaders, device, num_epochs=config["num_epochs"])

    # 5. Evaluate the deep learning model
    labels, preds, probs = evaluate_model(model, dataloaders['test'], device)

    # 6. Final Evaluation
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    conf_matrix = confusion_matrix(labels, preds)

    # Calculate overall AUC
    if model_type in ['vgg', 'resnet', 'alexnet', 'cnn-lstm']:
        overall_auc = roc_auc_score(labels, probs, multi_class='ovr')
        num_classes = probs.shape[1]
    else:
        overall_auc = roc_auc_score(labels, probs[:, 1])
        num_classes = probs.shape[1] if probs.ndim > 1 else 2

    # Calculate AUC for each class
    class_aucs = {}
    for class_idx in range(num_classes):
        binary_labels = (labels == class_idx).astype(int)
        if probs.ndim > 1:
            class_auc = roc_auc_score(binary_labels, probs[:, class_idx])
        else:
            class_auc = roc_auc_score(binary_labels, probs)
        class_aucs[f'Class {class_idx} AUC'] = class_auc

    # Print results
    print(f"Model Type: {model_type.upper()}")
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n {conf_matrix}')
    print(f'Overall AUC: {overall_auc:.4f}')
    for class_name, auc in class_aucs.items():
        print(f'{class_name}: {auc:.4f}')


## **For Machine Learning Models**
elif model_type in ['svm', 'knn', 'dtc', 'xgboost', 'catboost']:
    # 1. Read the data
    X, y, patient_ids = read_data(csv_file, audio_dir)
    df = pd.DataFrame({'path': X, 'label': y, 'patient_id': patient_ids})

    # 2. Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_wave_data_with_stratified_groups(csv_file)

    # 3. Extract the features
    Features_name = config["ml_feature_name"]  # Get feature type from config
    X_train_features = extract_features(X_train, Features_name, 'all')
    X_test_features = extract_features(X_test, Features_name, 'all')
    X_val_features = extract_features(X_val, Features_name, 'all')
    X_train_flat = [x.flatten() for x in X_train_features]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)

    # Train and evaluate the models
    trained_model, val_accuracy, val_f1, test_accuracy, test_f1, conf_matrix = train_and_evaluate(
        X_train_scaled, X_test_features, X_val_features,
        y_train, y_test, y_val, model_type,
        epochs=config["ml_epochs"], param_grid=param_grids
    )
