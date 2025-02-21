from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import opensmile
import os
import numpy as np
import pandas as pd
import pywt
import torchaudio
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def read_data(csv_file, audio_dir):
    data = pd.read_csv(csv_file)
    X = []
    y = []
    patient_ids = []

    for index, row in data.iterrows():
        wav_file = os.path.join(audio_dir, row['path'])
        label = row['label']
        patient_id = row['path'].split('_')[0]

        X.append(wav_file)
        y.append(label)
        patient_ids.append(patient_id)

    return X, y, patient_ids


def extract_compare_features(audio_file, features_type):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )

    # Get the full list of feature names
    feature_list = smile.feature_names

    # Define the feature groups
    feature_groups = {
        'Energy': [],
        'Spectral': [],
        'VoiceQuality': [],
        'Prosodic': [],
        'Formant': [],
        'Temporal': [],
        'Other': []
    }

    # Categorize features into groups
    for feature in feature_list:
        if any(substring in feature for substring in ['loudness', 'zcr', 'energy']):
            feature_groups['Energy'].append(feature)
        elif any(substring in feature for substring in ['mfcc', 'centroid', 'flux', 'entropy', 'rolloff', 'slope']):
            feature_groups['Spectral'].append(feature)
        elif any(substring in feature for substring in ['hnr', 'jitter', 'shimmer', 'apq']):
            feature_groups['VoiceQuality'].append(feature)
        elif 'F0' in feature or 'voicing' in feature:
            feature_groups['Prosodic'].append(feature)
        elif 'formant' in feature:
            feature_groups['Formant'].append(feature)
        elif any(substring in feature for substring in ['dur', 'rate', 'pause']):
            feature_groups['Temporal'].append(feature)
        else:
            feature_groups['Other'].append(feature)

    if features_type == 'all':
        # Extract all features
        selected_features = feature_list
    elif features_type not in feature_groups:
        raise ValueError(
            f"Invalid features_type: {features_type}. Must be one of {list(feature_groups.keys())} or 'all'.")
    else:
        # Extract features from the specified group
        selected_features = feature_groups[features_type]

    # Extract all features
    features = smile.process_file(audio_file)

    # Filter features based on the selected group
    extracted_features = features[selected_features]

    return extracted_features


def extract_gemaps_features(audio_path, features_type):
    # Initialize OpenSMILE with GeMAPSv01b configuration
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    # Define feature groups
    feature_groups = {
        'frequency': [
            'F0semitoneFrom27.5Hz_sma3nz', 'jitterLocal_sma3nz', 'F1frequency_sma3nz',
            'F2frequency_sma3nz', 'F3frequency_sma3nz', 'F1bandwidth_sma3nz'],
        'energy': [
            'loudness_sma3', 'shimmerLocaldB_sma3nz'],
        'spectral': [
            'alphaRatio_sma3', 'hammarbergIndex_sma3', 'slope0 - 500_sma3', 'slope500 - 1500_sma3'],
        'temporal': [
            'HNRdBACF_sma3nz'],
    }

    if features_type == 'all':
        # Extract all features
        selected_features = [feature for group in feature_groups.values() for feature in group]
    elif features_type not in feature_groups:
        raise ValueError(f"Invalid features_type: {features_type}. Choose from {list(feature_groups.keys())} or 'all'.")
    else:
        # Extract features from the specified group
        selected_features = feature_groups[features_type]

    # Extract features
    features = smile.process_file(audio_path)
    available_features = features.columns.tolist()

    # Filter the selected features based on availability
    selected_features = [f for f in selected_features if f in available_features]

    if not selected_features:
        raise ValueError(f"No features from {features_type} group were found in the extracted features.")

    return features[selected_features]


def extract_wavelet_features(y, wavelet='db4', level=3):
    coeffs = pywt.wavedec(y, wavelet, level=level)
    features = []
    for coeff in coeffs:
        features.append(np.mean(coeff))
        features.append(np.var(coeff))
        features.append(np.min(coeff))
        features.append(np.max(coeff))
    return np.array(features)


def extract_features(X, features_lib, features_type):
    features = []

    for wav_file in X:
        try:
            print(f"Processing file: {wav_file}")

            # Load the audio data from the file
            # y, sr = librosa.load(wav_file, sr=None)
            y, sr = torchaudio.load(wav_file)
            print(f"Loaded file with {len(y)} samples at {sr} Hz sample rate.")

            # Ensure the file has content
            if len(y) == 0:
                print(f"Warning: Empty audio file {wav_file}")
                file_features = np.zeros(50)
            else:
                if features_lib == "gemaps":
                    # Extract frame-wise GeMAPS features
                    gemaps_features = extract_gemaps_features(wav_file, features_type)
                    gemaps_df = pd.DataFrame(gemaps_features)
                    file_features = aggregate_statistics(gemaps_df)

                elif features_lib == "compare":
                    # Extract frame-wise ComParE features
                    compare_features = extract_compare_features(wav_file, features_type)
                    compare_df = pd.DataFrame(compare_features)
                    file_features = aggregate_statistics(compare_df)

                else:
                    raise ValueError(
                        f"Unsupported features_lib: {features_lib}. Use 'gemaps', 'compare', or 'combined'.")

            # Append the aggregated features (one row per file)
            features.append(file_features)

        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            features.append(np.zeros(50))  # Ensure consistent feature size for errors

    # Convert to a NumPy array and ensure proper shape
    features_array = np.array([f for f in features if f.size == len(features[0])])  # Filter for correct shapes
    return features_array


def aggregate_statistics(df):
    aggregated_features = {
        'mean': df.mean(),
        'std': df.std(),
        'min': df.min(),
        'max': df.max(),
        'q1': df.quantile(0.25),
        'median': df.median(),
        'q3': df.quantile(0.75)
    }
    # Convert aggregated features to a DataFrame and flatten it to a single row
    aggregated_df = pd.DataFrame(aggregated_features).T
    flattened_features = aggregated_df.values.flatten()
    return flattened_features


""" Classifiers """


def classifier(name):
    if name == "svm":
        return SVC(probability=True)
    elif name == "knn":
        return KNeighborsClassifier()
    elif name == "dtc":
        return DecisionTreeClassifier()
    elif name == "xgb":
        return XGBClassifier(eval_metric='logloss')
    elif name == "cat":
        return CatBoostClassifier(verbose=0)  # Suppress verbose output for CatBoost
    else:
        raise ValueError("Unsupported classifier name. Choose from 'SVM', 'KNN', 'DTC', 'XGB', or 'CAT'.")


def hyperparameter_tuning(model, param_grid, X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV with AUC as the scoring metric."""
    # Use 'roc_auc' for binary classification or 'roc_auc_ovr' for multiclass
    scoring_metric = 'roc_auc' if len(set(y_train)) == 2 else 'roc_auc_ovr'

    # Initialize GridSearchCV with AUC scoring
    search = GridSearchCV(model, param_grid, scoring=scoring_metric, cv=3, n_jobs=1, verbose=1)

    # Fit the model to find the best parameters
    search.fit(X_train, y_train)

    print(f"Best Parameters: {search.best_params_}")
    print(f"Best AUC Score: {search.best_score_:.4f}")

    return search.best_estimator_


def train_and_evaluate(X_train, X_test, X_val, y_train, y_test, y_val, model, epochs=10, param_grid=None):
    """Train the model with hyperparameter tuning and validation monitoring, then evaluate on test data."""
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    X_val = imputer.transform(X_val)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)

    # Hyperparameter tuning if a param_grid is provided
    if param_grid is not None:
        model = hyperparameter_tuning(model, param_grid, X_train, y_train)

    # Training with custom epochs loop if the model supports partial_fit
    for epoch in range(epochs):
        if hasattr(model, 'partial_fit'):
            model.partial_fit(X_train, y_train, classes=np.unique(y_train))
        else:
            model.fit(X_train, y_train)

        # Validation on the validation set
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}\n")

    # Final evaluation on the test set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # AUC calculation
    if len(set(y_test)) == 2:  # Binary classification
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_proba)
    else:
        lb = LabelBinarizer()
        y_test_binarized = lb.fit_transform(y_test)
        y_test_proba = model.predict_proba(X_test)
        test_auc = roc_auc_score(y_test_binarized, y_test_proba, multi_class='ovr')
        auc_per_class = roc_auc_score(y_test_binarized, y_test_proba, multi_class='ovr', average=None)

    # Output results
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Overall AUC: {test_auc:.4f}")

    if len(set(y_test)) > 2:
        for class_index, class_auc in enumerate(auc_per_class):
            print(f"AUC for class {lb.classes_[class_index]} vs rest: {class_auc:.4f}")

    return model, val_accuracy, val_f1, test_accuracy, test_f1, conf_matrix


# Define parameter grids for tuning (customize as needed for each model)
param_grids = {
    "SVM": {
        "kernel": ["linear", "rbf", "poly"],
        "C": [0.1, 1, 10],
        "degree": [2, 3, 4]  # Only for 'poly' kernel
    },
    "XGB": {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [50, 100, 200]
    },
    "CAT": {
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1, 0.2],
        "iterations": [100, 200, 300]
    }
}
