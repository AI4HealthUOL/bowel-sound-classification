from Straified_split import split_stratified
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class PIUS(Dataset):
    def __init__(self, csv_file, feature='mel', transform=None, target_length=2000, use_log_mel=False):
        self.data = pd.read_csv(csv_file)
        self.data_type = feature.lower()
        self.transform = transform
        self.target_length = target_length
        self.use_log_mel = use_log_mel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wave_path = Path(self.data.iloc[idx]['path'])
        label = self.data.iloc[idx]['label']
        patent_id = self.data.iloc[idx]['patent_id']

        waveform, sample_rate = torchaudio.load(wave_path)
        waveform = self._pad_or_truncate(waveform)

        if self.data_type == 'original':
            data = waveform  # Return raw waveform instead of spectrogram
        else:
            data = self._extract_features(waveform, sample_rate, wave_path)

        if self.transform:
            data = self.transform(data)

        return data, label, patent_id

    def _pad_or_truncate(self, waveform):
        if waveform.shape[1] < self.target_length:
            padding = self.target_length - waveform.shape[1]
            return torch.nn.functional.pad(waveform, (0, padding), "constant", 0)
        return waveform[:, :self.target_length]

    def _extract_features(self, waveform, sample_rate, wave_path):
        if self.data_type == 'mel':
            spectrogram = T.MelSpectrogram(
                sample_rate=sample_rate, n_fft=512, hop_length=256, n_mels=64
            )(waveform)
            return torch.log1p(spectrogram) if self.use_log_mel else spectrogram
        elif self.data_type == 'mfcc':
            return T.MFCC(sample_rate=sample_rate, n_mfcc=13, melkwargs={"n_fft": 512, "n_mels": 64})(waveform)
        elif self.data_type == 'spectrogram':
            return T.Spectrogram()(waveform)
        elif self.data_type == 'log-mel':
            spectrogram = T.MelSpectrogram(
                sample_rate=sample_rate, n_fft=512, hop_length=256, n_mels=64
            )(waveform)
            return torch.log1p(spectrogram)
        else:
            raise ValueError(f"Invalid data type: {self.data_type}")



def load_and_split_data(csv_file, subset_ratios, filename="subset.txt"):
    df = pd.read_csv(csv_file)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    df.rename(columns={'New_Wav_File': 'path'}, inplace=True)

    X_train, y_train, X_val, y_val, X_test, y_test = split_stratified(
        df, subset_ratios=subset_ratios, filename=filename, col_subset="subset", col_index=None, col_label="label"
    )

    train_data, val_data, test_data = [
        pd.concat([X, y], axis=1) for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    ]

    for data, name in zip([train_data, val_data, test_data], ['train', 'val', 'test']):
        data[['path', 'label']].to_csv(f'{name}_data.csv', index=False)

    return 'train_data.csv', 'val_data.csv', 'test_data.csv', le


def get_dataloaders(train_csv, val_csv, test_csv, feature, transform, batch_size=16):
    datasets = {
        'train': PIUS(train_csv, feature=feature, transform=transform, target_length=2000),
        'val': PIUS(val_csv, feature=feature, transform=transform, target_length=2000),
        'test': PIUS(test_csv, feature=feature, transform=transform, target_length=2000)
    }
    return {k: DataLoader(v, batch_size=batch_size, shuffle=(k == 'train')) for k, v in datasets.items()}