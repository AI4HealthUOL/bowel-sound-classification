import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torchvision.models as models


class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, num_classes=5):
        super(CNN_LSTM, self).__init__()

        # CNN layers (extract features from spectrogram)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((16, 50))
        )

        self.lstm_input_size = 32 * 16
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, width, channels * height)
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


def create_model(model_type, num_classes=5):
    if model_type == 'vgg':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        with torch.no_grad():
            model.features[0].weight = nn.Parameter(model.features[0].weight.mean(dim=1, keepdim=True))
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_type == 'resnet':
        model = models.resnet50(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            model.conv1.weight = nn.Parameter(model.conv1.weight.mean(dim=1, keepdim=True))
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        with torch.no_grad():
            model.features[0].weight = nn.Parameter(model.features[0].weight.mean(dim=1, keepdim=True))
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_type == 'cnn-lstm':
        model = CNN_LSTM(num_classes=num_classes)

    else:
        raise ValueError(f"Model {model_type} is not supported in this version")

    return model


def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=20):
    model.to(device)

    best_model_wts = None
    best_auc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_labels = []
            all_probs = []

            for inputs, labels, _ in tqdm(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)

                # Ensure input is in correct format for CNN-LSTM
                if isinstance(model, CNN_LSTM):
                    inputs = inputs.unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1) if outputs.size(1) > 1 else torch.sigmoid(outputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_labels.append(labels.detach().cpu())
                all_probs.append(probs.detach().cpu())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            all_labels = torch.cat(all_labels).numpy()
            all_probs = torch.cat(all_probs).numpy()

            if all_probs.shape[1] > 1:
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            else:
                auc = roc_auc_score(all_labels, all_probs[:, 0])

            print(f'{phase} Loss: {epoch_loss:.4f} AUC: {auc:.4f}')

            if phase == 'val' and auc > best_auc:
                best_auc = auc
                best_model_wts = model.state_dict()

    if best_model_wts:
        model.load_state_dict(best_model_wts)

    print(f'Best validation AUC: {best_auc:.4f}')
    return model


def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels, _ in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            if isinstance(model, CNN_LSTM):  # Ensure correct input format for CNN-LSTM
                inputs = inputs.unsqueeze(1)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)