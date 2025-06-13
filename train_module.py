import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
import mlflow
import mlflow.pytorch
import warnings
from typing import Dict, List, Tuple


def prepare_data(data_dir: str, batch_size: int = 4) -> Tuple[Dict, Dict, List[str]]:
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_path, data_transforms['train']),
        'val': datasets.ImageFolder(val_path, data_transforms['val'])
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4
        )
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print("\nDataset Information:")
    print(f"Train path: {train_path}")
    print(f"Val path: {val_path}")
    print(dataset_sizes)
    print(class_names)
    print()

    return dataloaders, dataset_sizes, class_names


class FlowerClassifier:
    def __init__(self, model_name: str = 'efficientnet_b0', num_classes: int = 5):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model()

    def _initialize_model(self) -> torch.nn.Module:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.model_name == 'resnet18':
                weights = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=weights)
                for name, param in model.named_parameters():
                    param.requires_grad = "fc" in name
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)

            elif self.model_name == 'efficientnet_b0':
                weights = models.EfficientNet_B0_Weights.DEFAULT
                model = models.efficientnet_b0(weights=weights)
                for name, param in model.named_parameters():
                    param.requires_grad = "classifier" in name
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)

            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

            return model.to(self.device)


def run_training(model_name: str, optimizer: str, learning_rate: float,
                 num_epochs: int, batch_size: int, data_dir: str,
                 status_updater=None):
    try:
        mlflow.set_experiment("Flower_Classification_UI")

        if status_updater:
            status_updater(
                step='Initializing',
                details='Preparing dataset and model'
            )

        dataloaders, dataset_sizes, class_names = prepare_data(data_dir, batch_size)

        if status_updater:
            status_updater(
                step='Dataset Loaded',
                dataset_info={
                    'train_path': os.path.join(data_dir, 'train'),
                    'val_path': os.path.join(data_dir, 'val'),
                    'train_count': dataset_sizes['train'],
                    'val_count': dataset_sizes['val'],
                    'classes': class_names
                }
            )

        classifier = FlowerClassifier(model_name, len(class_names))

        # Initialize optimizer
        params = filter(lambda p: p.requires_grad, classifier.model.parameters())
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(params, lr=learning_rate)
        else:
            optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)

        criterion = nn.CrossEntropyLoss()

        with mlflow.start_run():
            mlflow.log_params({
                "model_name": model_name,
                "optimizer": optimizer.__class__.__name__,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "num_classes": len(class_names),
                "class_names": ", ".join(class_names),
                "train_size": dataset_sizes['train'],
                "val_size": dataset_sizes['val']
            })

            for epoch in range(num_epochs):
                if status_updater:
                    status_updater(
                        step=f'Epoch {epoch + 1}/{num_epochs}',
                        details='Training epoch started',
                        epoch_progress={
                            'current': epoch + 1,
                            'total': num_epochs
                        }
                    )

                # Training phase
                classifier.model.train()
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders['train']:
                    inputs, labels = inputs.to(classifier.device), labels.to(classifier.device)
                    optimizer.zero_grad()
                    outputs = classifier.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes['train']
                epoch_acc = running_corrects.double() / dataset_sizes['train']

                # Validation phase
                classifier.model.eval()
                val_running_loss = 0.0
                val_running_corrects = 0

                for inputs, labels in dataloaders['val']:
                    inputs, labels = inputs.to(classifier.device), labels.to(classifier.device)
                    with torch.no_grad():
                        outputs = classifier.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    val_running_loss += loss.item() * inputs.size(0)
                    val_running_corrects += torch.sum(preds == labels.data)

                val_loss = val_running_loss / dataset_sizes['val']
                val_acc = val_running_corrects.double() / dataset_sizes['val']

                # Log metrics
                mlflow.log_metrics({
                    "train_loss": epoch_loss,
                    "train_accuracy": epoch_acc.item(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc.item()
                }, step=epoch)

                if status_updater:
                    status_updater(
                        step=f'Epoch {epoch + 1}/{num_epochs}',
                        details='Metrics calculated for epoch',
                        epoch_progress={
                            'current': epoch + 1,
                            'total': num_epochs,
                            'train_loss': epoch_loss,
                            'train_acc': epoch_acc.item(),
                            'val_loss': val_loss,
                            'val_acc': val_acc.item()
                        }
                    )

            if status_updater:
                status_updater(
                    step='Training Complete',
                    details='Model training finished successfully'
                )

            mlflow.pytorch.log_model(classifier.model, "model")

    except Exception as e:
        if status_updater:
            status_updater(
                step='Error',
                details=str(e)
            )
        raise e
