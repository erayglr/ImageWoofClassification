# 🐕 ImageWoof Dog Breed Classification

This project focuses on training and evaluating an image classification model using **ImageWoof**, a challenging subset of the **ImageNet** dataset containing 10 different dog breeds.

A pretrained **EfficientNet-B0** model is used with **transfer learning** to classify images into 10 dog breed categories.

## 🐕 Dog Breeds

The model classifies the following 10 breeds:

1. Australian Terrier
2. Beagle
3. Border Terrier
4. Dingo
5. English Foxhound
6. Golden Retriever
7. Old English Sheepdog
8. Rhodesian Ridgeback
9. Samoyed
10. Shih Tzu

## 🛠️ Technologies & Libraries

* **Python**
* **PyTorch**
* **Torchvision**
* **Torchinfo**
* **Matplotlib**
* **Pillow**
* **Google Colab GPU (CUDA)**

## 🚀 Model Architecture & Transfer Learning

The project uses **EfficientNet-B0 pretrained on ImageNet**.

The pretrained feature extractor layers are frozen to prevent their parameters from being updated during training:

```python
for param in model.features.parameters():
    param.requires_grad = False
```

The classifier head is replaced to match the 10 classes in the ImageWoof dataset:

```python
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280, out_features=10, bias=True)
)
```

### ⚙️ Training Configuration

| Parameter          | Value                   |
| ------------------ | ----------------------- |
| Model              | EfficientNet-B0         |
| Pretrained Weights | ImageNet                |
| Number of Classes  | 10                      |
| Loss Function      | CrossEntropyLoss        |
| Optimizer          | Adam                    |
| Learning Rate      | 0.001                   |
| Epochs             | 10                      |
| Device             | CUDA / Google Colab GPU |

## 📈 Training Results

The model was trained for **10 epochs**.

| Epoch | Train Accuracy | Test Accuracy |
| ----: | -------------: | ------------: |
|     1 |         66.82% |        86.88% |
|    10 |         93.60% |        88.33% |

The model achieved **93.60% training accuracy** and **88.33% test accuracy** after 10 epochs.

## 🔮 Sample Predictions

The trained model was also tested on custom images downloaded from the internet.

| Image            | Prediction       | Confidence |
| ---------------- | ---------------- | ---------: |
| Golden Retriever | Golden Retriever |  **99.8%** |
| Shih Tzu         | Shih Tzu         |  **98.2%** |

## 🎯 Project Goals

Through this project, the following concepts were practiced:

* Transfer learning with pretrained CNN models
* EfficientNet-B0 architecture
* Image classification using PyTorch
* Freezing pretrained model layers
* Customizing a classification head
* Model training and evaluation
* GPU-accelerated deep learning with CUDA
* Testing a trained model on custom images
