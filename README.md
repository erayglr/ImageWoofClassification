# 🐕 ImageWoof Dog Breed Classification

Bu proje, **ImageNet** veri setinin daha zorlu bir alt kümesi olan **ImageWoof** veri seti kullanılarak 10 farklı köpek cinsinin sınıflandırılmasını amaçlamaktadır.

Model mimarisi olarak **transfer learning** yöntemiyle önceden ImageNet üzerinde eğitilmiş **EfficientNet-B0** kullanılmıştır.

## 🐕 Sınıflandırılan Köpek Cinsleri

Model aşağıdaki 10 sınıfı sınıflandırmaktadır:

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

## 🛠️ Kullanılan Teknolojiler

* **Python**
* **PyTorch**
* **Torchvision**
* **Torchinfo**
* **Matplotlib**
* **Pillow**
* **Google Colab GPU (CUDA)**

## 🚀 Model ve Transfer Learning

Projede **ImageNet ağırlıklarıyla önceden eğitilmiş EfficientNet-B0** kullanılmıştır.

Modelin `features` katmanları dondurularak önceden öğrenilmiş özelliklerin korunması sağlanmıştır:

```python
for param in model.features.parameters():
    param.requires_grad = False
```

ImageWoof veri setindeki 10 sınıfa uygun olması için modelin classifier katmanı yeniden tanımlanmıştır:

```python
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280, out_features=10, bias=True)
)
```

### ⚙️ Eğitim Ayarları

| Parametre          | Değer                   |
| ------------------ | ----------------------- |
| Model              | EfficientNet-B0         |
| Pretrained Weights | ImageNet                |
| Number of Classes  | 10                      |
| Loss Function      | CrossEntropyLoss        |
| Optimizer          | Adam                    |
| Learning Rate      | 0.001                   |
| Epochs             | 10                      |
| Device             | CUDA / Google Colab GPU |

## 📈 Eğitim Sonuçları

Model 10 epoch boyunca eğitilmiştir.

| Epoch | Train Accuracy | Test Accuracy |
| ----: | -------------: | ------------: |
|     1 |         66.82% |        86.88% |
|    10 |         93.60% |        88.33% |

Model, eğitim sonunda **%93.60 train accuracy** ve **%88.33 test accuracy** değerine ulaşmıştır.

## 🔮 Örnek Tahminler

Eğitilen model, eğitim sırasında kullanılmayan internet kaynaklı özel görseller üzerinde de test edilmiştir.

| Görsel           | Tahmin           |  Olasılık |
| ---------------- | ---------------- | --------: |
| Golden Retriever | Golden Retriever | **99.8%** |
| Shih Tzu         | Shih Tzu         | **98.2%** |

## 📂 Proje İçeriği

```text
ImageWoof/
│
├── ImageWoof_Classification.ipynb
├── README.md
└── ...
```

## 🎯 Projenin Amacı

Bu proje ile:

* Transfer learning yaklaşımını uygulamak
* Önceden eğitilmiş CNN modellerini kullanmak
* EfficientNet-B0 mimarisini kullanarak görüntü sınıflandırmak
* PyTorch ile model eğitimi gerçekleştirmek
* Model performansını train/test accuracy üzerinden değerlendirmek
* Eğitilmiş modeli gerçek görüntüler üzerinde test etmek

amaçlanmıştır.
