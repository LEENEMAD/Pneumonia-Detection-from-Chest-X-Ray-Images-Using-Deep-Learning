# Pneumonia Detection from Chest X-ray Images

This project focuses on detecting pneumonia from chest X-ray images using deep learning models. The goal is to assist doctors in early diagnosis and improve decision-making in medical settings.

##  Project Overview
Pneumonia is a life-threatening disease that can be difficult to diagnose due to overlapping symptoms with other respiratory conditions. This project uses deep learning to classify chest X-ray images into:
- Normal
- Pneumonia

## Dataset
- Source: Kaggle Chest X-ray Pneumonia Dataset
- Total images: 5,863
- Classes: Binary (Normal vs Pneumonia)

### Data Processing
- Data merging and custom 3-way split (train/validation/test)
- Image resizing to 224×224
- Conversion to RGB
- Normalization
- Data augmentation (rotation, flipping, zooming, etc.)
- Class balancing using augmentation

##  Models Used
The following CNN architectures were implemented and compared:

- ResNet50 (transfer learning)
- VGG16 (transfer learning)
- DenseNet121 (transfer learning)
- AlexNet (custom implementation)

##  Training Strategy
- Transfer learning using pretrained ImageNet weights
- Stratified K-Fold Cross Validation
- Hyperparameter tuning (learning rate & dropout)
- Early stopping to prevent overfitting

## 📈 Results

| Model        | Accuracy | Precision | Recall | F1-score |
|-------------|--------|----------|--------|---------|
| ResNet50    | 77.8%  | 73.7%    | 86.4%  | 79.6%   |
| VGG16       | 83.4%  | 75.5%    | 98.8%  | 85.6%   |
| DenseNet121 | 79.6%  | 71.4%    | 98.8%  | 82.9%   |
| AlexNet     | 69.0%  | 62.4%    | 95.6%  | 75.5%   |

### 🏆 Best Model
- **VGG16 achieved the highest accuracy (83.4%)**
- High recall makes it suitable for medical diagnosis (minimizing missed cases)

##  Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

##  Observations
- ResNet50 showed the best generalization
- VGG16 achieved highest performance but showed overfitting
- DenseNet121 was efficient and lightweight
- AlexNet had high recall but low precision

##  Interface
A simple interface was built using **Gradio**, allowing users to:
- Upload chest X-ray images
- Select model for prediction
- Get real-time classification results

##  Tech Stack
- Python
- TensorFlow / Keras
- NumPy / Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Gradio

##  Future Work
- Improve generalization with larger datasets
- Reduce overfitting
- Deploy as a web application
- Add explainability (Grad-CAM)
