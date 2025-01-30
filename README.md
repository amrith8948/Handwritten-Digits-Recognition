# Handwritten Digit Recognition

## Introduction
Handwritten digit recognition is a crucial task in computer vision with numerous real-world applications, including postal code recognition, bank check processing, and automated form digitization. This project utilizes the MNIST dataset to train and evaluate multiple machine learning models, achieving high accuracy in digit recognition.

## Dataset Overview
- **Dataset**: MNIST
- **Images**: 70,000 grayscale images of handwritten digits.
- **Image Dimensions**: 28x28 pixels.
- **Splits**:
  - Training Set: 60,000 images.
  - Test Set: 10,000 images.
- **Challenges**:
  - Imbalance in digit representation.
  - Ambiguously written digits.
  - High-dimensionality (784 features per image).
  - Risk of overfitting in complex models like CNN.

## Techniques Applied
- **Data Normalization**: Scaled pixel values to a range of 0 to 1 for stability and convergence.
- **Dropout Regularization**: Applied to the CNN to reduce overfitting.
- **RBF Kernel**: Used in SVM for handling non-linear data.

## Project Workflow
1. **Library Imports**: Loaded required libraries for preprocessing and model building.
2. **Data Preparation**:
   - Normalized and reshaped the data.
   - Prepared datasets for training and testing.
3. **Model Creation**:
   - Built CNN, KNN, SVM, and Decision Tree models.
4. **Model Evaluation**:
   - Evaluated accuracy, training time, and inference speed.

## Models and Performance

1. **Convolutional Neural Network (CNN)**
   - **Architecture**:
     - Two convolutional layers (ReLU + MaxPooling).
     - Fully connected layers with dropout for regularization.
     - Softmax output layer.
   - **Performance**:
     - Training Accuracy: **99.43%**
     - Test Accuracy: **98.83%**
     - Loss: **Training: 1.67%, Test: 4.61%**
   
2. **K-Nearest Neighbors (KNN)**
   - **Performance**:
     - Training Accuracy: **97.9%**
     - Test Accuracy: **96.7%**
   
3. **Support Vector Machine (SVM) with RBF Kernel**
   - **Performance**:
     - Validation Accuracy: **97.75%**
     - Test Accuracy: **97.7%**
   
4. **Decision Tree**
   - **Performance**:
     - Training Accuracy: **100%**
     - Test Accuracy: **87.5%**

## Model Comparison

| **Model**             | **Accuracy (%)** | **Training Time** | **Inference Speed** | **Scalability** |
|-----------------------|------------------|------------------|---------------------|----------------|
| **CNN**               | 99.4             | Moderate         | Fast                | High           |
| **KNN**               | 97.9             | Fast             | Slow                | Low            |
| **SVM (RBF Kernel)**  | 97.7             | High             | Moderate            | Medium         |
| **Decision Tree**     | 86.7             | Fast             | Fast                | Medium         |

## Results and Conclusion
- **Best Model**: CNN
  - Accuracy: **99.4%**
  - Scalable and fast inference.
- **SVM**: Effective but less scalable.
- **KNN**: Slow on larger datasets.
- **Decision Tree**: Fast but less accurate.

**Recommendation**: The CNN model is ideal for production due to its high accuracy, scalability, and efficient inference. Further optimizations, such as enhancing the CNN or using transfer learning, could improve the model's performance.

## How to Run
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter notebook for model training and evaluation.
