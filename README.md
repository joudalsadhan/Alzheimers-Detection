# Alzheimers-Detection
Deep learning project for early Alzheimer’s detection using brain MRI scans. Includes preprocessing, CNN &amp; transfer learning models (Inception-ResNetV2), evaluation metrics, and Grad-CAM for explainability.
# Motivation and Objective
​​​The primary motivation behind this project is to leverage deep learning techniques, particularly Convolutional Neural Networks (CNNs), for the early and accurate detection of Alzheimer’s disease using brain MRI scans. Alzheimer’s is a progressive neurodegenerative disease where early diagnosis can significantly improve patient care and treatment planning. The objective is to build a robust classification system capable of automatically distinguishing between MRI scans of healthy individuals and those with Alzheimer’s disease. This project aims to achieve a classification accuracy of at least 98%, along with sensitivity and specificity of at least 95%, by combining diverse MRI datasets, applying data preprocessing and augmentation, optimizing CNN architectures, and utilizing interpretability methods such as Grad-CAM to ensure clinical relevance. 
# Data Source
- **Dataset**: Images OASIS
- **Source**: [Kaggle - Images OASIS Dataset](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)
- **Description**: MRI brain scans for Alzheimer's disease classification
- It contains four classes of grayscale brain MRI images: Non Demented, Very Mild Demented, Mild Demented, and Demented. The data was organized into class-specific folders and automatically indexed. Unique patient IDs were extracted to prevent data leakage, and the dataset was split into training, validation, and testing sets using a stratified group-based approach.
# Training Methodology
​​​(A) Transfer Learning — Inception-ResNetV2 (Fine-Tuned). 

​Backbone: Inception-ResNetV2 (ImageNet). Input size 299×299×3. Grad-CAM layer: conv_7b_ac. 
​Classes (merged): Non Demented, Very mild Dementia, Mild+Moderate Dementia. 
​Classification head: GlobalAveragePooling2D → BatchNorm → Dense(1024, ReLU) → Dropout(0.5) → Dense(512, ReLU) → Dropout(0.5) → Dense(256, ReLU) → Dropout(0.5) → Softmax(3). 
​Backbone training: Backbone frozen for base training; head trained and best checkpoint kept (fine-tuning optional). 
​Optimizer: Adam, learning rate 1e-4. 
​Loss: Categorical Cross-Entropy. 
​Batch size: 32 
​Epochs/Callbacks: Max 20 with EarlyStopping, ModelCheckpoint (best val_accuracy), and ReduceLROnPlateau. 
​Regularization: Dropout + BatchNorm. 
​Preprocessing & Augmentation: preprocess_input for IRv2; light rotation/flip/brightness augmentation. 
​Test results (hold-out set): 
​Accuracy: 0.847 , Loss: 0.391 
​Per-class (precision / recall / F1, n=500 each): 
​Non Demented: 0.87 / 0.94 / 0.90 
​Very mild: 0.86 / 0.79 / 0.83 
​Mild+Moderate: 0.81 / 0.81 / 0.81 
​Confusion matrix (rows = true, cols = pred): 
​Observation: most confusion is between Very mild and Mild+Moderate—expected for borderline cases. 
<img width="524" height="432" alt="image" src="https://github.com/user-attachments/assets/d83bc48c-2ac4-42df-beb0-e5c24ffdf328" />
<img width="1280" height="544" alt="image" src="https://github.com/user-attachments/assets/1b79c3e2-b2d3-4dfd-9c74-ac9ad8df4e4e" />

