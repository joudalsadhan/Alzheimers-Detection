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

​(B) CNN Trained From Scratch 

​Architecture: 

​Blocks: Conv(32) → BN → MaxPool → Dropout(0.25) 

​Conv(64) → Conv(64) → BN → MaxPool → Dropout(0.25) 

​Conv(128) → Conv(128) → BN → MaxPool → Dropout(0.25) 

​Flatten → Dense(256, ReLU) → Dropout(0.25) → Softmax(4) 

​Classes: 4-way (Non, Very mild, Mild, Moderate). 

​Optimizer/Loss: Adam + Categorical Cross-Entropy. 

​Regularization: BatchNorm + Dropout(0.25). 

​Reported internal result: ~0.999 accuracy and 0.002 loss on the internal split. 


# GRAD-CAM 

<img width="694" height="1280" alt="image" src="https://github.com/user-attachments/assets/406d8098-5830-44de-a968-7324d6ad80fb" />

# System Design
​​Modular layout: clear separation of data, model, and serving code. 

​Backend (training/inference): Python + TensorFlow/Keras. 

​data/ — loading, preprocessing, augmentation. 

​models/ — IRv2 head + scratch CNN definitions. 

​training/ — experiments, callbacks, logging, checkpoints. 

​inference/ — model loading, prediction, Grad-CAM generation. 

​Frontend: lightweight CODE CELL to upload an MRI slice/scan and return predictions. 

​Integration/IO: 

​Endpoint /predict accepts image input, handles resize to 299×299, returns JSON with class probabilities. 

​Optionally returns a Grad-CAM overlay for interpretability. 

​Runtime: GPU; batch size 1 for inference; optional confidence thresholding or reject option for low-confidence cases. 

​Quality controls: store metrics, plots, confusion matrices; enable later confidence calibration (e.g., temperature scaling). 

# Data Preprocessing 

​​Before training, all MRI scans were resized to 299×299 pixels with 3 color channels to match the input requirements of the Inception-ResNetV2 architecture. Pixel values were normalized to the [0,1] range. 

​To enhance generalization and reduce overfitting, data augmentation was applied to the training set, including random rotations, horizontal flips, brightness and contrast adjustments, and slight zooming. Validation and test sets were only resized and normalized. 

​The preprocessed data was then organized into TensorFlow datasets with batching, shuffling, and prefetching for efficient training 

<img width="1280" height="507" alt="image" src="https://github.com/user-attachments/assets/c67a6cdc-2c74-4dda-bea2-ec7a804de5fb" />

# Exploratory Data Analysis (EDA)​​

​During EDA, the class distribution of the dataset was examined. The four categories (Non-Demented, Very Mild Demented, Mild Demented, and Demented) showed some imbalance, with the non-demented class containing more samples than the others. 
​Basic image visualizations confirmed the structural differences between healthy and Alzheimer’s-affected brains. Intensity histograms also highlighted variation across disease stages. 

<img width="1280" height="417" alt="image" src="https://github.com/user-attachments/assets/43ba826b-ded9-46bf-b2fd-fefdd31af693" />

# Modeling

​​​Final model — Transfer Learning (Inception-ResNetV2, 3-class merged). 
Pretrained IRv2 used for feature extraction (input 299×299×3) with a custom head: GAP → BatchNorm → Dense(1024, ReLU) → Dropout(0.5) → Dense(512, ReLU) → Dropout(0.5) → Dense(256, ReLU) → Dropout(0.5) → Softmax(3) for Non Demented, Very mild, Mild+Moderate. Backbone was frozen during base training; best checkpoint selected (fine-tuning optional). Training used Adam (lr=1e-4), categorical cross-entropy, batch size=16,  
​and callbacks (EarlyStopping, ModelCheckpoint on val_accuracy,  ReduceLROnPlateau). 
​Light augmentation (rotate/flip/brightness) and IRv2 preprocess_inputapplied.  
​Test performance (hold-out): 
​Accuracy 0.847, Loss 0.391.  
​Per-class metrics (P/R/F1, n=500 each): •  
​Non Demented: 0.87/0.94/0.90 
​Very mild: 0.86/0.79/0.83  
​Mild+Moderate: 0.81/0.81/0.81  
​Confusion concentrated between Very mild ↔ Mild+Moderate.  
​Grad-CAM (layerconv_7b_ac`) mostly focused on brain tissue; a few samples showed edge/skull attention, pointing to benefit from stronger skull-stripping/registration. 
​Scratch model — CNN baseline (4-class). 
Architecture: three Conv blocks (32→64→128 with BN/MaxPool/Dropout 0.25), then Flatten → Dense(256, ReLU) → Dropout(0.25) → Softmax(4) (Non, Very mild, Mild, Moderate). Trained with Adam + categorical cross-entropy. Reported internal score ~0.999 accuracy, 0.002 loss on the internal split—likely overfitting given the transfer model’s 0.85 on a true test set. Kept as an ablation; not promoted for deployment until validated on a separate hold-out or via k-fold CV. 

​ 


