
---

# Disease Classification and Symptom Analysis using Deep Learning Models

## Introduction

This project focuses on applying machine learning models to medical data, particularly for classifying diseases like Tuberculosis using Chest X-ray images and symptom data. The project is divided into three phases:

1. **Baseline Model**: Implementation using DenseNet for Tuberculosis classification.
2. **First Improvement**: Integration of Efficient ViT with Grad-CAM for better explainability and classification accuracy, and inclusion of SHAP for feature importance analysis on the Symptoms Dataset.
3. **Final Improvement**: Combining feature vectors from Efficient ViT and Symptoms Score Dataset, and passing them through an MLP for the final classification score.

---

## Motivation

The motivation behind this project is to leverage advancements in deep learning for disease prediction. By using **Grad-CAM** visualizations for interpretability and **SHAP** for feature importance analysis, the project aims to improve not only the accuracy but also the transparency and trust in the system's predictions, particularly in a medical context.

---

## Directory Structure

```plaintext
├── Baseline/
│   ├── models.py
│   ├── models.ipynb
│   ├── symptom_score_dataset.py
│   ├── symptom_score_dataset.ipynb
│   ├── outputs          # Contains outputs comparing DenseNet with the original paper metrics
├── Improvement_with_effecient_Vit/
│   ├── explainable_AI_radiology.py
│   ├── explainable_AI_radiology.ipynb
│   ├── explainable_AI_symptoms_Dataset.py
│   ├── explainable_AI_symptoms_Dataset.ipynb
├── final_improvement/
│   ├── combined_dataset_notebook.py
│   ├── combined_dataset_notebook.ipynb
└── README.md
```

---

## Datasets

1. **Chest X-ray Dataset**: A collection of chest X-ray images classified as "Normal" or "Tuberculosis."
2. **Symptoms Dataset**: A dataset with symptom-based features to predict diseases like Tuberculosis.

---

## Models

### **DenseNet**

The baseline model in this project uses **DenseNet201**, a convolutional neural network known for its dense connections that facilitate feature reuse and enhance gradient flow. The model is pre-trained on ImageNet and fine-tuned for Tuberculosis classification from Chest X-ray images.

* **Model Setup**:

  * DenseNet201 is loaded from the torchvision library, and the pre-trained weights are utilized.
  * The final fully connected layers are replaced with a custom classification layer suitable for the binary classification task (Normal vs. Tuberculosis).
* **Hyperparameters**:

  * **Learning Rate**: `0.001`
  * **Optimizer**: Stochastic Gradient Descent (SGD) with momentum (`0.9`).
  * **Loss Function**: Cross-Entropy Loss, suitable for binary classification.
  * **Batch Size**: `8`
  * **Epochs**: `15`
* **Training Methodology**:

  * Pre-trained DenseNet is used as a feature extractor, and only the final classification layers are trained.
  * **Gradients**: Frozen for most layers except the fully connected layers to prevent overfitting and speed up training.

### **Efficient ViT (First Improvement)**

The **Efficient Vision Transformer (Efficient ViT)** is introduced as an improvement over the baseline DenseNet model. Efficient ViT integrates transformer-based architectures, which are particularly good at capturing global dependencies and context in images. This allows the model to learn richer features compared to CNN-based models.

* **Model Setup**:

  * Efficient ViT is implemented using a pre-trained version of the model, which is then fine-tuned for Tuberculosis classification.
  * **Grad-CAM** is used to generate visual explanations of the model's decision-making process.
* **Hyperparameters**:

  * **Learning Rate**: `0.0001` (lower learning rate due to transformer-specific optimization).
  * **Optimizer**: Adam optimizer with weight decay to prevent overfitting.
  * **Batch Size**: `16` (to improve training stability with transformer models).
  * **Epochs**: `10`
* **Training Methodology**:

  * Efficient ViT is fine-tuned for the chest X-ray classification task.
  * **Grad-CAM** is integrated into the model to provide visual insights into which parts of the X-ray images contribute most to the model’s classification decision.

### **Grad-CAM Visualization**

Grad-CAM is used in both DenseNet and Efficient ViT models to provide insights into the areas of the input images that are most relevant for the classification decision. Grad-CAM generates heatmaps that highlight the important regions, making the model’s predictions more interpretable and trustworthy.

---

## Explainable AI Symptom Score (First Improvement)

### **SHAP (Shapley Additive Explanations)**

For the **Symptoms Dataset**, SHAP values are introduced to explain which features (symptoms) contribute the most to predicting the likelihood of Tuberculosis. SHAP is a powerful explainability method that quantifies the importance of each feature in a model’s decision-making process, providing more transparency.

* **Integration**:

  * SHAP was used in the **Explainable AI Symptom Score** notebook to assess the importance of different symptoms in predicting Tuberculosis. This added layer of interpretability is essential when dealing with medical datasets to understand which features are most influential.

  * By applying SHAP, we gain a clearer view of how different symptoms contribute to the model’s predictions and can make informed decisions based on this data.

* **Use Case**:

  * SHAP is applied to the symptoms dataset to improve the interpretability of the predictions. This helps clinicians understand the correlation between symptoms and the likelihood of Tuberculosis.

---

## Improvements

1. **Integration of Efficient ViT**:

   * Efficient ViT was chosen for its superior ability to handle larger datasets and its efficiency in processing medical images.
   * Unlike DenseNet, Efficient ViT utilizes attention mechanisms that allow the model to focus on different parts of the image, improving its ability to capture contextual relationships in medical images.

2. **Grad-CAM Visualization**:

   * Grad-CAM is used to generate heatmaps that show which regions of an image contributed most to the model’s decision. This is particularly important in medical applications where model interpretability is crucial for clinical decision-making.
   * The integration of Grad-CAM helps enhance model transparency, making the system more acceptable in real-world healthcare applications.

3. **SHAP for Symptom-based Predictions**:

   * SHAP has been integrated into the symptom dataset model to explain the contribution of different symptoms in the prediction of Tuberculosis. This provides a more interpretable framework for healthcare professionals and can guide them in identifying the most relevant symptoms for diagnosis.

---

## Final Improvement

The **Final Improvement** phase combines feature vectors from the **Efficient ViT** and **Symptoms Score Dataset** and passes them through an **MLP** (Multi-Layer Perceptron) for the final classification score. This combination leverages both image and symptom data for improved prediction accuracy.

---

## Methodology

### **Data Preprocessing**

* **Chest X-ray Dataset**: Images are resized to `224x224` pixels to match the input size required by the models. Normalization is applied based on the ImageNet mean and standard deviation for consistency.
* **Symptoms Dataset**: Symptom data is preprocessed to create a binary classification task for Tuberculosis detection, using engineered features from symptom descriptions.

### **Model Training**

* **DenseNet**:

  * The DenseNet model is pre-trained on ImageNet and fine-tuned with the Tuberculosis dataset. The final fully connected layers are adjusted to output predictions for two classes: "Normal" and "Tuberculosis."
  * Dropout layers are added to prevent overfitting, and training is done for 15 epochs.

* **Efficient ViT**:

  * Efficient ViT is fine-tuned for the task of Tuberculosis detection. The transformer model uses a reduced learning rate and is trained for 10 epochs. This model is particularly effective for small datasets as it learns global context information across the entire image.

* **Final Model (MLP)**:

  * The final model combines feature vectors from **Efficient ViT** and **Symptoms Score Dataset**. These combined features are passed through an **MLP** to produce the final classification result.

### **Loss Function**:

* **Cross-Entropy Loss** is used for both DenseNet, Efficient ViT, and the final MLP-based model, which is suitable for classification tasks where the goal is to predict a probability distribution across classes.

### **Hyperparameters**:

* **Learning Rate**: `0.001` for DenseNet and `0.0001` for Efficient ViT. A smaller learning rate for Efficient ViT is used to account for the complexity of transformers.
* **Batch Size**: `8` for DenseNet and `16` for Efficient ViT, as the latter is more computationally demanding.
* **Early Stopping**: Applied to prevent overfitting and ensure that the models generalize well.

---

## Performance Evaluation

### **Metrics**:

* Accuracy, precision, recall, F1-score, and AUC are used to evaluate the models' performance on the Tuberculosis classification task.

### **Comparison**:

* The **DenseNet model** serves as the baseline, and the **Efficient ViT** model is compared to it to demonstrate improvements in classification accuracy, computational efficiency, and model interpretability.

---

## Results

1. **Baseline Model** (DenseNet):

   * DenseNet achieved strong performance in identifying Tuberculosis cases with high accuracy (89%).

2. **First Improvement** (Efficient ViT with AI explainability):

   * Efficient ViT improved the prediction accuracy while reducing the computational complexity of the model. The transformer-based model was able to capture more detailed patterns from the chest X-ray images.
   * **Grad-CAM visualizations** made the predictions more interpretable and transparent.

3. **Final Improvement** (Combined Datasets):

   * Combining multiple datasets (Chest X-ray and Symptoms) resulted in improved model performance, showing the effectiveness of multi-modal learning.

4. **Model Comparison** (Baseline vs. Paper Metrics):

   * The **output file** in the **Baseline** directory compares the metrics of our **DenseNet** implementation (`models.py`) with the original **DenseNet paper** results. The performance metrics show the efficacy of our implementation, which achieved a comparable accuracy of 89% in detecting Tuberculosis from Chest X-ray images.

---

## Usage

To run the models and evaluate their performance, you can use the following commands:

```bash
# Baseline model
python Baseline/models.py  # For Python script
jupyter notebook Baseline/models.ipynb  # For Jupyter Notebook

# First Improvement - Efficient ViT
python Improvement_with_effecient_Vit/explainable_AI_radiology.py  # For Python script
jupyter notebook Improvement_with_effecient_Vit/explainable_AI_radiology.ipynb  # For Jupyter Notebook

# First Improvement - Explainable AI Symptoms Score Dataset
python Improvement_with_effecient_Vit/explainable_AI_symptoms_Dataset.py  # For Python script
jupyter notebook Improvement_with_effecient_Vit/explainable_AI_symptoms_Dataset.ipynb  # For Jupyter Notebook

# Final Improvement
python final_improvement/combined_dataset_notebook.py  # For Python script
jupyter notebook final_improvement/combined_dataset_notebook.ipynb  # For Jupyter Notebook
```

---

## Conclusion

This project showcases the effectiveness of deep learning models like DenseNet and Efficient ViT for medical image classification. **Grad-CAM visualizations** enhance interpretability, making these models suitable for real-world applications in healthcare. **SHAP feature importance** adds further transparency to symptom-based models, improving their clinical applicability. The final improvement by combining **Efficient ViT** and **Symptoms Score Dataset** with an MLP has shown improved performance, providing a robust framework for disease classification. Future work could explore additional datasets, hyperparameter optimization, and techniques like self-supervised learning.

---

## References

1. **DenseNet**:
   Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2261-2269.*


2. **Medical X-Ray Attention (MXA) Block**:
   Ibrahim, H., & Rand, A. (2025). Beyond Conventional Transformers: The Medical X-ray Attention (MXA) Block for Improved Multi-label Diagnosis Using Knowledge Distillation. *arXiv preprint, 2504.02277.*
   [https://arxiv.org/abs/2504.02277](https://arxiv.org/abs/2504.02277)

3. **Explainable AI in Radiology**:
   Maheswari, B. U., Sam, D., Mittal, N., Sharma, A., Kaur, S., Askar, S. S., & Abouhawwash, M. (2024). Explainable deep-neural-network supported scheme for tuberculosis detection from chest radiographs. *BMC Medical Imaging, 24(32).*
   [https://doi.org/10.1186/s12880-024-01202-x](https://doi.org/10.1186/s12880-024-01202-x)

---


