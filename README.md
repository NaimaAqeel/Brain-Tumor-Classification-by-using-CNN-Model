

#  Brain Tumor Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** model to classify brain MRI images into two categories:

* **Yes Tumor** 🟥
* **No Tumor** 🟩

The model is trained on MRI images to help automate the detection of brain tumors from medical scans.

---


##  Dataset

* Total images: **3000** (1500 with tumor, 1500 without tumor)
* Size after preprocessing: **128x128 RGB images**
* Split: **80% training** / **20% testing**

---

##  Model Architecture

The CNN model was built using **TensorFlow/Keras**:

1. **Conv2D Layer** with 32 filters (3×3) + ReLU
2. **MaxPooling2D Layer** (2×2)
3. **Conv2D Layer** with 64 filters (3×3) + ReLU
4. **MaxPooling2D Layer** (2×2)
5. **Conv2D Layer** with 64 filters (3×3) + ReLU
6. **Flatten Layer**
7. **Dense Layer** with 64 neurons + ReLU
8. **Output Dense Layer** with 1 neuron + Sigmoid

**Loss Function:** Binary Crossentropy
**Optimizer:** Adam
**Metrics:** Accuracy

---

##  Results

* **Training Accuracy:** 99.84%
* **Validation Accuracy:** 95.42%
* **Test Accuracy:** 96.67%

### Confusion Matrix
<img width="683" height="547" alt="image" src="https://github.com/user-attachments/assets/73bf15aa-27c0-425b-8bbb-5d344622971d" />

* True Negatives: 283
* True Positives: 297
  # Misclassifications
*  Few false positives: 8
*  False negatives: 12

### Precision, Recall & F1-score:

* High overall performance showing **good generalization**.

---

## 📉 Training Curves

The notebook also includes plots for:

* Training vs Validation **Accuracy**
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/7b4f2e55-6648-4da2-9276-f2dff441fcfe" />

* Training vs Validation **Loss**
<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/af1d077a-09a1-4e63-aef6-1f9f711fe1b5" />

These show the model converges well with minimal overfitting.

---


##  Lessons Learned

* CNNs are highly effective for **medical image classification**.
* Proper preprocessing (resizing, normalization) significantly improves results.
* Overfitting can be reduced with techniques like **regularization, dropout, or data augmentation**.
* Evaluation metrics beyond accuracy (precision, recall, F1-score, confusion matrix) provide deeper insight.

---

##  Future Work

* Apply **Data Augmentation** to increase robustness.
* Experiment with deeper architectures (ResNet, VGG, Inception).
* Deploy model as a **web app** for real-time brain tumor detection.

---

✨ This project demonstrates the application of **Deep Learning (CNNs)** in healthcare, assisting in the early detection of brain tumors.

---

