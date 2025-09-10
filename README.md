

# 👁️ Eye Disease Classification

A deep learning project for classifying common eye diseases using **TensorFlow** and **Convolutional Neural Networks (CNNs)**.

This project trains and evaluates a model on the [Eye Diseases Classification Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification), which contains labeled images of various eye conditions.

---

## 📂 Dataset

- **Source:** [Kaggle – Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- **Categories:**
  - Cataract  
  - Diabetic Retinopathy  
  - Glaucoma  
  - Normal  

Download the dataset from Kaggle and place it in a folder named `dataset/` inside the project root:

```

Eye-Disease-Classification/
│── dataset/
│   ├── Cataract/
│   ├── Diabetic Retinopathy/
│   ├── Glaucoma/
│   └── Normal/

````

---

## 🛠️ Requirements

Make sure you have Python 3.8+ installed. Then install dependencies:

```bash
pip install -r requirements.txt
````

**requirements.txt**

```txt
tensorflow
numpy
matplotlib
scikit-learn
```

---

## 🚀 Usage

You can run the training pipeline directly from the Jupyter Notebook:

* [eye-diseases-classification.ipynb](https://github.com/BhargavBJ/Eye-Disease-Classification/blob/main/eye-diseases-classification.ipynb)

---

## 📊 Model

* Built with **TensorFlow / Keras**
* Architecture: Convolutional Neural Network (CNN)
* Loss: `categorical_crossentropy`
* Optimizer: `adam`
* Metrics: Accuracy

---

## 🔍 Results

* Achieves good accuracy on validation and test sets
* Confusion matrix and classification report included in the notebook

(Replace this section with actual accuracy and plots after training)

---

## 📈 Future Improvements

* Hyperparameter tuning (batch size, learning rate, epochs)
* Data augmentation for better generalization
* Try transfer learning (e.g., ResNet, EfficientNet)

---

## 🤝 Contributing

Pull requests are welcome! If you’d like to add new features (e.g., transfer learning, better visualizations), feel free to fork the repo and submit a PR.

---

## 📜 License

This project is licensed under the MIT License.

