# DLAssignment
KMNIST Classification with MLP and CNN

This repository contains an implementation of KMNIST dataset classification using both a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN) in TensorFlow/Keras.

Dataset

The Kuzushiji-MNIST (KMNIST) dataset is a drop-in replacement for MNIST but features handwritten Japanese characters. It consists of 10 classes and follows the same structure as MNIST.

Features

MLP Model: A dense feedforward neural network with hyperparameter tuning.

CNN Model: A convolutional neural network for improved feature extraction.

Hyperparameter Tuning: Multiple configurations are tested to find the best-performing MLP model.

Evaluation Metrics: Uses accuracy, loss, confusion matrix, and classification report.

Training Visualization: Plots accuracy and loss trends over epochs.

Dependencies

Ensure you have the following dependencies installed:

pip install tensorflow tensorflow-datasets numpy pandas matplotlib seaborn scikit-learn

Usage

Clone this repository and run the script:

git clone https://github.com/yourusername/kmnist-classification.git
cd kmnist-classification
python kmnist_classification.py

Model Training

Data Loading: The dataset is loaded and split into training, validation, and test sets.

Model Training:

MLP models with different hyperparameters are trained and evaluated.

The best-performing MLP model is selected based on validation accuracy.

A CNN model is trained separately for comparison.

Evaluation:

Generates accuracy/loss plots.

Prints classification reports and confusion matrices.

Results

After training, the script will display the best model's performance and visualize key metrics.

Author

Rallabandi Siddhartha

License

This project is licensed under the MIT License.

