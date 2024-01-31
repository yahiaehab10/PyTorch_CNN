# Convolutional Neural Network (CNN) for FashionMNIST

This repository contains a Python script that builds and trains a Convolutional Neural Network (CNN) on the FashionMNIST dataset using PyTorch. The model architecture is based on the TinyVGG model from the CNN explainer (https://poloclub.github.io/cnn-explainer/). The primary goal is to replicate the CNN architecture explained in the CNN explainer and achieve high accuracy on the FashionMNIST dataset.

## Getting Started

### Prerequisites
- [Python](https://www.python.org/) (>=3.6)
- [PyTorch](https://pytorch.org/) (>=1.7.0)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)

### Installation
You can install the required dependencies using the following command:

```bash
pip install torch torchvision matplotlib numpy
```

### Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   ```

2. Run using Juptier notebook

3. The script will download the FashionMNIST dataset, create a CNN model based on the TinyVGG architecture, train the model, and evaluate its performance on the test set.

## Model Architecture

The `FashionMNISTModel` class defines the CNN architecture, replicating the TinyVGG model from the CNN explainer. The model consists of two convolutional blocks followed by a classifier. The architecture is designed to capture patterns in visual data efficiently.

## FashionMNIST Dataset

The script uses the FashionMNIST dataset, a collection of 28x28 grayscale images of fashion items such as clothing and accessories. The dataset is split into training and test sets.

## Training

The script trains the CNN model using stochastic gradient descent (SGD) as the optimizer and cross-entropy loss as the loss function. Training is performed for a specified number of epochs, and progress is displayed for each epoch.

## Evaluation

After training, the model is evaluated on the test set to assess its performance. The evaluation includes metrics such as test loss, test accuracy, and a classification report.

## Visualization

The script includes visualization of random samples from the training set, the model architecture, and a confusion matrix to further evaluate the model's predictions.

## Acknowledgements

This project was inspired by Daniel Bourke, and you can find his GitHub repository [here](https://github.com/mrdbourke).

Feel free to modify and extend this script for your specific use case or experiment with different hyperparameters to improve model performance. If you encounter any issues or have suggestions, please open an issue or submit a pull request.
