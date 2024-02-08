# Multioutput CNN Model Training for Plant Disease and Severity Identification

## Overview
This repository contains the code and resources for a Multioutput Convolutional Neural Network (CNN) designed for disease classification and severity estimation in hydroponically grown Pechay (Brassica rapa ssp. chinensis). You can also use any plant you want, in my use case I used Pechay. 

A multioutput or multi-task learning is an approach where multiple tasks are performed simultaneously. This is given that the tasks are closely related to one another. To implement a multioutput approach in our study, a CNN model shall be trained to output the plant disease and disease severity of Pechay with a single input image. The figure aboves illustrates the process of building the model.

## Table of Contents

- [Model Training Process](#section1)
- [Project Structure](#section2)
- [Requirements](#section3)
- [Usage](#section4)

<a name="section1"></a>
## Model Training Process
The model training process involves two primary phases: transfer learning and custom fully connected layer integration.

### Transfer Learning

In this phase, a pre-trained model based on the InceptionV3 architecture and trained on the imagenet dataset is utilized as the foundation. Leveraging pre-existing knowledge from a vast dataset like imagenet allows for enhanced performance, particularly when the available dataset for the specific task is limited.

### Custom Fully Connected Layer

Following transfer learning, a custom fully connected layer is integrated into the model. This layer is manually designed to meet the specific requirements of the user, allowing for further customization and optimization.

For a detailed overview of the model training workflow, refer to the figure below

<p align="center">
   <img src = "resources/model training 2.jpg" alt="Logo">
</p>

Multioutput model training framework. We adapted a pre-trained model trained on the imagenet dataset, froze the layers, and combined it with our custom fully connected layers that output (1) plant disease and (2) disease severity.

<a name="section2"></a>
## Project Structure

- **/src**: Contains the source code for the CNN model.
- **/data**: Placeholder for datasets used in training and testing.
- **/docs**: Documentation related to the project.
- **/results**: Store evaluation metrics, graphs, and other results.

<a name="section3"></a>
## Requirements
- A virtual environment (recommended)
- Python 3.x (I used 3.9.13)
- Dependencies listed in `requirement_pyqt5.txt`

<a name="section4"></a>
## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/CarlosDwain/pechay-disease-cnn.git
   
2. Install the dependencies:
   ```bash
   pip install -r requirement_pyqt5.txt

3. Train the CNN model:
   ```bash
   python src/train_pechay.py

4. Plot the Training and Validation Accuracy and Loss:
   ```bash
   python src/plots.py
   
5. Test the model:
   ```bash
   python src/test_pechay.py
