# Multioutput CNN Model Training for Plant Disease and Severity Identification

## Overview
This repository contains the code and resources for a Multioutput Convolutional Neural Network (CNN) designed for disease classification and severity estimation in hydroponically grown Pechay (Brassica rapa ssp. chinensis). You can also use any plant you want. 

<p align="center">
   <img src = "resources/model training 2.jpg" alt="Logo">
</p>

A multioutput or multi-task learning is an approach where multiple tasks are performed simultaneously. This is given that the tasks are closely related to one another. To implement a multioutput approach in our study, a CNN model shall be trained to output the plant disease and disease severity of Pechay with a single input image. The figure aboves illustrates the process of building the model.

## Project Structure
- **/src**: Contains the source code for the CNN model.
- **/data**: Placeholder for datasets used in training and testing.
- **/docs**: Documentation related to the project.
- **/results**: Store evaluation metrics, graphs, and other results.

## Requirements
- A virtual environment (recommended)
- Python 3.x (I used 3.9.13)
- Dependencies listed in `requirement_pyqt5.txt`

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

4. Plot the Training and Validation Accuracy and Validation:
   ```bash
   python src/plots.py
   
5. Test the model:
   ```bash
   python src/test_pechay.py
