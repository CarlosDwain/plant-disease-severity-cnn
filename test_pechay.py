import os
import datetime
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from helpers.dataset import *
from helpers.multi_output_generator import MultiOutputGenerator
from helpers.utils import *
from models.classifiers import *
from models.pechayplantNet import pechayplantNet
from sklearn.metrics import ConfusionMatrixDisplay


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#SETTINGS
BATCH_SIZE = 32
X_COL = "filename"
Y_COL = ['early_healthy','late_healthy','early_chlorotic','late_chlorotic','necrotic','severity_0','severity_1', 'severity_2', 'severity_3']
IMG_W = 224
IMG_H = 224

try:
    #***creates a temp folder and stores path in _MEIPASS***#
    base_path = sys._MEIPASS2   
except Exception:
    base_path = os.path.abspath(".")

LABEL_DIR = os.path.join(base_path,'data','pechayDataset','csv')
DATA_DIR = os.path.join(base_path,'img')
OUT_PATH = os.path.join(base_path,'out','pechayPlant')

networks = ['inceptionV3']

DATE = "29_01_2024"

def change_label(y_label):
    """
    This function changes the labels to 3 categories only instead of 5 
    From healthy_young, healthy_adult, chlorotic_young, chlorotic_adult, necrotic ---> healthy, chlorotic, necrotic
    """

    y = []
    for label in y_label:
        if label == 1:
            label = 0
        if label == 2:
            label = 1
        if label == 3:
            label = 1
        if label == 4:
            label = 2
        y.append(label)
    y = np.array(y)
    return y

if __name__ == "__main__":

    #test_folder = OUT_PATH + '/csv/' + DATE + '/'
    test_folder = OUT_PATH + '/csv/'
    test_path = os.path.join(test_folder,'test.csv')

    test = load_data(path_csv=test_path, sep=',')


    for net in networks:

        test_datagen = ImageDataGenerator(
            rescale=1./255,
        )

        test_generator = MultiOutputGenerator(
                        generator = test_datagen,
                        dataframe = test,
                        directory=DATA_DIR,
                        batch_size = 1,
                        x_col = X_COL,
                        y_col = Y_COL,
                        class_mode="raw",
                        target_size = (IMG_H, IMG_W),
                        shuffle=False
        )


        model = pechayplantNet(conv_base=net, shape=(IMG_H,IMG_W,3))


        train_save_path = os.path.join(OUT_PATH, net, DATE, 'train')
        arch_filename = net + '_arch.h5'
        model.load_model(path_arch=os.path.join(train_save_path,arch_filename))

        weights_filename = net + '_weights.h5'
        model.load_weights(path_weights=os.path.join(train_save_path,weights_filename), by_name = True, skip_mismatch=True)

        y_pred = model.test(test_generator= test_generator,
                            batch_size = 1,
                            steps=int(len(test)/1),
                            verbose=1)

        test_save_path = os.path.join(OUT_PATH, net, DATE, 'test')
        if not(os.path.exists(test_save_path)):
                os.makedirs(test_save_path)

        y_true = change_label(test_generator.labels[:,[0,1,2,3,4]].argmax(axis=1))
        y_pred_results = change_label(y_pred[0].argmax(axis=1))

        print("\nCM - Plant Disease:\n")
        compute_confusion_matrix(actual_labels = y_true,
                                 predicted_labels = y_pred_results,
                                 labels = [0,1,2], #0 talaga to magsisimula
                                 name_labels = ['Healthy','Chlorotic','Necrotic'],
                                 save_to = test_save_path + '/' + net + '_CM_plant_disease.png'
        )

        print("\nCM - Severity:\n")
        compute_confusion_matrix(actual_labels = test_generator.labels[:,[5,6,7,8]].argmax(axis=1),
                                 predicted_labels = y_pred[1].argmax(axis=1),
                                 labels = [0,1,2,3], #0 talaga to magsisimula
                                 name_labels = ['No risk','Low risk', 'Medium risk', 'High Risk'],
                                 save_to = test_save_path + '/' + net + '_CM_severity.png'
        )

        print("Report - Plant Disease:\n")
        compute_classification_report(actual_labels = y_true,
                                      predicted_labels = y_pred_results,
                                      save_to = test_save_path + '/' + net + '_report_plant_disease.csv'
        )

        print("Report - Severity: \n")
        compute_classification_report(actual_labels = test_generator.labels[:,[5,6,7,8]].argmax(axis=1),
                                      predicted_labels = y_pred[1].argmax(axis=1),
                                      save_to = test_save_path + '/' + net + '_report_severity.csv'
        )

"""
    --> test_generator.labels[:,[0,1,2,3,4]].argmax(axis=1)
    --> get the index of the max value of a row (axis=1) 
            get the label of that index in the test_generator -> do this for all rows in the first, second, and third column (disease column)

    --> y_pred returns 2 arrays (disease and severity) of prediction. 
    --> First array = 5 probabilities predictions [pred1,pred2,pred3,pred4,pred5] 
    --> Second array = 4 probabilities predictions [pred1,pred2,pred3,pred4]

    --> y_pred[0].argmax(axis=1)
    --> get the index of the max value of a row (axis=1)
        get the prediction of that index in the first array (y_pred[0]) of the y_pred variable

    --> y_pred[0] means the First Array (the disease array)

    print(y_true = test_generator.labels[:,[0,1,2,3,4]].argmax(axis=1))

    Sample Print Output:
    [2 1 4 3 1 2 3 4 4 3 1 1 4 2 0 1 4 3 0 3 1 4 0 3 2 1 3 4 0 3 2 2 2 4 4 0 1
    1 4 4 4 0 4 4 2 4 1 1 4 4 2 3 1 3 4 4 2 1 3 0 3 3 1 2 2 2 0 1 2 4 3 4 2 2
    0 3 1 1 1 3 2 1 1 3 1 1 3 2 3 1 1 3 1 1 1 1 1 3 1 1 0 0 3 1 1 1 1 0 3 0 3
    1 1 2 1 2 4 3 1 0 4 3 1 3 2 1 4 2 3 3 4 3 1 2 2 2 2 3 3 1 1 3 3 4 1 1 3 3
    3 4 2 4 1 3 0 2 1 2 0 4 3 2 1 1 3 2 1 3 0 1 4 1 0 2]
    print('type: ', type(y_true)) #output = numpy.ndararay
    print('shape: ', y_true.shape)
    print('size: ', y_true.shape)
    print('length: ', len(y_true))
    print(y_true[1])
"""