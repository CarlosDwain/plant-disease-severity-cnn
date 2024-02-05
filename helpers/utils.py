import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix


def get_name_labels(category, label):
    '''Given number class, it return the corresponding label name'''

    if category == 'plant_disease':

        if label == 0:
            return 'early_healthy'

        if label == 1:
            return 'late_healthy'

        if label == 2:
            return 'early_chlorotic'

        if label == 3:
            return 'late_chlorotic'

        if label == 4:
            return 'necrotic'

    if category == 'severity':

        if label == 0:
            return 'no risk'

        if label == 1:
            return 'low'

        if label == 2:
            return 'medium'

        if label == 3:
            return 'high'


def compute_confusion_matrix(
                         actual_labels,
                         predicted_labels,
                         labels,
                         save_to = None,
                         name_labels=None):

    cm = confusion_matrix(actual_labels, predicted_labels, labels=labels)
    print("\nConfusion Matrix:\n")
    print(cm)
    font_size = 18
    sns.set(font_scale=1.2) #scales the font of the row and column names of the confusion matrix

    if save_to:
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
        fig, ax = plt.subplots(figsize=(10,10))
        ax = sns.heatmap(cm, annot=True, fmt='d',cmap='BuPu', annot_kws={'size': font_size}, xticklabels=name_labels, yticklabels=name_labels)
        
        #bottom, top = ax.get_ylim()
        #print(ax.get_ylim())
        #ax.set_ylim(bottom + 0.5, top - 0.5)

        plt.ylabel('Actual label\n', fontsize=font_size)
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclassification={:0.4f}'.format(accuracy, misclass), color="black", labelpad=20, fontsize=font_size)
        plt.savefig(save_to, bbox_inches = "tight", dpi=1000)

        print("\nSaved confusion matrix to {}\n".format(save_to))


def compute_classification_report(actual_labels, predicted_labels, save_to=None):
    print("\nClassification Report:\n")
    report= pd.DataFrame(classification_report(actual_labels,predicted_labels,output_dict=True)).transpose()
    print(report)
    if save_to:
        report.to_csv(save_to, index=True)
        print("Saved classification report to {}".format(save_to))



def compute_misclassification(actual_labels, predicted_labels,generator, save_to, category ='plant_disease',name_label = True):

    count = 0
    df_misclassifications = pd.DataFrame(columns=['Index test set', 'Actual Label', 'Predicted Label', 'Image'])


    for index, (first, second) in enumerate(zip(actual_labels,predicted_labels)):
        if first != second:
            count = count +1
            path = os.path.basename(os.path.normpath(generator.filepaths[index]))
            if name_label:
                df_misclassifications = df_misclassifications.append({'Index test set': index,'Actual Label': get_name_labels(category, first), 'Predicted Label': get_name_labels(category, second), 'Image': path.replace("'", "")},ignore_index=True)
            else:
                df_misclassifications = df_misclassifications.append({'Index test set': index,'Actual Label': first, 'Predicted Label': second, 'Image': path.replace("'", "")},ignore_index=True)

    print("\nFound {} miscalssifications for {}.\n".format(count, category))
    df_misclassifications.to_csv(save_to, index=False)
    print("\nSaved csv in {}\n".format(save_to))



def save_history(history, path):
    df = pd.DataFrame(history)
    df.to_csv(path, index=False)
    print("Saved history to {}".format(path))
