import matplotlib.pyplot as plt
import pandas as pd
import os

try:
    #***creates a temp folder and stores path in _MEIPASS***#
    base_path = sys._MEIPASS2   
except Exception:
    base_path = os.path.abspath(".")


name = 'pechayPlant'
net = 'inceptionV3'
DATE = '29_01_2024'   

LABEL_DIR = os.path.join(base_path,'out', name, net, DATE,'train')
SAVE_DIR = os.path.join(base_path,'out', name, net, DATE,'train')

plot_save_path = SAVE_DIR + '/plots/'
if not(os.path.exists(plot_save_path)):
    os.makedirs(plot_save_path)

df = pd.read_csv(os.path.join(LABEL_DIR,'inceptionV3_history.csv'))
# print(df)
# print(df.shape)
# print(df.info())

train_disease_acc = df['disease_accuracy']
train_severity_acc = df['severity_accuracy']
val_disease_acc = df['val_disease_accuracy']
val_severity_acc = df['val_severity_accuracy']

train_disease_loss = df['disease_loss']
train_severity_loss = df['severity_loss']
val_disease_loss = df['val_disease_loss']
val_severity_loss = df['val_severity_loss']

epochs = range(len(df))
print(epochs)

plt.plot(epochs, train_disease_acc, 'r', label='Training disease accuracy')
plt.plot(epochs, val_disease_acc, 'b', label='Validation disease accuracy')
#plt.title('Training and Validation Accuracy of Plant Disease')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(plot_save_path, 'accuracy_disease_plot.png'), dpi=1000)
plt.show()

plt.figure()
plt.plot(epochs, train_disease_loss, 'r', label='Training disease loss')
plt.plot(epochs, val_disease_loss, 'b', label='Validation disease loss')
#plt.title('Training and Validation Loss of Plant Disease')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(plot_save_path, 'loss_disease_plot.png'), dpi=1000)
plt.show()

plt.plot(epochs, train_severity_acc, 'r', label='Training severity accuracy')
plt.plot(epochs, val_severity_acc, 'b', label='Validation severity accuracy')
#plt.title('Training and Validation Accuracy of Disease Severity')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(plot_save_path, 'accuracy_severity_plot.png'), dpi=1000)
plt.show()

plt.plot(epochs, train_severity_loss, 'r', label='Training severity loss')
plt.plot(epochs, val_severity_loss, 'b', label='Validation severity loss')
#plt.title('Training and Validation Loss of Disease Severity')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(plot_save_path, 'loss_severity_plot.png'), dpi=1000)
plt.show()



