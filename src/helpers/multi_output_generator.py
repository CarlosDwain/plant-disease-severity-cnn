
import numpy as np

class MultiOutputGenerator():

    def __init__(self,
                 generator,
                 dataframe,
                 directory=None,    
                 image_data_generator=None,
                 x_col="filename",
                 y_col="class",
                 weight_col=None,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 interpolation='nearest',
                 dtype='float32',
                 validate_filenames=True):
        'Initialization'

        self.keras_generator = generator.flow_from_dataframe(
                 dataframe,
                 directory=directory,
                 image_data_generator=image_data_generator,
                 x_col=x_col,
                 y_col=y_col,
                 weight_col=weight_col,
                 target_size=target_size,
                 color_mode=color_mode,
                 classes=classes,
                 class_mode=class_mode,
                 batch_size=batch_size,
                 shuffle=shuffle,
                 seed=seed,
                 data_format=data_format,
                 save_to_dir=save_to_dir,
                 save_prefix=save_prefix,
                 save_format=save_format,
                 subset=subset,
                 interpolation=interpolation,
                 dtype=dtype,
                 validate_filenames=validate_filenames
        )

    #--- ---#
    #It enters an infinite loop (while True) to continuously generate batches of data.
    #Inside the loop, it uses self.keras_generator.next() to retrieve the next batch of data from self.keras_generator. 
    #This is a common practice when working with custom data generators.
    #It then processes the labels y from the batch to ensure that they are in float32 format using np.float32(gnext[1]). 
    #This conversion is done to make sure that the labels are in the correct data type for the model.
    #Finally, it yields a tuple containing the input data (gnext[0]) and a dictionary with two keys: 'disease' and 'severity'. 
    #The values associated with these keys are subsets of the processed labels y. 'disease' contains columns [0, 1, 2] from y, and 'severity' contains columns [3, 4, 5, 6, 7].

    #gnext[0], gnext[1] = gnext --> **gnext is a tuple** --> img, labels = gnext
    #print(gnext[0].shape) #returns (batch_size, *target_size, channels) of images
    #print(gnext[1].shape) #returns (batch_size, *target_size, channels) of labels

    #yield keyword is used to create a generator function. A type of function that is memory efficient and can be used like an iterator object.
    #In layman terms, the yield keyword will turn any expression that is given with it into a generator object and return it to the caller. 
    #Therefore, you must iterate over the generator object if you wish to obtain the values stored there.

    def getGenerator(self):

        while True:

            gnext = self.keras_generator.next() #In here, we are unpacking a tuple where gnext has two values the (images=gnext[0],  labels=gnext[1]) 
            y = np.float32(gnext[1]) #Convert the labels (originally dtype=object) to float32 data type.
            yield (gnext[0],{'disease': y[:,[0,1,2,3,4]], 'severity':  y[:, [5,6,7,8]]}) #Basically, returns a generator object


    def getNext(self):
        return self.keras_generator.next()

    @property
    def labels(self):
        if self.keras_generator.class_mode in {"multi_output", "raw"}:
            return self.keras_generator._targets
        else:
            return self.keras_generator.classes

    @property
    def sample_weight(self):
        return self.keras_generator._sample_weight

    @property
    def filepaths(self):
        return self.keras_generator._filepaths


#Example of yield

# def fun_generator():
#     yield "Hello world!!"
#     yield "Geeksforgeeks"
 
# obj = fun_generator()
 
# print(type(obj))
# print(next(obj))
# print(next(obj))

#output
# <class 'generator'>
# Hello world!!
# Geeksforgeeks


