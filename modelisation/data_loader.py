### initial code : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import constants

from tensorflow import keras
from PIL import Image

class DataGenerator(keras.utils.Sequence):
    """
    Loading batch on the fly, compiling on keras. 
    """
    
    def __init__(self, list_IDs, labels, n_classes, n_channels=3, batch_size=32, shuffle=True, resize_dim=(128,128)):
        
        """
        Initialization
        
        Parameters
        ----------
        list_IDs : dic
            define which labels (values) will be in train, validation and test (keys). 
            
        labels : dic
            define output value (values) for label (keys). 
            
        resize_dim : tuple
            define resolution to resize image.  
            
        n_channels : int
            define number of channel on image. 
            
        batch_size : int (default=32)
            define the batch size.  
            
        n_classes : int 
            define the number of classes to predict.  
            
        shuffle : bool
            define if image will be shuffle before creating batch at each epoch. 
        
        """
    
        self.list_IDs = list_IDs
        self.labels = labels
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resize_dim = resize_dim
        self.on_epoch_end()

    def __len__(self):
        
        """
        Denotes the number of batch per epoch (n/batch_size)
        """
        
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        
        """
        Generate one batch of data
        
        Returns
        -------
        X (image) and y (output) data
        """
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        
        """
        Updates indexes after each epoch
        """
        
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        
        """
        Generates data containing batch_size samples.
        
        Parameters
        ----------
        list_IDs_temp : list
             list of labels of batch. 
             
        Returns
        -------
        X (image) and y (output) data
        """
        
        # Initialization
        X = np.empty((self.batch_size, *self.resize_dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            #generate tiles
            X[i,] = np.array(Image.open(constants.PATH_IMAGES+ID+".jpg").resize(self.resize_dim))
            
            # Store class
            y[i] = self.labels[ID]

        return X, y