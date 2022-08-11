import numpy as np
from tensorflow import keras

def build_Unet(dim, act_conv="selu", nb_output=1, activation=None):
    """
    Encoder Unet model with activation function for the convolution part (default "selu")

    Parameters
    ----------
    dim : tuple
        dimensions of image (resolution, resolution, n_channel), . 
        
    act_conv : str
        choose the activation function for convolutionnal layer  (could be "relu", "sigmoid", "selu"...).

    nb_output : int
        Number of neurons at last layer.
        
    activation : str
        Activation function at last layer
        
    Returns
    ------
    Keras model
    """

    inputs = keras.layers.Input(dim)

    c1 = keras.layers.BatchNormalization()(inputs)
    c1 = keras.layers.Conv2D(16, (3,3), activation=act_conv, kernel_initializer="he_normal", padding = "same")(c1)
    c1 = keras.layers.BatchNormalization()(c1)
    c1 = keras.layers.Conv2D(16, (3,3), activation=act_conv, kernel_initializer="he_normal", padding = "same")(c1)
    p1 = keras.layers.MaxPooling2D(2,2)(c1)

    c2 = keras.layers.Conv2D(32, (3,3), activation=act_conv, kernel_initializer="he_normal", padding = "same")(p1)
    c2 = keras.layers.BatchNormalization()(c2)
    c2 = keras.layers.Conv2D(32, (3,3), activation=act_conv, kernel_initializer="he_normal", padding = "same")(c2)
    p2 = keras.layers.MaxPooling2D(2,2)(c2)

    c3 = keras.layers.Conv2D(64, (3,3), activation=act_conv, kernel_initializer="he_normal", padding = "same")(p2)
    c3 = keras.layers.BatchNormalization()(c3)
    c3 = keras.layers.Conv2D(64, (3,3), activation=act_conv, kernel_initializer="he_normal", padding = "same")(c3)
    p3 = keras.layers.MaxPooling2D(2,2)(c3)

    c4 = keras.layers.Conv2D(128, (3,3), activation=act_conv, kernel_initializer="he_normal", padding = "same")(p3)
    c4 = keras.layers.BatchNormalization()(c4)
    c4 = keras.layers.Conv2D(128, (3,3), activation=act_conv, kernel_initializer="he_normal", padding = "same")(c4)
    p4 = keras.layers.MaxPooling2D(2,2)(c4)

    c5 = keras.layers.Conv2D(256, (3,3), activation=act_conv, kernel_initializer="he_normal", padding = "same")(p4)
    c5 = keras.layers.BatchNormalization()(c5)
    c5 = keras.layers.Conv2D(256, (3,3), activation=act_conv, kernel_initializer="he_normal", padding = "same")(c5)

    F = keras.layers.Flatten()(c5)
    D = keras.layers.Dense(50)(F)
    D = keras.layers.Dropout(.2)(D)
    D = keras.layers.Dense(10)(D)
    D = keras.layers.Dropout(.2)(D)
    
    outputs = keras.layers.Dense(nb_output, activation=activation)(D)
                             
    model = keras.Model(inputs=[inputs], outputs=[outputs], name="Encoder-UNET")
    
    return model


def classify_images(images, model, classes_names= None):
    """
    Classify images through a TF model.
    
    Parameters
    ----------
    images (np.array): set of images to classify
    model (tf.keras.model) : TF/Keras model
    classes_names:dictionnary with names of classes
    
    Returns 
    --------
    predicted classes
    """
    results = model.predict(images) #predict for images
    classes = np.argmax(results, axis=1) #np.argmax is computed row by row 
    if classes_names is not None:
        classes = np.array(classes_names[classes])
    return classes 