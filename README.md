# Project MAG3 AMSE : "Airplane classification"
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

## Description of the project
The aim of this project is to recognize and classify the manufacturer, variant and family of an airplaines from images using Neural Networks with Keras. In this project, deployement of a streamlit app is available for testing aircraft's identification.



## Install
• clone this repository

• download constant.py files with the label of each classes : the name of the panels.

• check requirements files.

• Build train and test set

• Train the model

• Save the model



## Manipulation to be done directly on the computer environment :
If you are using a Google collab for the neural network model and you want to reload the model on a Jupyter lab, you have to proceed in the same way :

1. From an anaconda prompt, create a new environnement [conda create --name airplane python=3.8] [conda install pandas tensorflow scikit-learn seaborn pillow]
2. Activate this environnement [conda activate airplane]

3. Launch python from [tensorflow import keras] [classifier] = [keras.models.load_model('models.target.h5')]

4. Install the module that allows Jupyter to talk with the Python environment conda install ipykernel python -m ipykernel install --user --name=airplane


For the application, we have to install streamlit :

Install streamlit in the airplane environment : conda install -c conda-forge streamlit




## License

Distributed under the MIT License.

## References
• Data : https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

## Authors
• Collaboration with Dan Busnach (@BSNDAN) for project at Aix-Marseille School of Economics

 --  Nastasia Henry - @henry_nastasia - nastasia.henry@etu.univ-amu.fr
 
 --  Dan Busnach -dan.busnach@etu.univ-amu.Fr
