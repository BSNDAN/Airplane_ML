# Project MAG3 AMSE : "Airplane classification"

# Project MAG3 AMSE : "Airplane classification"
### Description of the project : 
Build **Neurals Netowrks** and **an interface** to classify the manufacturer, variant and family of airplanes.

## To start the project properly
* clone this repository
* download constant.py files with the label of each classes : the name of the panels.
* check requirements files.


## To reproduce models 
* for the Neural network :
    - the whole code is on the folder notebook and it's **train.ipynb**
    - you can also just load models : it is on the models folder and it's **models/target.h5** (target is manufacturer/variant/family)


## To reproduce the application 
* You need to download the application folder with :
    - a parameters file app.yaml read at the start of the application containing the path to the file, the resize of the images 
    - a python file with all the code 


## Manipulation to be done directly on the computer environment : 
If you are using a Google collab for the neural network model and you want to reload the model on a Jupyter lab, you have to proceed in the same way : 
1. From an anaconda prompt, create a new environnement 
`` conda create --name airplane python=3.8``
`` conda install pandas tensorflow scikit-learn seaborn pillow``
2. Activate this environnement 
``conda activate airplane``
3. Launch python 
``from tensorflow import keras``
``classifier = keras.models.load_model('models.target.h5')`` 
4. Install the module that allows Jupyter to talk with the Python environment
``conda install ipykernel``
``python -m ipykernel install --user --name=airplane``

For the application, we have to install streamlit : 
1. Install streamlit in the airplane environment :
``conda install -c conda-forge streamlit``

## License

Distributed under the MIT License.

## References
• Data : https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

## Authors
• Collaboration with Nastasia Henry for project at Aix-Marseille School of Economics

 --  Nastasia Henry - @henry_nastasia - nastasia.henry@etu.univ-amu.fr
 
 --  Dan Busnach -dan.busnach@etu.univ-amu.Fr 
