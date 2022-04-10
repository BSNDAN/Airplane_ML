import io
import yaml

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import tensorflow as tf
import pandas as pd
from PIL import Image

from constant import manufacturer_class, variant_class, family_class


with open('app.yaml') as yaml_data:
    parameters = yaml.safe_load(yaml_data)

IMAGE_WIDTH = parameters[0]['IMAGE_WIDTH']
IMAGE_HEIGHT = parameters[1]['IMAGE_HEIGHT']
IMAGE_DEPTH = parameters[2]['IMAGE_DEPTH']
MODEL_PATH = parameters[23]['MODEL_PATH']

def load_image(path):
    """Load an image as numpy array
    """
    return plt.imread(path)

def predict_image(path,model  ):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction 
    
    Returns
    -------
    Predicted class
    """
    images = np.array([np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))])
    prediction_vector = model.predict(images)
    predicted_classes = np.argmax(prediction_vector, axis=1)
    return predicted_classes[0],prediction_vector

def load_model(path):
    """Load tf/Keras model for prediction
    """
    return tf.keras.models.load_model(path)


st.title("airplanes identification")

problem = ["manufacturer","variant","family"]
choose_pb = st.radio("choose what kind of prediction",problem)

if choose_pb == "manufacturer":
    model,problem_class = load_model(MODEL_PATH + "/manufacturer.h5"), manufacturer_class
    
elif choose_pb == "variant":
    model,problem_class = load_model(MODEL_PATH + "/variant.h5"), variant_class
    
else : 
    model, problem_class = load_model(MODEL_PATH + "/family.h5"), family_class

uploaded_file = st.file_uploader("load an airplane image")

if uploaded_file:
    loaded_image = load_image(uploaded_file)
    st.image(loaded_image)
      
predict_btn = st.button("Identify", disabled=(uploaded_file is None))
if predict_btn:
    prediction, probabilities = predict_image(uploaded_file, model)
    st.write(f" {problem_class[prediction]} is the most probable option. Probability : {round(np.max(probabilities)*100, 2)}%")

details_btn = st.button("Identify with details")    
if details_btn:
    
    prediction, probabilities = predict_image(uploaded_file, model)
    st.write(f" {problem_class[prediction]} is the most probable option. Probability : {round(np.max(probabilities)*100, 2)}%")
    
    df = pd.DataFrame(data = probabilities.tolist()[0], index = problem_class, columns=["probability"]).reset_index()
    
    sns.set(font_scale = 2)
    fig = plt.figure(figsize=(25,35))
    sns.barplot(x="probability",y="index",data=df)
    plt.xlabel("probability", fontsize=45)
    plt.ylabel("index", fontsize=45)
    st.pyplot(fig)
