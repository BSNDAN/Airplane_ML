import numpy as np
import pandas as pd
import pathlib  
import constants

from PIL import Image
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


def specifiy_problem():    
    """ 
    Choose output and build data loader dictionnarys corresponding
    
    Parameters
    --------
    
    Returns
    -----
    Output choosen in str, dictionnary which join label image and train/val/test set corresponding, dictionnary which join label image and output corresponding, number of occurences of output 
    """     
    choosen_output = input("which target do you want? (choose between manufacturer, variant and family) : ")

    if choosen_output in ["manufacturer", "variant", "family"]: 
        
        path = "images_" + choosen_output
        path_train, path_val, path_test, target = path + "_train.txt", path + "_val.txt", path + "_test.txt", choosen_output
    
        train_df = build_img_database(constants.PATH_METADATA + path_train, target).assign(split = "train")
        val_df = build_img_database(constants.PATH_METADATA + path_val, target).assign(split = "val")
        test_df = build_img_database(constants.PATH_METADATA + path_test, target).assign(split = "test")
    
        data = pd.concat([train_df, val_df, test_df])
        data = data.assign(output = LabelEncoder().fit_transform(data[target]))
        
        nb_output = data["output"].nunique()
    
        partition = {"train" : data[data["split"] == "train"]["images_id"].tolist(),
                    "val": data[data["split"] == "val"]["images_id"].tolist()}

        labels = data[["images_id", "output"]].set_index("images_id").to_dict()["output"]       

    else : 
        raise ValueError(f"don't recognize target : '{choosen_output}' ")
      
    print("-----------------")
    print(f"{choosen_output} choosen!")
        
    return  choosen_output, partition, labels, nb_output


def build_img_database(path, target):    
    """ 
    Build a pandas dataframe with target class and access path to images
    
    Parameters
    --------
    path: path pattern to read csv file containing images information.
    target (str): name of the target column.
    
    Returns
    -----
    A pandas dataframe, including target class and path to image.
    """    
    _df = pd.read_csv(path, sep="\t", names=["all"],  dtype = {"all" : str}) 
    
    _df["images_id"] = _df['all'].apply(lambda x: x.split(' ')[0]) # create id 
    
    _df[target] = _df['all'].apply(lambda x: ' ' .join(x.split(' ')[1:])) #create target variable 
    
    # path column is the access path to image
    _df['path'] = _df['images_id'].apply(lambda x: pathlib.Path('images') / (constants.PATH_IMAGES + x + '.jpg'))
    
    return _df.drop("all", axis=1)
