#  MAG3 AMSE Project : "Airplane classification"
## Description: 
This project is made for a class project.
It aims to build **Neurals Netowrks** which classify the manufacturer, variant and family of airplanes.
The important part of this project is the **code structure** and the **streamlit interface**, 
model optimization is not important. 

## To start the project properly:
* Clone this repository.
* Download data : https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/.
* Drag and drop data into data folder.


## To reproduce models:
* In modelisation folder, execute main_train.py. You can customize model architecture in model.py or batch
* you can also just load models : it is on the saved_models folder and it's **models/target.h5** file. (target could be manufacturer/variant/family)


## To reproduce the application:
* You need to download the application folder with :
    - a parameters file app.yaml read at the start of the application containing the path to the file, the resize of the images 
    - a python file with all the code 

## License:
Distributed under MIT License.

