# Readme
This file will tell you all you need to know to navigate and use this project.

This project is also availible under: "https://github.com/MattAlex99/Master-tf"

# Instalation
You will need to install tensorflow to train models. The official tensorflow website offers 
information on how to install it (https://www.tensorflow.org/install/pip#windows-wsl2). Tenorlow GPU is reccomended,as
CPU training takes to long. Installing Tensorflow GPU is notoriously difficult on Windows. If oyu use a 
windows device, I recomend using the WSL2 installation. It takes some time but works consistently. Pay 
close attention to the CUDA and driver versions you install. The guide linked above will tell you exactly what you 
need to do.  

If you use pycharm, you can also use the virtual environment, provided in the venv folder.

## Packages and files 
- Required for training
  - cv2 
  - matplotlib.pyplot 
  - numpy 
- For model visualization in weight evaluation
  - graphviz (this can be difficult on windows)
  - pydot

# Important scripts and project structure 
- **application** is a package, that includes some scripts, that can be used to apply the 
   model to some dataset and visualize the result. Scripts for segmentation and graspability 
   models are provided.
- **data_manipulation_test** includes scripts, that process a given dataset change it in some way.
  - generate_random_dataset.py takes a dataset of images a and 4DOG grasp labels, and applies some 
    transforms to images and labels. This can be used to gain more comprehensive training data for dish4 models
  - maks_exr_to_png.py simply transforms a dataset of exr images to a dataset of png images
- **model_evaluation** is a package, that includes scripts for evaluating the attentional weights of a trained model
  - weight_evaluation.py is the script, that performs the actual evaluation. It reads the weights 
    calcualates averages and creates a histogram that visualizes the result. 
  - visuazlizazion_of_attentinoal_weight_impact.py does NOT read information from a model. It simluates
    a nueral network with weights of a certain distribution, to show how big the impact of attentional weights
    is for the activation of featuers
- **test** includes some scripts, that I have used to make sure, some components work fine. You can safely 
  ignore this package
- **the main folder**
  - train_graspability_model.py is the main training file. The structure of the neural network and the 
    training process are defined here.
  - train_on_custom_data.py is a file, that manages the training of semantic segmentation models. 
    It is very similar to train_graspability_model.py
  - helpers.py contains the logic of how tf.datasets are read and pre processed, before training.
  - Usefull_blocks.py defines the structure of the Resplacement Block, the Squeeze and Excitation 
    block, the encoder Fusion block as well as some others, which were not used in the final model. 
  
    
# Preparing datasets
This project does not contain scripts for dataset Creation. check out the other project 