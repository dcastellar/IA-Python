# IA-Python
Examples of a neuronal net for classify flowers

Train.py --> Python script for train a network.

  you need to indicate the data directory for images.
  --save_dir -> Directory to save checkpoint
  --arch -> choose model (default vgg16)
  --learning_rate -> model learning_rate
  --hidden_units -> Number of hidden layers
  --epochs -> Numbers of epochs train
  --gpu -> Use gpu for training
  
Predict.py --> Python script for predict the class of a flower using a neuronal network
  you need the path to the image
  and the file of the neuronal network trainned saved
  --top_k -> K most likely classes'
  --category_names -> mapping categories, a json file
  --gpu -> Use gpu for inference
  
 Image Classifier Project.ipynb
 
    a Jupiter notebook with the proyect for classifing a flower.
