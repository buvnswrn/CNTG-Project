# Controllable Neural Text Generation -Project
   For pretraining procedure, run 
-  Reconstruction_Model_Training.ipynb or use model from https://huggingface.co/buvnswrn/daml-t5-pretrain
-  Copy the Discriminator model file from https://drive.google.com/file/d/1hNIdR8mestU-mpLEjYUq_aC0FkeVaTe7/view?usp=sharing and use it for training or use the code snippet provided in Final_Training.ipynb

Once we have the above models, we can pretrain the model using the procedure given in Final_Training.ipynb file. Note: Watchout for which dataset you want to train and comment and uncomment the codes accordingly provided in the file (Since we need to change only dataset name and the text field names in tokenizer).
