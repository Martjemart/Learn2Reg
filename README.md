# Learn2Reg
the model directory is only to save the model pth, all model specifics can be found in the code. 
The code is split into Functions.py, test_cLapIRN_lite.py, Train_cLapIRN_lite.py, miccai2021_model_lite.py and results_analysis.ipynb.
Functions.py is there to deal with loading files and all calculations outside of the model.
The miccai2021_model_lite.py has the model specifics and the loss functions, except for the loss function of the keypoints.
The loss for the keypoints can be found in the Train_cLapIRN_lite.p file. To explore our code you should start in this file and relate the functions from outside of the file to the code here.
The test function is a short code to save the results of the model over a file. 
The results_analysis.ipynb shows how we got figure 2 from the paper.
