# Prerequisites:
1) Python libraries: numpy,pandas,pytorch,pillow,numpy,opencv-python should be installed.
2) Discrete GPU supporting CUDA is preferred for better performance.

# Instructions for executing ResNet 18:

1) Modify line 47 of resnet18_MLP.py to `dataset = ImageDataset('raw_directory', 'enhanced_directory', transform=transform)` . Please ensure that the raw and enhanced images have the same name and no other type of file is present in the directories.

2) Now execute the python script and wait for it to complete. 

3) Now go to test_resnet18_MLP.py if you want to test the model. Make similar modifications to RAW_DIR at the start of script and modify line 83 to the correct file path for the stored model.

4) Now specify the directory where the predicted images should be stored in line 117. Now execute the script. The loss after prediction will be outputted to terminal.

5) Alternately, if you want to only predict the images, use predict_resnet18_MLP.py after making similar modifications to step 3 and step 4.

