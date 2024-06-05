Instructions and analysis for regression:

The regression.ipynb is a combined python notebook for the regression model for auto color correction of images

Just change 3 things in the 2nd cell of the python nbk to train on different set of images by photographer

raw_folder_path = "../../adobe5k/raw"
edited_folder_path = "../../adobe5k/a"
csv_file_path = "adobe5k_a.csv"

and to apply this model to new set of photos 
make changes here(last but one cell):

test_data = "test_data.csv"
test_folder_path = "../../adobe5k/test"
output_folder = "../../adobe5k/test_a_output"

The final predicted images will be stored in test_a_output folder

These are basically trained on adobe5k data set with photographers a b and c

The csv files in this folder correspond to the same
The testing is done on some images from selected wedding scenes to be displayed on the Website

Conclusion: To run Regression - just run this python notebbok with correct folder paths

Good points : 
The time to train is roughly around 7 mins for 5000 photos
The testing time is roughly 4-5 seconds 
This is a good amount of time for a quick edit 

Bad points:
The quality of editing is not that great and is influenced by the training dataset largely
