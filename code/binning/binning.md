This is binning module - another model to auto color correct the images is binnig
The idea behind this is to iterate over pixels and map them to 0-255 bins based on the RGB values
The performance is quite good as the MSE is pretty less 
But there is a huge overhead for time in training itself
Hence this idea was dropped 
There are 2 python notebooks : one for extraction of the parameters and one for training as calculating the MSE
(No implementation has been done to apply to new images) - as training time is itself so high, we dropped the idea

The 2 csv files attached are on randomly chosen test data set

Conclusion: to run binning just run the 2 python notebooks with correct folder paths
