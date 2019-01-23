Roads Detection in Satellite Images
========

Thsi repository implements the neural network architecture provided in the paper [Roads Extraction by Res U-Net](https://arxiv.org/abs/1711.10684.pdf).

The performance of the model can be seen in the predicted output image below.

![sample image](/sample_images/sample.tiff)
![predicted output](/sample_images/predicted_output.tiff)



## Steps for running the model

1. Create the environment using the environment.yml file,  use the command 'conda env create -f environment.yml'.
You can refer [here](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for help.

	> To manually install the requirements, use the Requirements.txt file with the command 'pip install -r Requirements.txt'.

2. Run train.py file for creating and training the model.The dataset link can be obtianed from the paper.

3. Use predict.py for predicting the result for a test image.


## Hardware & Dependencies

Nvidia GPU
Tensorflow-GPU
OpenCV
python3.5


	 
