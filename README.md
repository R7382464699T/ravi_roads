#Roads Detection in Satellite Images
========

Thsi repository implements the neural network architecture provided in the paper [Roads Extraction by Res U-Net](https://arxiv.org/abs/1711.10684.pdf).

The performance of the model can be seen in the predicted output image below.

![sample image](/sample_images/sample.tiff)
![predicted output](/sample_images/predicted_output.tiff)



##Steps for running the model

1. Create the environment using the environment.yml file. The steps for creating the environment are given [here.](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

	> For creating the environment manually, the requirements are given in Requirements.txt file.

2. Run train.py file for creating and training the model.The dataset link can be obtianed from the paper.

3. Use predict.py for predicting the result for a test image.


## Hardware & Dependencies

Nvidia GPU
Tensorflow-GPU
OpenCV
python3.5


	 
