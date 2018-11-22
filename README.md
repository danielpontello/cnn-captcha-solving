# cnn-captcha-solving

This repository contains our final thesis for the Computer Engineering graduation at the [National Telecommunications Institute](http://inatel.br/), titled "Evidencing CAPTCHA Vulnerabilites using Convolutional Neural Networks" (available as the 'paper.pdf' file; in Portuguese).

![Block diagram](https://github.com/danielpontello/cnn-captcha-solving/blob/master/results/block-diagram.png)

## Abstract

This document aims to describe the development process of a Convolutional Neural Network that seeks to assess the reliability of CAPTCHAs, security mechanisms present in several websites. It is presented a theoretical revision about the technologies used and related scientific works. We also present the experiments, metrics and details of the construction and operation of the Neural Network. In the end, we present the work's results.

## Directory Structure

This project is comprised of the following directories:

 - **dataset-generator**: Contains code for the generation of the artificial dataset used on the training of the neural network.
 - **neural-network**: Contains code used for training and testing of the neural network.
 - **experiments**: Miscellaneous files scripts used on the project.
 - **results**: Results collected from the experiment.

## Running

To generate an artificial dataset for the training of the neural network, on the `dataset-generator` directory run the following command:

```shell
$ python fies-generate.py <number_of_samples>
```

Where `<number_of_samples>` is the number of sample CAPTCHAs to be generated. This command can take a long time to run, depending on the number of samples being generated. The images will be saved on the `dataset/raw` folder.

After generating the dataset, the images must be filtered and segmented to be used to train the neural network. To do that, run the following command:

```shell
$ python fies-filter.py
```

The segmented images will be saved on subdirectories of the `dataset/segmented` directory.

With our dataset ready, we can start training the network. To do that, on the `neural-network` folder, run the command:

```shell
$ python train-network.py
```

**WARNING: This step can consume large amounts of RAM (about ~8GB for 72000 segmented images). Close any unnecessary programs before running.**

Various parameters of the network can be changed by editing this script, as shown below:
```python
num_samples = 2000          # number of samples to use on training
epochs = 1024               # number of epochs of training
learning_rate = 1e-3        # learning rate of the network
batch_size = 128            # batch size
validation_split=0.66       # the train/validation split percentage to be used
min_delta = 1e-6            # minimum improvement of the validation accuracy before stopping training
patience = 10               # number of epochs without improvement before stopping training
```

The trained model will be saved to the `models` folder.

## Used Libraries:

The following libraries were used on this project:

- **[Numpy](http://www.numpy.org/)**: Scientific computing package for Python
- **[OpenCV](https://opencv.org/)**: Computer Vision library
- **[Keras](https://keras.io/)**: Machine learning library that runs atop Tensorflow
- **[TensorFlow](https://www.tensorflow.org/)**: High-performance machine learning library
- **[Pillow](https://python-pillow.org/)**: Image creation and manipulation library.
- **[PlaidML](https://github.com/plaidml/plaidml)**: Keras backend, used for enabling GPU acceleration on OpenCL-enabled devices
- **[Matplotlib](https://matplotlib.org/)**: Chart plotting library
- **[Memory Profiler](https://pypi.org/project/memory_profiler/)**: Memory Profiler for Python
## Authors

### Advisor
 - Marcelo V. C. Aragão ([https://github.com/marcelovca90](https://github.com/marcelovca90))

### Students

 - Daniel S. P. Neves ([https://github.com/danielpontello](https://github.com/danielpontello))
 - Fernanda C. Avelar
 - Karina V. V. Ribeiro