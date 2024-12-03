# XiaoFish
Install TensorFlow
To reproduce our experiments, please install TensorFlow 2.0. The experiments can be reproduced on machines with or without GPUs. However, training with CPU only is extremely slow and we do not recommend it. Inference with CPU only is slow but tolerable.

We suggest following the official instructions to install the library. The code has been tested on Ubuntu 16.04 + CUDA 10.0 + CUDNN 7.6.3.

We recommend using a virtual environment to avoid conflicts with other libraries. 

Here are the steps to install TensorFlow 2.0 by Window system:
```
conda create --name nl2bash python=3.9 -y
conda activate nl2bash
pip install tensorflow==2.10
pip install -r requirements.txt
```

Then the environment is ready to use. 

Run ```python load_model.py``` to load the pre-trained model. And test on the test set to get the bleu score.
Run ```python train.py``` to train the model on the training set. Using parse_args to set the hyperparameters.# XiaoFish
