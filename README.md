

# A PyTorch Implementation of EtinyNet

### What is this project?

This project implements EtinyNet (https://ojs.aaai.org/index.php/AAAI/article/view/20387) in PyTorch. 

Uses tiny-imagenet-200 to train and test the network.

This specify branch adds regularisation to the network in an attempt to improve test accuarcy.

Dropout and image cropping are used to achieve this.


### Blogs


### Where is the code?

The code is located in the following files:

* main.py - main entry to train EtinyNet
* EtinyNet.py - EtinyNet Network
* train_test.py - Functions to train and test EtinyNet
* etinynet_depthwise_layers.py - Building Blocks for EtinyNet

### Requirements

All pip packages needed can be found in requirements.txt
Additionally the early-stopping-pytorch module was used (https://github.com/Bjarten/early-stopping-pytorch)