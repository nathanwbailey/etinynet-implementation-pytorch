

# A PyTorch Implementation of EtinyNet

### What is this project?

This project implements EtinyNet (https://ojs.aaai.org/index.php/AAAI/article/view/20387) in PyTorch. 

Uses tiny-imagenet-200 to train and test the network.


### Blogs

In addition to the code I wrote 2 blogs on EtinyNet.

The first explained the architecture:

https://nathanbaileyw.medium.com/etinynet-explained-a-size-reduction-does-not-always-indicate-a-drop-in-accuracy-75a78707bc0a

The second detailed the implementation:

https://nathanbaileyw.medium.com/implementing-etinynet-1-0-in-pytorch-01ce18dbf2c2


### Where is the code?

The code is located in the following files:

* main.py - main entry to train EtinyNet
* EtinyNet.py - EtinyNet Network
* train_test.py - Functions to train and test EtinyNet
* etinynet_depthwise_layers.py - Building Blocks for EtinyNet

### Requirements

All pip packages needed can be found in requirements.txt

Additionally the early-stopping-pytorch module was used (https://github.com/Bjarten/early-stopping-pytorch)