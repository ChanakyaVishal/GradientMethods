# Parallel Gradient Methods Machine Learning

## 1. Sequential implementation of some signSGD algorithms.
* In this component of the project we have implemented the sequential version of various SGD algorithms and have compared it to the signSGD.
* This code is in the folder /codes/BTP_Code/Sequential_SGD/ in the repository.
* To run the code execute: **python3 SGD_caller.py** (Note: ignore any warnings)
* This would output the accuracy vs iterations graph for all the algorithms.
* The experiments are done on a logistic regression model with the Iris Dataset (We considered only 2 classes of the 3 classes in the dataset)
* We have precompiled .so file for the C code in the repository.

## 2. Implementation of Majority voting sign based algorithms in CUDA from scratch
* As a next step we implemented the SGD algorithms and their signed variants (with Majority voting) using CUDA for a parallel implementation.
* This code is in the folder /codes/BTP_Code/CUDA_SGD/ in the repository.
* We have added the precompiled files of the CUDA code in the repository. (But to recompile the CUDA code execute **./make.sh**)
* To run the code execute **./app**. [The initial loading would take a few seconds after which a prompt stating "Enter Type: " would appear indicating that arguments can be entered]
* The first argument ("Enter Type: ") specifies the type of optimizer to use for training, the various options along with the argument input are:
    * Vanilla SGD - 1
    * signSGD - 2
    * signADAM - 3
    * signSVRG - 4
    * signCumSVRG - 5 (Our own experimental version of SVRG with gradient accumulation)
* In this version of the code we are using the FashionMNIST dataset (https://github.com/zalandoresearch/fashion-mnist) and we utilise a logistic regression cost function.

## 3. Modifying Caffe library to work with sign gradients and some comparisons of their readings on different models
* We then ventured to try and modify the BVLC Caffe library to work with sign based optimizers.
* We were successful in modifying the Caffe library to verify the efficacy of using the sign of gradients (without Majority voting) to train a model. This required a thorough understanding of how Caffe worked, which we invested a significant amount of time in.
* The major challenge we faced in this task was to implement a majority voting accumulator, this was due to the rigid nature of Caffe which did not favour such an implementation. 
* However we still obtained results for signed versions (without Majority voting) of SGD, Nesterov, ADAM, Adagrad, Adadelta and contrasted these with the results of their corresponding unsigned versions.
* We conducted these experiments first on a simple custom CNN with the CIFAR 10 dataset and then on a Resnet56 model with the Cifar10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html)
* We have uploaded the code for the modified version of caffe here: https://drive.google.com/open?id=1pH0Lr8U0-9Rlf9cHq7s0lbf4DE-vX4co
* We have created a script to run all the algorithms one after the other, the scripts also store the intermediary model in a file to resume training from that point. The final results are stored in the root folder of the caffe codebase in a specific format (for example signAdam with a learning rate of 0.1 and max iterations of 20000 is stored under the filename "output_adam_s_lr0.1_iter20000.txt"). 
* To run the experiment test bed go to the root folder of Caffe and execute **./examples/cifar10/V_tests/train_all.sh** (Note: This is a very time consuming process as it runs all the algorithms for 20000 iterations) 
* Note: No output is printed directly on the terminal, the output is redirected to the specific file with an "output" prefix.

