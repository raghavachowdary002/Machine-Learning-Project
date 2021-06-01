# Machine-Learning-Project
A sound wave upon its transmission doesn’t just  stop after reaching the end of the medium. Certain behaviors will be exhibited by the sound wave such as reflection off the 
obstacle, diffraction around the obstacle, and transmission through the obstacle. By recording the reflected signal, a time signal can be generated. This time signal results from the convolution of the incident acoustic wave with the surface properties of the reflecting object. By analyzing the reflected signal one can conclude on the reflecting object. The Machine learning (ML) techniques have enabled significant advancements in automated data processing and pattern recognition capabilities in a variety of fields. ML in acoustics is a promising development with many credible solutions to the acoustics challenges. This project reviews the ML model called MLP (Multi-Layer Perceptron) to discriminate the reflected sound signal using the time signal and to predict the type of reflecting object. 

# Project Architecture

![ml](https://user-images.githubusercontent.com/84661500/120301528-3d739000-c2cd-11eb-8b6d-56bc399d5714.jpg)

**1. Collecting Dataset:** We used dataset for reflected waves from object1, object2, object3. This dataset contains 3400 column sound excerpts. There are 3 different classes in the   dataset, i.e., Object #1, Object #2, Object #3. 

**2. Data Preprocessing:** Data Processing is an important task when preparing data for its processing in machine learning algorithms. we have done addition of some relevant   attributes, labels, and addiction of target column for training model which also ensures the error avoidance and productivity increase.

**3. Importing Libraries:** This step involves the various libraries importing in python environment which will help in numerical and graphical analysis. While working on model we need to plot visuals for different functions which works along with training and testing of algorithms used in this project. There are some of the important libraries such as pandas, seaborn, numpy, sklearn.  

**4. Training Model:** We are training our model using Multi-layer Perceptron classifier model in this project. As shown in Figure 9, first layer is taken from data and it is visible layer of model. The second layer of model is called as hidden layer as it is not visible to input, and which can consist multiple layers of neurons in it. The last hidden layer is also known as an output layer which is accountable for generating an output targeted values of given problem. 
    ![mlp](https://user-images.githubusercontent.com/84661500/120302693-6f392680-c2ce-11eb-9f8e-0a78de7c21d3.png)

The actual concept of neural network model which is specifically known as Multi-layer Perceptron classifier, is to obtain linear combinations of the given inputs as derived features and then model the target as a nonlinear function of these features. We used the MLPClassifier function in the sklearn Python library for MLP. We selected stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba as the solver forweight optimization (ie, solver = “adam”) because of large dataset. We considered hyperparameters, including different hidden layer sizes and activation functions.

**5. Prediction:** This field consist of replication of biological brain model to resolve the computational problems n machine learning. The purpose of this model creation is not redesigning the real model like brain, but to form an algorithm to resolve the problems with various data structure. The understanding of neural network comes by their representation ability of learning training data and relate it to the output or target data that we want to predict. larger weights prone to complexity of network and fragility of model. It is always expected to keep your model weights minimum to reduce complexity. Activation function which is also called as transfer function in mathematical terms helps all weights to passed through t for further process. In this scenario neural networks trying to learn a mapping. The hierarchical and multilayer structure of MLP classifier helps model’s prediction capability. 
**6. Measurements:** Performance of the multilayer neural network classifier evaluated using parameters such as precision, sensitivity and specificity and the receiver operating curves (ROC). We have achieved an accuracy of 92%.

# Application Testing:

  



