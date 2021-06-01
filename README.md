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



