# Bug Triage using Information Fusion

# Architecture of Deep Bug Triage Model
![Architecture of Deep Bug Triage Model](https://github.com/dipongkor/bug-triage/blob/main/Deep%20Triage%20Model%20V2.jpg?raw=true)

# Benchmark Models
We create six benchmark models using ablation study to justify whether it is possible to achieve the same performance by removing any of these modules.

### ANN+CNN2+CNN3
This model is consisted of a single ANN layer and two CNN layers having different filters. For instance, CNN2 and CNN3 capture contextual relationship between 2 and 3 words, respectively. Then, max-pooling is used to find important features from CNN layers. The Source code of this model is located [here](ANN-CNN2-CNN3.ipynb).

### ANN+CNN2
This model also aims to observe the performance of IF, however, it has less filter than above benchmark model. It is developed by removing CNN3 from the above ANN+CNN2+CNN3. The source code of this model is located [here](ANN-CNN2.ipynb)

### ANN
There is no IF in this model, i.e., it only learns from repeating keywords. This benchmark model is developed by removing CNN2 and CNN3 from ANN+CNN2+CNN3. The aim of this model is to triage bugs without IF. This model takes repeating keywords (TF-IDF vectors) as input and assigns weights to the input in its dense layer. Finally, softmax layer triages the bug reports. The source code of this model is located [here](ANN.ipynb)

### CNN2+CNN3
This model learns from the contextual relationship of consecutive words. Again, for instance, CNN2 and CNN3 capture contextual relationship between 2 and 3 words, respectively. Then, important features are selected using max-pooling operation. The outcome of pooling operation is processed via a dense and softmax layer in order to triage bug reports. This model is developed by removing ANN from ANN+CNN2+CNN3. The source code of this model is located [here](CNN2-CNN3.ipynb)

### CNN2 and CNN3, individually
As CNN2+CNN3 uses contextual relationship between both 2 and 3 words, we develop further two individual models such as CNN2 and CNN3. The aim of these model is find whether it is 2 or 3 consecutive words should be the choice for CNN layers. [CNN2 Source Code](CNN2.ipynb) and  [CNN3 Source Code](CNN3.ipynb)

## Traditional ML Models
Apart from the above benchmark models, we train [traditional ML models](Traditional-ML.ipynb) such as SVM, RF and NB to compare our DL models' results against them. To train these models, we use tf-idf vectors (reduced using PCA) as features. In other words, our ANN and ML models are trained using the same set of features. We consider these models because we find that they were widely used in existing studies for solving the bug triage problem. 
