# Bug triage using information fusion
Automated bug triage is especially important for large-scale software projects where the rate of reported bugs are high and require expert developers to fix them in a timely manner. Based on the severity of the reported bug, finding an appropriate developer with the necessary skill sets and prior experience of fixing similar bugs is challenging and can be an expensive process. To overcome this challenge, researchers have proposed several Machine Learning (ML) and Deep Learning (DL)-based automated bug triage techniques that utilize the available historical data on the reported bugs and the information of their fixers. However, there is a sufficient scope of improvement for the performance of these techniques. In this paper, we propose a novel DL-based technique that utilizes two set of features from the textual data of the reported bugs, namely the contextual information and the occurrence of repeating keywords. To mine these features, we develop Convolutional Neural Network (CNN) and Artificial Neural Network (ANN) modules, respectively. We then fuse these two set of extracted features to automatically assign bugs among developers, appropriately. To evaluate the effectiveness of our model, we conduct extensive experiments on eight benchmark dataset of open-source, real-world software projects. The experimental results demonstrate that our approach of information fusion is able to outperform performance of the previous models and improve automated bug triage.
# Architecture of Deep Bug Triage Model
![Architecture of Deep Bug Triage Model](https://github.com/dipongkor/bug-triage/blob/main/Deep%20Triage%20Model%20V2.jpg?raw=true)
