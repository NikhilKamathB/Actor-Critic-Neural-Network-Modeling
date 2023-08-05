# Actor-Critic-Neural-Network-Modeling

Training deep neural networks, such as transformers, deep vision models, etc. from scratch with sufficient data is undeniably expensive in terms of time and resources. However, the necessity of this level of depth can be questioned, especially when aiming to deploy models in real-time applications where low latency is crucial. To make deep networks usable for real-time scenarios, researchers have explored various approaches to reduce the computational burden. This repository aims to propose a hypothetical approach (Konwledge Distillation) that enables modeling shallow neural networks by guided gradient descent to achieve performance similar to that of deep networks for a given task. Note: This is just an experiment.

### Folder
```
Actor-Critic-Neural-Network-Modeling repo

|- src
    |- data.py (defines data loaders/generators)
    |- nn.py (holds all the neural network related classes - activations, linear layers, loss functions, etc.)
    |- pipeline.py (defines an end-to-end pipeline for ML training - fetch data -> process -> model building -> training -> testing -> predicting)
|- actor.py (represents the shallow neural net with performance similar to the critic)
|- critic.py (represents the deep neural net with higher performace)
```