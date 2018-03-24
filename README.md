# DeepLearningHW

## Reinforcement Learning
* [Asynchronous Advantage Actor-Critic](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
* [Q-Learning](https://link.springer.com/article/10.1007/BF00992698)
* [Policy Gradient](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf)
* [Monte Carlo Tree Search](http://mcts.ai/pubs/mcts-survey-master.pdf)
* [Deep Q Network](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Approximate Policy Iteration](https://link.springer.com/article/10.1007/s11768-011-1005-3)
* [DeepMind StarCraft II](https://arxiv.org/abs/1708.04782)
* [MSC: A Dataset for Macro-Management in StarCraft II](https://arxiv.org/abs/1710.03131)
* [AlphaGo](https://www.nature.com/articles/nature16961)
* [AlphaZero](https://www.nature.com/articles/nature24270)

## Generative Models
* [Variational Autoencoder](http://kvfrans.com/variational-autoencoders-explained/)
* [Generative Adversarial Network](https://deeplearning4j.org/generative-adversarial-network)
* [Wasserstain GAN](https://arxiv.org/abs/1701.07875)
* [Image Style Transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

## Transfer Learning
* **Similar & small dataset**: avoid overfitting by not fine-tuning the weights on a small dataset, and use extracted features from the highest levels of the ConvNet to leverage dataset similarity.
* **Different & small dataset**: avoid overfitting by not fine-tuning the weights on a small dataset, and use extracted features from lower levels of the ConvNet which are more generalizable.
* **Similar & large dataset**: with a large dataset we can fine-tune the weights with less of a chance to overfit the training data.
* **Different & large dataset**: with a large dataset we again can fine-tune the weights with less of a chance to overfit.

## Common techniques for improving model performance:
* Normalize inputs to [0, 1] to avoid high variance due to uneven scales
* Add Batch Normalization layers between the convolution layer and activation function to avoid internal covariate shift
   * [BatchNorm](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization)
* Larger batch sizes stablize the training process at the cost of GPU memory consumption
* Consider using learning rate exponential decay or learning rate scheduler
* Consider ELU\Leaky ReLU as alternative options for ReLU activation function to avoid dead neurons
   * [ELU](https://www.tensorflow.org/api_docs/python/tf/nn/elu) (Exponential Linear Unit)
   * [Leaky ReLU](https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu)
* Consider skip connections between convolution blocks to share features and avoid vanishing gradients
   * [Residual Network](https://arxiv.org/abs/1512.03385) (CVPR 2015 Best Paper)
   * [Densely Connected Convolutional Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) (CVPR 2017 Best Paper)
* Early stopping: train the model until performance on the validation set starts dropping (due to overfitting), and then retrain the model up to that epoch. 
* Use pre-trained model on large datasets and finetune it for our purpose, instead of training it from scratch
* Use k-fold cross validation to avoid statistical error on performance, especially when the amount of data available is small
* Data augmentation (especially useful on images):
    * Rotation / Scale / Shift / Shearing / Horizontal flip / Vertical flip / Noise
* Overfitting vs. Underfitting:
    * Overfitting: Good performance on the training set, but poor performance on the validation set. 
        * Add more data
        * Increase regularization/dropout
        * Reduce model complexity (layers, nodes)
    * Underfitting: Poor performance on both the training set and the validation set. 
        * Add more data
        * Relax regularization/dropout
        * Increase model complexity (layers, nodes)
