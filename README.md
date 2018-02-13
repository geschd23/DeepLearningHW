# DeepLearningHW

## Common techniques for improving model performance:
* Normalize inputs to [0, 1] to avoid high variance due to uneven scales
* Add Batch Normalization layers between the convolution layer and activation function to avoid internal covariate shift
   * [BatchNorm](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization)
* Larger batch sizes stablize the training process at the cost of GPU memory consumption
* Consider ELU\Leaky ReLU as alternative options for ReLU activation function to avoid dead neurons
   * [ELU](https://www.tensorflow.org/api_docs/python/tf/nn/elu) (Exponential Linear Unit)
   * [Leaky ReLU](https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu)
* Consider skip connections between convolution blocks to share features and avoid vanishing gradients
   * [Residual Network](https://arxiv.org/abs/1512.03385) (CVPR 2015 Best Paper)
   * [Densely Connected Convolutional Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) (CVPR 2017 Best Paper)
* Early stopping: train the model until performance on the validation set starts dropping (due to overfitting), and then retrain the model up to that epoch. 
* Use pre-trained model on large datasets and finetune it for our purpose, instead of training it from scratch
* Consider learning rate exponential decay or learning rate scheduler
* Data augmentation (especially useful on images):
    * Rotation
    * Scale
    * Shift
    * Shearing
    * Horizontal flip
    * Vertical flip
    * Noise
* Overfitting vs. Underfitting:
    * Overfitting: Good performance on the training set, but poor performance on the validation set. 
        * Add more data
        * Increase regularization/dropout
        * Reduce model complexity (layers, nodes)
    * Underfitting: Poor performance on both the training set and the validation set. 
        * Add more data
        * Relax regularization/dropout
        * Increase model complexity (layers, nodes)
