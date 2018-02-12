# DeepLearningHW

## Common techniques for improving model performance:
* Normalize inputs to [0, 1] to avoid high variance due to uneven scales
* Early stopping: train the model until performance on the validation set starts dropping (due to overfitting), and then retrain the model up to that epoch. 
* Data augmentation (especially useful on images):
    * Rotation
    * Scale
    * Shift
    * Shearing
    * Horizontal flip
    * Hertical flip
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
