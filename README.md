# FruitHarvestingSystem
A fruit harvesting system based on a modified version of Faster RCNN  `object detection` `deep learning` architecture<br><br>
In this implementation, we are using Tensorflow/Keras frameworks.<br><br>
This work is still `incomplete`.<br>
We have already done the data `preprocessing` part and implemented the Region Proposal Network (RPN) in the original `FasterRCNN` paper: https://arxiv.org/abs/1506.01497v3 .<br><br>

We used `pre-trained ResNet50` weights for transfer learning as the backbone of our architecture.<br>
Here is a link to the weights: https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 <br><br>


Next, we are going to use this Region Proposal Network as an attention mechanism for the `regression` and `classification` heads as in the original paper.<br>
Then another head, called the `Picking Point Estimator` head, will be added to the model.<br><br>
This is the dataset currently being used: https://strawdi.github.io/


