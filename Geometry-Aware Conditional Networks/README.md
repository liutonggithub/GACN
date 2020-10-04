# Geometry-Aware Conditional Networks

##Pre-requisites
 (1) Python 3.6.
 
 (2) Scipy.
 
 (3) TensorFlos (r0.12) .
 
 *Please note that you will get errors if running with TensorFlow r1.0 or higher version because the definition of input arguments of some function have changed, e.g., 'tf.concat', 'tf.nn.sigmoid_cross_entropy_with logits', and 'tf.nn.softmax_cross_entropy_with_logits'*

 ##Datasets
 (1) You may use any dataset with labels of expression and pose. In our experiments, we use Multi-PIE, KDEF, and RaFD. 
 
 (2) It is better to detect the face before you train the model.

 ##Training
 ```
 $ python mainexpression.py
 ```

 During training, two new folders named 'PFER', and 'result', and one text named 'testname.txt' will be created. 

 'PFER': including four sub-folders: 'checkpoint', 'test', 'samples', and 'summary'.

 (1) 'checkpoint' saves the model
 
 (2) 'test' saves the testing results at each epoch (generated facial images with different expressions based on the input faces).
 
 (3) 'samples' saves the reconstructed facial images at each epoch on the sample images. 
 
 (4) 'summary' saves the batch-wise losses and intermediate outputs. To visualize the summary.
 
 *You can use tensorboard to visualize the summary.*
 
```
 $ cd PFER/summary
 $ tensorboard --logidr . 
```

 *After training, you can check the folders 'samples' and 'test' to visualize the reconstruction and testing performance, respectively.*

