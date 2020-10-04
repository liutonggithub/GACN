# Geometry-Aware Conditional Networks

##Pre-requisites
 
 (1) Python 3.6.
 
 (2) Scipy.
 
 (3) TensorFlows (r1.12).
 

 ##Datasets
 
 (1) You may use any dataset with labels of expression and pose. In our experiments, we use Multi-PIE, KDEF, and RaFD. 
 
 (2) It is better to detect the face before you train the model.

 ##Training
 ```
 $ python main.py
 ```

 During training, two new folders named 'PFER', and 'result', and one text named 'testname.txt' will be created. 
 
 *You can use tensorboard to visualize the summary.*
 
```
 $ cd PFER/summary
 $ tensorboard --logidr . 
```

 *After training, you can check the folders 'samples' and 'test' to visualize the reconstruction and testing performance, respectively.*

