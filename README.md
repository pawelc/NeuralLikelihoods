# NeuralLikelihoods
Source code for Neural Likelihoods via Cumulative Distribution Functions paper

Currently only the code for [experiments/density/synthetic](experiments/density/synthetic) is working because
the code is being rewritten from Tensorflow 1.x to Tensorflow 2.0.

Please create conda environment from provided environment.yml.

# Training models in a notebook on simple generated data. 

Open notebook and choose the model to train:
1) [experiments/density/synthetic/sin_normal/test.ipynb](experiments/density/synthetic/sin_normal/test.ipynb)
1) [experiments/density/synthetic/sin_t/test.ipynb](experiments/density/synthetic/sin_t/test.ipynb)
1) [experiments/density/synthetic/inv_sin_normal/test.ipynb](experiments/density/synthetic/inv_sin_normal/test.ipynb)
1) [experiments/density/synthetic/inv_sin_t/test.ipynb](experiments/density/synthetic/inv_sin_t/test.ipynb)
1) [experiments/density/synthetic/mv_nonlinear/test.ipynb](experiments/density/synthetic/mv_nonlinear/test.ipynb)

# Running experiments:
1) Experiment for each model/data set can be run by executing its script, for example 
[experiments/density/synthetic/inv_sin_normal/monde_ar.py](experiments/density/synthetic/inv_sin_normal/monde_ar.py).
User has to adjust configuration options that are set on the conf object in the experiment. These options
depend on the hardware like available GPUs, available memory and number of CPUs.