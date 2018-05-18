Implementation of SVM active proposed [here](http://infolab.stanford.edu/~echang/svmactive.pdf).

Needs `keras`, `PIL`, `pandas` and a few other common packages.

USAGE
-----

First, have a look at the `config.yaml` file.

Execute `python get_data_ready.py` to get the data ready.  This will download 
`tiny-imagenet-200.zip`, unzip it into a folder and create an `npy` file
of the loaded feature vectors obtained by using `keras`.

Then execute `python run_svm_with_feedback.py`.  This will create a plot
using the configuration specified in `config.yaml` which shows the performance
of the SVM when samples to get feedback from are chosen using the algorithm
in the paper or chosen randomly.

The SVM which is used in this implementation is the one used by [SMQTK.](https://github.com/Kitware/SMQTK/tree/viame/master/TPL/libsvm-3.1-custom) The path is specified in `config.yaml`.


