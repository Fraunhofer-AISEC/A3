This project is not maintained. It has been published as part of the following conference paper at ECML-PKDD 2020:

# Activation Anomaly Analysis
## by Philip Sperl*, Jan-Philipp Schulze* and Konstantin Böttinger
\* Philip Sperl and Jan-Philipp Schulze are co-first authors.

Inspired by recent advances in coverage-guided analysis of neural networks, we propose a novel anomaly detection method.
We show that the hidden activation values contain information useful to distinguish between normal and anomalous samples.
Our approach combines three neural networks in a purely data-driven end-to-end model.
Based on the activation values in the target network, the alarm network decides if the given sample is normal.
Thanks to the anomaly network, our method even works in strict semi-supervised settings.
Strong anomaly detection results are achieved on common data sets surpassing current baseline methods.
Our semi-supervised anomaly detection method allows to inspect large amounts of data for anomalies across various applications.

### Citation
Activation Anomaly Analysis was published at ECML-PKDD 2020 [8].
If you find our work useful, please cite our paper:
```
Sperl P., Schulze JP., Böttinger K. (2021) Activation Anomaly Analysis. In: Hutter F., Kersting K., Lijffijt J., Valera I. (eds) Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2020. Lecture Notes in Computer Science, vol 12458. Springer, Cham. https://doi.org/10.1007/978-3-030-67661-2_5
```

### Dependencies
The software dependencies are listed in ``requirements.yml`` for ``conda`` and in ``requirements.txt`` for ``pip``.
We use Python 3.7.

We recommend using a conda environment:  
``conda env create python=3.7 --name A3 --file ./requirements.yml``.

Alternatively, using ``virtualenv``:  
``virtualenv -p python3 A3``,
``source A3/bin/activate``,
``pip install -r requirements.txt``.

### Instructions
#### Data sets
Except for MNIST and EMNIST, the raw data sets are stored in ``./data/``.
You need to add NSL_KDD [1], CreditCard [2] and IDS [3] manually from the respective website.

#### Train models
For each data set, all applicable experiments are bundled in the respective ``do_*.py``.
You need to provide a random seed and whether the results should be evaluated on the "val" or "test" data, e.g. ``python ./do_mnist.py 2409 val``.
(note that the evaluation function was implemented externally - the parameter "val" and "test" do not have any implication in this version)
By default, a Gaussian noise generator is used as anomaly network.
For MNIST, you can set the flag ```--use_vae 1``` to use a VAE instead.

For our experiments, we used the random seeds ``2409``, ``2903``, ``3003``, ``3103`` and ``706``.
The ROC curves show ``706``.
Please note that we trained the models on a GPU, i.e. there will still be randomness while training the models.
Your models are stored in ``./models/``.
All results used for the paper are found on our cloud storage [4].

For ``do_mnist_emnist.py`` please specify a suitable target model (e.g. the classifier trained by ``keras_mnist.py``) by setting ``-target_path``.

#### Evaluate models
We use a separate script to evaluate the models, ``evaluate_models.py``.
As parameter, you need to provide a random seed (the very same you trained the model on).
Moreover, you need to provide the correct suffix, e.g. ``--folder_suffix _2409``.
Note that the script expects the same folder structure as we used [4], i.e. a subfolder per data set.
Otherwise, you might need to adapt ``path`` in ``evaluate_models.py`` for the respective experiment.

Also here, by default, a Gaussian noise generator is used as anomaly network.
For MNIST, you can set the flag ```--use_vae 1``` to use a VAE instead.

The script generates one ROC curve per experiment and the file ``all_results.csv`` where all test results are stored.
If you like to combine these results (as we did for our results section), use ``evaluate_results.py``.
This generates the files ``mean``, ``std`` and ``count`` in ``./results/``.
Specify the folder where your ``all_results.csv`` are by setting ``--subfolder``.

#### Baselines
We compared the performance of A3 to the state-of-the-art anomaly detection methods Isolation Forest, DAGMM and DevNet.
For the latter methods, please add the respective implementation by Nakae [6] and Pang [7] to the ``./baselines/`` folder.

### File Structure
```
A3
│   do_*.py                     (start experiment on the respective data set)
│   evaluate_models.py          (calculate the metrics on the trained models)
│   evaluate_results.py         (calculate the mean over the test results)
│   parameter_mnist.py          (grid search on some parameters on MNIST)
│   do_mnist_manually.py        (example how to use the A3 framework on MNIST)
│   keras_mnist.py              (classifier target used in do_mnist_emnist.py, please use the code from [5])
│   requirements.yml            (dependencies for conda)
│   README.md                   (file you're looking at)
│
└─── data                       (raw data)
│   │
│   └───creditcard              (Credit Card Transactions, [2])
│   └───IDS                     (IDS, [3])
│   └───NSL-KDD                 (NSL-KDD, [1])
│
└─── libs
│   │   A3.py                   (main library for our anomaly detection method)
│   │   DataHandler.py          (reads, splits, and manages the respective data set)
│   │   ExperimentWrapper.py    (wrapper class to generate comparable experiments)
│   │   Metrics.py              (methods to evaluate the data)
│
└─── baselines                  (baseline methods)
│   │
│   └───dagmm                   (DAGMM, [6])
│   └───devnet                  (DevNet, [7])
│ 
└─── models                     (output folder for the trained neural networks)
│
└─── results                    (measurements used for the paper)
│
```

### Additional remarks
Please note that we fixed minor bugs for the final version of our paper.
For a sound evaluation, we repeated all experiments on the new codebase.
The provided code and models reflect the results in final version, not the preprint on arXiv.

### Links
* [1] https://www.unb.ca/cic/datasets/nsl.html
* [2] https://www.kaggle.com/mlg-ulb/creditcardfraud
* [3] https://www.unb.ca/cic/datasets/ids-2018.html
* [4] https://1drv.ms/u/s!AgYaaBPIqOAdkuIfPAqteV40Rl53sA?e=55CtZS
* [5] https://keras.io/examples/mnist_cnn/
* [6] https://github.com/tnakae/DAGMM
* [7] https://sites.google.com/site/gspangsite/sourcecode
* [8] https://link.springer.com/chapter/10.1007%2F978-3-030-67661-2_5
