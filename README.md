# deepCR HST ACS/WFC

To reproduce the models, run ``deepCR-ACS-WFC.py`` in the command line. It will sequentially execute
1. Downloading the training and test datasets
2. Fetching the training data and test directories.
3. Training and testing the models. You'll need [scalable_datasets branch](https://github.com/profjsb/deepCR/tree/f017545e34559db93a8fdffa239f60d367fd9226) set up for this step.

    * The models that will be trained are ACS/WFC F434W, F606W, F814W individual filter models, and all-in-one global model.
    * The individual filter models are tested with respective test dataset, and the global model is tested with all three test datsets concatenated. Additionally, LACosmic will be tested with the individual test datasets.
4. Plotting the testing results. (ROC curves)

You'll likely need ~50GB disk space to download the training and test data, as well as GPU support to intiate training.
