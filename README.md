# LLM classifier

This model is a multiclass text classifier, which classifies a comment as a positive, negative, or neutral comment.

To train the model, we used **sentiment analysis dataset**. The dataset is open source and can be found in kaggle: <https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset>.

We used TensorFlow as the ML framework.

## Requirements

You need the following libraries to use the classifier:

- Tensorflow
- NumPy

How to install Tensorflow: <https://www.tensorflow.org/install>
How to install NumPy: <https://numpy.org/install/>

## Train

To train the model, open the *model_trainer.py* and there you can enter your own dataset with your own parameters.

Please be careful with the `VOCAB_SIZE` and `MAX_LEN` since they are related with tokenizer file and dataset respectfully:

- `VOCAB_SIZE`: Number of tokens one allow in the token dictionary. The whole project must use the same `VOCAB_SIZE`.
- `MAX_LEN`: Lenght of the each sample. Each sample needs to have same size since they enter the Neural Network with input neurons of `MAX_LEN`.

We cannot offer the cleaned datasets because of the large size for GitHub, but you can create them with *sentiment analysis dataset/cleaned_ds/dataset_cleaner.py*.

Be sure to open the project in root folder of the project, or the directories might not work.

To increase the training time, increase the `EPOCH` parameter.

One can play with different optimizers and `batch_size`s. More information on optimizers: <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>

## Classify

To use the final model run the *classifier.py*. It does not have an user interface and for now work on the terminal.
