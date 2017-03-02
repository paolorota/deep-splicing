# **Deep Splicing**
This code is used to apply Deep Learning framework for image foresnics (splicing detection)


----------


# Dependencies
- Python 3.4.4
- [Theano](http://deeplearning.net/software/theano/)
- [Keras](http://keras.io/)
- [pwprutils](https://github.com/paolorota/pyprutils)

# Notes

Refer to `config.ini` for the configuration parameters and paths.

# How To

## `CASIA_edge_exractor.m`

It is a MATLAB script that exploits the [P.Dollar edge detection toolbox](https://github.com/pdollar/edges) to generate edges images.
This procedure is needed in order to create the dataset.

The dataset has to be placed in several directories:
- *Authentics*: containing the Authentics images of the dataset
- *Tampered*: containing the Tampered images of the dataset
- *Au_Borders*: containing the border extraction for Authentic images
- *Tp_Borders*: containing the border extraction for Tampered images
- *Masks*: containing the tampering masks generated automatically using the procedure in the paper

## `CASIA_training_test_creator.py`
        
This is used to randomly generate the textual file containing the paths to the files for testing and training.

# `deep_tester.py`

Runs the experiment.