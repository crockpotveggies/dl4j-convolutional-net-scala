Convolutional Network with Scala and DL4J
=========================================

A simple example of the [DL4J framework](http://deeplearning4j.org/) in Scala on a simple dataset of your choice.

To run the test, type `gradle run`.

#Dataset

Data needs to be placed in the root folder `cnn_dataset`. Images must can be JPG or PNG image format, single channel only (grayscale). Extra caution must be taken if you dare use RGB formats - Canova, the image loading library, has some bugs.

#Prerequisites

This class requires the latest version of Gradle.

#Debugging

You may also run this library in IntelliJ for debugging purposes if you have the Gradle and Scala plugins enabled.

#Helpful Resources

- [Andrej Karparthy's Convolutional Neural Networks Github Pages](http://cs231n.github.io/convolutional-networks/)