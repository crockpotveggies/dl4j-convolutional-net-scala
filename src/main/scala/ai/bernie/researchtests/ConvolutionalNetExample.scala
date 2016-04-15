package ai.bernie.researchtests

import java.util.{Collections, Random}

import org.canova.api.split.FileSplit
import org.canova.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.weights.HistogramIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable


/**
  * This object is an example to demonstrate a basic classifier of a dataset in the
  * Deeplearning4j framework. This test trains a convolutional neural network on
  * a dataset of your choice. You may adjust the layers as much as you want.
  *
  */

object ConvolutionalNetExample {
    lazy val log: Logger = LoggerFactory.getLogger(ConvolutionalNetExample.getClass)

    def main(args: Array[String]) = {
        val imageWidth = 100
        val imageHeight = 100
        val nChannels = 1
        val outputNum = 8 // number of labels
        val numSamples = 9383 // LFWLoader.NUM_IMAGES

        val batchSize = 10 // how many images processed at once
        val iterations = 5
        val splitTrainNum = (batchSize*.8).toInt
        val seed = 123
        val listenerFreq = iterations/5
        val testInputBuilder = mutable.ArrayBuilder.make[INDArray]
        val testLabelsBuilder = mutable.ArrayBuilder.make[INDArray]


        log.info("Load data.....")

        // below we load our dataset using Canova's ImageRecord Reader
        // (Canova is a data library written by the DL4J team)
        // each class of images can be placed in their own indexed directory, and the name of
        // the directory will be appended to our labels list
        val labels = new java.util.ArrayList[String]()
        val recordReader = new ImageRecordReader(imageWidth, imageHeight, nChannels, true, labels)
        val file = new java.io.File("./cnn_dataset")
        recordReader.initialize(new FileSplit(file))

        println(s"Labels size: ${labels.size()}") // note, ImageRecordReader automatically detects the parent directories of classes and adds them to labels list

        // use carefully...
        //val dataSetIterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, imageWidth*imageHeight*nChannels, labels.size())
        val dataSetIterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, -1, labels.size())

        log.info("Build model....")
        val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
          .seed(seed)
          .iterations(iterations)
          .activation("relu")
          .weightInit(WeightInit.XAVIER)
          .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
          .learningRate(0.01)
          .momentum(0.9)
          .regularization(true)
          .updater(Updater.ADAGRAD)
          .useDropConnect(true)
          .list(9)
          .layer(0, new ConvolutionLayer.Builder(3, 3)
              .padding(1,1)
              .name("cnn1")
              .nIn(nChannels)
              .stride(1, 1)
              .nOut(20)
              .weightInit(WeightInit.XAVIER)
              .activation("relu")
              .build())
          .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
              .name("pool1")
              .build())
          .layer(2, new ConvolutionLayer.Builder(3, 3)
              .name("cnn2")
              .padding(1,1)
              .stride(1,1)
              .nOut(40)
              .weightInit(WeightInit.XAVIER)
              .activation("relu")
              .build())
          .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
              .name("pool2")
              .build())
          .layer(4, new ConvolutionLayer.Builder(3, 3)
              .name("cnn3")
              .stride(1,1)
              .nOut(60)
              .weightInit(WeightInit.XAVIER)
              .activation("relu")
              .build())
          .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
              .name("pool3")
              .build())
          .layer(6, new ConvolutionLayer.Builder(2, 2)
              .name("cnn4")
              .stride(1,1)
              .nOut(80)
              .weightInit(WeightInit.XAVIER)
              .activation("relu")
              .build())
          .layer(7, new DenseLayer.Builder()
              .weightInit(WeightInit.XAVIER)
              .name("ffn1")
              .nOut(160)
              .dropOut(0.5)
              .weightInit(WeightInit.XAVIER)
              .activation("relu")
              .build())
          .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
              .nOut(outputNum)
              .weightInit(WeightInit.XAVIER)
              .activation("softmax")
              .build())
          .backprop(true).pretrain(false)

        new ConvolutionLayerSetup(builder, imageWidth, imageHeight, nChannels)

        val model = new MultiLayerNetwork(builder.build())
        model.init()

        log.info("Train model....")
        // this listener fires with each minibatch training iteration
        model.setListeners(Collections.singletonList(new ScoreIterationListener(listenerFreq).asInstanceOf[IterationListener]))
        // the histogram listener will start up a server and display data about the net
        model.setListeners(new HistogramIterationListener(1))


        while(dataSetIterator.hasNext) {
            val next: DataSet = dataSetIterator.next()
            next.scale()
            val trainTest = next.splitTestAndTrain(splitTrainNum, new Random(seed))  // train set that is the result
            val trainInput = trainTest.getTrain  // get feature matrix and labels for training
            testInputBuilder += trainTest.getTest.getFeatureMatrix
            testLabelsBuilder += trainTest.getTest.getLabels
            model.fit(trainInput)
        }

        val testInput = testInputBuilder.result
        val testLabels = testLabelsBuilder.result

        log.info("Evaluate model....")
        val eval = new Evaluation(labels)
        testInput.zip(testLabels).foreach { case (input, label) =>
          val output: INDArray = model.output(input)
          eval.eval(label, output)
        }
        val output: INDArray = model.output(testInput.head)
        eval.eval(testLabels.head, output)
        log.info(eval.stats())
        log.info("****************Example finished********************")
    }

}
