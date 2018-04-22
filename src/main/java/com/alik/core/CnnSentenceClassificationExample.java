package com.alik.core;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Convolutional Neural Networks for Sentence Classification - https://arxiv.org/abs/1408.5882
 *
 * Specifically, this is the 'static' model from there
 *
 * @author Alex Black
 */
public class CnnSentenceClassificationExample {

    /** Data URL for downloading */
    /** Location to save and extract the training/testing data */
    public static final String DATA_PATH = "src/main/resources/";
    /** Location (local file system) for the Google News vectors. Set this manually. */
    public static final String WORD_VECTORS_PATH = "src/main/resources/trainedVector_izin.bin";
    static final String OCR_RESULT_DIR = "src/main/resources/ocr-result/";

    public static void main(String[] args) throws Exception {


        //Basic configuration
        int batchSize = 32;
        int vectorSize = 300;               //Size of the word vectors. 300 in the Google News model
        int nEpochs = 1;                    //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 256;  //Truncate reviews with length (# words) greater than this

        int cnnLayerFeatureMaps = 100;      //Number of feature maps / channels / depth for each CNN layer
        PoolingType globalPoolingType = PoolingType.MAX;
        Random rng = new Random(12345); //For shuffling repeatability

        //Set up the network configuration. Note that we have multiple convolution layers, each wih filter
        //widths of 3, 4 and 5 as per Kim (2014) paper.

        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .updater(new Adam(0.01))
                .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
                .l2(0.0001)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn3", new ConvolutionLayer.Builder()
                        .kernelSize(3,vectorSize)
                        .stride(1,vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn4", new ConvolutionLayer.Builder()
                        .kernelSize(4,vectorSize)
                        .stride(1,vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn5", new ConvolutionLayer.Builder()
                        .kernelSize(5,vectorSize)
                        .stride(1,vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")      //Perform depth concatenation
                .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                        .poolingType(globalPoolingType)
                        .dropOut(0.5)
                        .build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3*cnnLayerFeatureMaps)
                        .nOut(3)    //2 classes: positive or negative
                        .build(), "globalPool")
                .setOutputs("out")
                .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();

        System.out.println("Number of parameters by layer:");
        for(Layer l : net.getLayers() ){
            System.out.println("\t" + l.conf().getLayer().getLayerName() + "\t" + l.numParams());
        }

        //Load word vectors and get the DataSetIterators for training and testing
        System.out.println("Loading word vectors and creating DataSetIterators");
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
        DataSetIterator trainIter = getDataSetIterator(true, wordVectors, batchSize, truncateReviewsToLength, rng);
        DataSetIterator testIter = getDataSetIterator(false, wordVectors, batchSize, truncateReviewsToLength, rng);

        System.out.println(wordVectors.vocab().words());

        System.out.println("Starting training");
        net.setListeners(new ScoreIterationListener(100));
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainIter);
            System.out.println("Epoch " + i + " complete. Starting evaluation:");

            //Run evaluation. This is on 25k reviews, so can take some time
            Evaluation evaluation = net.evaluate(testIter);

            System.out.println(evaluation.stats());
        }

        // After training: load a single sentence and generate a prediction
        File[] listOfFiles = new File(OCR_RESULT_DIR).listFiles();

        System.out.println("\nLaporan Hasil");
        System.out.println("Metode Training : Deep Learning CNN dengan Neural Network Deeplearning4j");
        System.out.println("Metode Retrival Text : Ontology Vector Word2Vec");
        System.out.println("--------------------------------------------------------------------------");

        if (new File(OCR_RESULT_DIR).exists()) {

            for (int i = 0; i < listOfFiles.length; i++) {
                if (listOfFiles[i].isFile()) {

//                    System.out.println(listOfFiles[i].getName());
                    String contentsFirstNegative = null;
                    try {
                        contentsFirstNegative = FileUtils.readFileToString(new File(listOfFiles[i].getPath()));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    INDArray featuresFirstNegative = ((CnnSentenceDataSetIterator) testIter)
                            .loadSingleSentence(contentsFirstNegative);

                    INDArray predictionsFirstNegative = net.outputSingle(featuresFirstNegative);
                    List<String> labels = testIter.getLabels();

                    System.out.println("\nPrediksi jenis dokument dari file : " + listOfFiles[i].getName());
                    for (int z = 0; z < labels.size(); z++) {
                        System.out.println("P(" + labels.get(z) + ") = " + predictionsFirstNegative.getDouble(z));
                    }
                }
            }
        }

        //After training: load a single sentence and generate a prediction
//        String pathFirstNegativeFile = FilenameUtils.concat(DATA_PATH, "aclImdb/test/neg/0_2.txt");
//        String contentsFirstNegative = FileUtils.readFileToString(new File(pathFirstNegativeFile));
//        INDArray featuresFirstNegative = ((CnnSentenceDataSetIterator)testIter).loadSingleSentence(contentsFirstNegative);
//
//        INDArray predictionsFirstNegative = net.outputSingle(featuresFirstNegative);
//        List<String> labels = testIter.getLabels();
//
//        System.out.println("\n\nPredictions for first negative review:");
//        for( int i=0; i<labels.size(); i++ ){
//            System.out.println("P(" + labels.get(i) + ") = " + predictionsFirstNegative.getDouble(i));
//        }
    }


    private static DataSetIterator getDataSetIterator(boolean isTraining, WordVectors wordVectors, int minibatchSize,
                                                      int maxSentenceLength, Random rng ){
        String path = FilenameUtils.concat(DATA_PATH, (isTraining ? "training-surat/" : "testing-surat/"));
        String sandwichBaseDir = FilenameUtils.concat(path, "sandwich");
        String suratIzinBaseDir = FilenameUtils.concat(path, "surat-izin");
        String suratKuasaBaseDir = FilenameUtils.concat(path, "surat-kuasa");

        File fileSandwich = new File(sandwichBaseDir);
        File fileSuratIzin = new File(suratIzinBaseDir);
        File fileSuratKuasa = new File(suratKuasaBaseDir);

        Map<String,List<File>> reviewFilesMap = new HashMap<String,List<File>>();
        reviewFilesMap.put("Sandwich", Arrays.asList(fileSandwich.listFiles()));
        reviewFilesMap.put("Surat Izin", Arrays.asList(fileSuratIzin.listFiles()));
        reviewFilesMap.put("Surat Kuasa", Arrays.asList(fileSuratKuasa.listFiles()));


        LabeledSentenceProvider sentenceProvider = new FileLabeledSentenceProvider(reviewFilesMap, rng);

        return new CnnSentenceDataSetIterator.Builder()
                .sentenceProvider(sentenceProvider)
                .wordVectors(wordVectors)
                .minibatchSize(minibatchSize)
                .maxSentenceLength(maxSentenceLength)
                .useNormalizedWordVectors(false)
                .build();
    }
}
