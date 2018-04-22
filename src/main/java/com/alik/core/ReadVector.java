package com.alik.core;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;

import java.util.Iterator;

public class ReadVector {
    public static final String WORD_VECTORS_PATH = "src/main/resources/trainedVector.bin";
    public static void main (String args[]){
        System.out.println("Loading word vectors and creating DataSetIterators");
        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(WORD_VECTORS_PATH);
        System.out.println(wordVectors.vocab().words());



//        for (Iterator<String> i = wordVectors.vocab().words().iterator(); i.hasNext();) {
//            String item = i.next();
//            System.out.println(item);
//        }



    }

}