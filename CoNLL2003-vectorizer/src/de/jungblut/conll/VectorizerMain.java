package de.jungblut.conll;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.regex.Pattern;

public class VectorizerMain {

  // how many words back and forth are taken into account for the given word
  private static final Pattern SPLIT_PATTERN = Pattern.compile(" ");
  private static final int SEQUENCE_CONTEXT = 2;
  private static final int EMBEDDING_VECTOR_SIZE = 50;
  private static final String DEFAULT_TOKEN = "unknown";
  private static final String DATA_PATH = "../data/";
  private static final String GLOVE_FILE_NAME = "glove.6B.50d.txt";
  private static final String NER_TRAIN_FILE_NAME = "eng.train.txt";
  private static final String TRAIN_OUT_FILE_NAME = "vectorized.txt";
  private static final String META_OUT_FILE_NAME = "meta.yaml";
  private static final String OUT_LABEL = "O";

  public static void main(String[] args) throws IOException {
    // read the glove embeddings
    HashMap<String, Integer> labelMap = new HashMap<>();
    HashMap<String, double[]> embeddingMap = readGloveEmbeddings();

    final int vectorsToBuffer = SEQUENCE_CONTEXT * 2 + 1;
    final int numFinalFeatures = vectorsToBuffer * EMBEDDING_VECTOR_SIZE;
    int nextLabelIndex = 0;
    labelMap.put(OUT_LABEL, nextLabelIndex++);
    double[] defaultVector = Objects.requireNonNull(
        embeddingMap.get(DEFAULT_TOKEN), "couldn't find embedding for token "
            + DEFAULT_TOKEN);

    try (BufferedWriter writer = new BufferedWriter(new FileWriter(DATA_PATH
        + TRAIN_OUT_FILE_NAME))) {

      Deque<double[]> vectorBuffer = new LinkedList<>();
      Deque<Integer> labelBuffer = new LinkedList<>();
      addPadding(defaultVector, vectorBuffer, SEQUENCE_CONTEXT);

      try (BufferedReader reader = new BufferedReader(new FileReader(DATA_PATH
          + NER_TRAIN_FILE_NAME))) {

        String line;
        while ((line = reader.readLine()) != null) {
          if (line.isEmpty()) {
            continue;
          }
          // format is as follows: "German JJ I-NP I-MISC"
          String[] split = SPLIT_PATTERN.split(line);
          String tkn = split[0].toLowerCase().trim();
          double[] embedding = embeddingMap.getOrDefault(tkn, defaultVector);

          String label = split[3].trim();
          Integer labelIndex = labelMap.get(label);
          if (labelIndex == null) {
            labelIndex = nextLabelIndex++;
            labelMap.put(label, labelIndex);
          }

          vectorBuffer.addLast(embedding);
          labelBuffer.addLast(labelIndex);

          // if we reach the buffer size we can flush the next item in the queue
          if (vectorBuffer.size() == vectorsToBuffer) {
            writer.write(computeOutputLine(vectorBuffer, labelBuffer,
                vectorsToBuffer, defaultVector));
            writer.newLine();
          }
        }
      }

      while (!labelBuffer.isEmpty()) {
        writer.write(computeOutputLine(vectorBuffer, labelBuffer,
            vectorsToBuffer, defaultVector));
        writer.newLine();
      }
    }

    // dump the label map with # features as YAML map.
    List<String> lines = new ArrayList<String>(Arrays.asList(new String[] {
        "features: " + numFinalFeatures, //
        "labels:"//
    }));
    appendLabels(labelMap, lines, "\t");
    Files.write(Paths.get(DATA_PATH + META_OUT_FILE_NAME), lines);
  }

  private static void appendLabels(HashMap<String, Integer> labelMap,
      List<String> lines, String identation) {
    for (Entry<String, Integer> entry : labelMap.entrySet()) {
      lines.add(identation + entry.getKey() + ": " + entry.getValue());
    }
  }

  private static String computeOutputLine(Deque<double[]> vectorBuffer,
      Deque<Integer> labelBuffer, int vectorsToBuffer, double[] defaultVector) {

    StringBuilder sb = new StringBuilder();
    Integer nextLabel = labelBuffer.pop();
    sb.append(nextLabel);

    if (vectorBuffer.size() < vectorsToBuffer) {
      // add the padding for the last elements in the deque and flush the
      // remainder of the elements
      addPadding(defaultVector, vectorBuffer,
          vectorsToBuffer - vectorBuffer.size());
    }

    int numVectorsUsed = 0;
    for (double[] vector : vectorBuffer) {
      for (double d : vector) {
        sb.append(" ");
        sb.append(d);
      }
      numVectorsUsed++;
    }

    if (numVectorsUsed != vectorsToBuffer) {
      throw new IllegalArgumentException(
          "didn't get enough vectors to create feature vector.");
    }

    // remove the first element in our sequence stride
    vectorBuffer.pop();
    return sb.toString();
  }

  private static void addPadding(double[] defaultVector,
      Deque<double[]> vectorBuffer, int numVectors) {
    for (int i = 0; i < numVectors; i++) {
      vectorBuffer.addLast(defaultVector);
    }
  }

  public static HashMap<String, double[]> readGloveEmbeddings()
      throws IOException {
    HashMap<String, double[]> map = new HashMap<>();

    try (BufferedReader reader = new BufferedReader(new FileReader(DATA_PATH
        + GLOVE_FILE_NAME))) {
      String line;
      while ((line = reader.readLine()) != null) {
        String[] split = SPLIT_PATTERN.split(line);
        if (split.length != EMBEDDING_VECTOR_SIZE + 1) {
          throw new IllegalArgumentException(
              "invalid embeddings used, encountered unexpected number of columns! "
                  + split.length);
        }
        double[] vector = new double[split.length - 1];
        for (int i = 0; i < split.length - 1; i++) {
          vector[i] = Double.parseDouble(split[i + 1]);
        }
        map.put(split[0], vector);
      }
    }
    return map;
  }
}
