package de.jungblut.conll;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.yaml.snakeyaml.Yaml;

public class VectorizerMain {

  // how many words back and forth are taken into account for the given word
  private static final Pattern SPLIT_PATTERN = Pattern.compile(" ");
  private static final int SEQUENCE_CONTEXT = 2;
  private static final int EMBEDDING_VECTOR_SIZE = 50;
  private static final String DEFAULT_TOKEN = "unknown";
  private static final String DATA_PATH = "data/";
  private static final String GLOVE_FILE_NAME = "glove.6B.50d.txt";
  private static final String NER_TRAIN_FILE_NAME = "eng.train.txt";
  private static final String TRAIN_OUT_FILE_NAME = "vectorized";
  private static final String META_OUT_FILE_NAME = "meta.yaml";

  private LabelManager labelManager;
  private int sequenceContextSize;
  private int embeddingVectorSize;
  private String embeddingPath;
  private String inputFilePath;
  private String outputFolder;
  private boolean binaryOutput;

  public VectorizerMain(int sequenceContextSize, //
      int embeddingVectorSize, //
      String embeddingPath, //
      String inputFilePath,//
      String outputFolder, //
      boolean binaryOutput,//
      LabelManager labelManager) {
    this.sequenceContextSize = sequenceContextSize;
    this.embeddingVectorSize = embeddingVectorSize;
    this.embeddingPath = embeddingPath;
    this.inputFilePath = inputFilePath;
    this.outputFolder = outputFolder;
    this.binaryOutput = binaryOutput;
    this.labelManager = labelManager;
  }

  public void vectorize() throws IOException {
    System.out.println("Sequence context length: " + sequenceContextSize);
    System.out.println("Embedding vector dimension: " + embeddingVectorSize);
    System.out.println("Embedding path: " + embeddingPath);
    System.out.println("Input path: " + inputFilePath);
    System.out.println("Binary output: " + binaryOutput);
    System.out.println("Output folder: " + outputFolder);

    // read the glove embeddings
    HashMap<String, double[]> embeddingMap = readGloveEmbeddings();
    System.out.println("read " + embeddingMap.size()
        + " embedding vectors. Vectorizing...");

    final int vectorsToBuffer = sequenceContextSize * 2 + 1;
    final int numFinalFeatures = vectorsToBuffer * embeddingVectorSize;

    double[] defaultVector = Objects.requireNonNull(
        embeddingMap.get(DEFAULT_TOKEN), "couldn't find embedding for token "
            + DEFAULT_TOKEN);

    try (SequenceFileWriter writer = createWriter(outputFolder, binaryOutput)) {

      Deque<double[]> vectorBuffer = new LinkedList<>();
      Deque<Integer> labelBuffer = new LinkedList<>();
      addPadding(defaultVector, vectorBuffer, sequenceContextSize);

      try (BufferedReader reader = new BufferedReader(new FileReader(
          inputFilePath))) {

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
          int labelIndex = labelManager.getOrCreate(label);

          vectorBuffer.addLast(embedding);
          labelBuffer.addLast(labelIndex);

          // if we reach the buffer size we can flush the next item in the queue
          if (vectorBuffer.size() == vectorsToBuffer) {
            writer.write(
                labelBuffer.pop(),
                computeSequenceAndDropHead(vectorBuffer, vectorsToBuffer,
                    defaultVector));
          }
        }
      }

      while (!labelBuffer.isEmpty()) {
        writer.write(
            labelBuffer.pop(),
            computeSequenceAndDropHead(vectorBuffer, vectorsToBuffer,
                defaultVector));
      }
    }

    // dump the label map with # features as YAML map.
    Map<String, Object> data = new HashMap<String, Object>();
    data.put("embedding_dim", embeddingVectorSize);
    data.put("context", vectorsToBuffer);
    data.put("nlabels", labelManager.getLabelMap().size());
    data.put("feature_dim", numFinalFeatures);
    // inverse the map so we can do int->string lookups somewhere else
    data.put("labels", labelManager.getLabelMap().inverse());
    Yaml yaml = new Yaml();
    Files.write(Paths.get(outputFolder + META_OUT_FILE_NAME), yaml.dump(data)
        .getBytes());

    System.out.println("Done.");
  }

  public static void main(String[] args) throws IOException, ParseException {

    Options options = new Options();
    options
        .addOption(
            "c",
            "contextlen",
            true,
            "how many words around it are taken into account to create the final feature vector. default 2");
    options.addOption("d", "embvecdim", true,
        "the dimensionality of the embedding vectors");
    options.addOption("b", "binary", false,
        "if supplied, outputs in binary instead of text format");
    options.addOption("i", "input", true, "the path of the dataset");
    options.addOption("o", "output", true, "the folder for the output");
    options.addOption("e", "embeddings", true, "the path of the embeddings");
    options.addOption("l", "meta", true,
        "the path of the train meta yaml to get the labels");

    if (args.length > 0 && args[0].equals("-h")) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("vectorizer", options);
      System.exit(0);
    }

    System.out.println("add -h for more options!");
    CommandLineParser parser = new DefaultParser();
    CommandLine cmd = parser.parse(options, args);

    int sequenceContextSize = Integer.parseInt(cmd.getOptionValue('c',
        SEQUENCE_CONTEXT + ""));
    int embeddingVectorSize = Integer.parseInt(cmd.getOptionValue('d',
        EMBEDDING_VECTOR_SIZE + ""));

    String embeddingPath = cmd.getOptionValue('e', DATA_PATH + GLOVE_FILE_NAME);
    String inputFilePath = cmd.getOptionValue('i', DATA_PATH
        + NER_TRAIN_FILE_NAME);
    String outputFolderPath = cmd.getOptionValue('o', DATA_PATH);
    boolean binaryOutput = cmd.hasOption('b');

    LabelManager manager = new LabelManager();
    if (cmd.hasOption('l')) {
      Yaml yaml = new Yaml();
      @SuppressWarnings("unchecked")
      Map<String, Object> map = (Map<String, Object>) yaml.load(new String(
          Files.readAllBytes(Paths.get(cmd.getOptionValue('l')))));
      @SuppressWarnings("unchecked")
      Map<Integer, String> labels = (Map<Integer, String>) map.get("labels");
      manager = new LabelManager(labels);
    }

    VectorizerMain m = new VectorizerMain(sequenceContextSize,
        embeddingVectorSize, embeddingPath, inputFilePath, outputFolderPath,
        binaryOutput, manager);
    m.vectorize();
  }

  private List<double[]> computeSequenceAndDropHead(
      Deque<double[]> vectorBuffer, int vectorsToBuffer, double[] defaultVector) {

    if (vectorBuffer.size() < vectorsToBuffer) {
      // add the padding for the last elements in the deque and flush the
      // remainder of the elements
      addPadding(defaultVector, vectorBuffer,
          vectorsToBuffer - vectorBuffer.size());
    }

    List<double[]> toReturn = new ArrayList<>();
    for (double[] vector : vectorBuffer) {
      toReturn.add(vector);
    }

    if (toReturn.size() != vectorsToBuffer) {
      throw new IllegalArgumentException(
          "didn't get enough vectors to create feature vector.");
    }

    // remove the first element in our sequence
    vectorBuffer.pop();

    return toReturn;
  }

  private static void addPadding(double[] defaultVector,
      Deque<double[]> vectorBuffer, int numVectors) {
    for (int i = 0; i < numVectors; i++) {
      vectorBuffer.addLast(defaultVector);
    }
  }

  private static HashMap<String, double[]> readGloveEmbeddings()
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

  private static SequenceFileWriter createWriter(String outputFolder,
      boolean binary) throws IOException {
    if (binary) {
      return new SequenceFileWriter.BinaryWriter(outputFolder
          + TRAIN_OUT_FILE_NAME);
    } else {
      return new SequenceFileWriter.TextWriter(outputFolder
          + TRAIN_OUT_FILE_NAME);
    }
  }
}
