This is a repo about deep sequence learning in the named entity recognition domain using Keras and Glove. 

Run it!
-------

This task works on the Conll2003 dataset found and described here:
http://www.cnts.ua.ac.be/conll2003/ner/

Get glove embeddings from:
http://nlp.stanford.edu/data/glove.6B.zip

For this training we're using only the 50 dimensional embeddings, copy them into the data directory.
Now transform the conll dataset into a sequence vector representation using the Java program in the repo:

> java -jar vectorizer.jar

You can add -h as an argument for some more options with regards to I/O and context sizes.

Then on top of that file, you can run the keras model training, which does a n-fold CV:

> python3 train.py

It takes roughly 8s per epoch with batch size of 512 samples with 970GTX GPU and tensorflow backend. 

The model
---------

The model is an LSTM over a convolutional layer which itself trains over a sequence of seven glove embedding vectors (three previous words, word for the current label, three following words). The last layer is a softmax over all output classes.

CV categorical accuracy and weighted F1 is about 98.2%.
To assess the test set performance we are ensembling the model outputs from each CV fold and average over the predictions.

Test set performance
--------------------

Conll2003 ships with two test files (A and B). You can generate the vectorized forms using this bash:

> ./vectorize_all_data.sh

and then run the f1 prediction and evaluation with:

> python3 predict_test.py

For test set A we get:

``` 
labels: {0: O, 1: I-ORG, 2: I-MISC, 3: I-PER, 4: I-LOC, 5: B-LOC, 6: B-MISC, 7: B-ORG}
Confusion matrix:
[[42755    99    38    49    33     0]
 [  302  1684    22    33    51     0]
 [  247    30   948    15    23     1]
 [   71     4     6  3056    12     0]
 [   85    45    14    18  1932     0]
 [    0     0     4     0     0     0]]
F1 Score  0.97597281064
```

and for test set B we get:

``` 
labels: {0: O, 1: I-ORG, 2: I-MISC, 3: I-PER, 4: I-LOC, 5: B-LOC, 6: B-MISC, 7: B-ORG}
Confusion matrix:
[[38175   167    98    48    65     0     0     0]
 [  304  1987    46    33   121     0     0     0]
 [  194    38   642    18    17     0     0     0]
 [   89    26     0  2639    19     0     0     0]
 [   72    78    25    16  1728     0     0     0]
 [    1     1     0     2     2     0     0     0]
 [    3     0     5     0     1     0     0     0]
 [    1     4     0     0     0     0     0     0]]
F1 Score  0.967240147344
```


Vectorizer Compilation
----------------------

The vectorizer itself can be compiled with maven using:

> mvn clean install package

this generates a fatjar (vectorizer-0.0.1-jar-with-dependencies.jar) to execute the vectorizer as above and a jar with just the code itself.

License
-------

Since I am Apache committer, I consider everything inside of this repository 
licensed by Apache 2.0 license, although I haven't put the usual header into the source files.

If something is not licensed via Apache 2.0, there is a reference or an additional licence header included in the specific source file.
