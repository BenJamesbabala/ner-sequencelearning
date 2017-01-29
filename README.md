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

Then on top of that file, you can run the keras model training, which does a five fold CV:

> python3 train.py

It takes roughly 6s per epoch with batch size of 512 samples with 970GTX GPU and tensorflow backend. 

The model
---------

The model is an LSTM over a convolutional layer which itself trains over a sequence of seven glove embedding vectors (three previous words, word for the current label, three following words). The last layer is a softmax over all output classes.

CV categorical accuracy is about 97.51%, class weighted F1 score is also about 97.49%.

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
[[42706   105    67    59    37     0]
 [  338  1637    29    29    59     0]
 [  265    34   924    17    24     0]
 [   93     7     8  3027    14     0]
 [   92    51    21    17  1913     0]
 [    1     0     3     0     0     0]]
F1 Score  0.972595415174
```

and for test set B we get:

``` 
labels: {0: O, 1: I-ORG, 2: I-MISC, 3: I-PER, 4: I-LOC, 5: B-LOC, 6: B-MISC, 7: B-ORG}
Confusion matrix:
[[38131   190   117    52    63     0     0     0]
 [  323  1960    53    36   119     0     0     0]
 [  196    39   640    15    19     0     0     0]
 [  115    29     0  2610    19     0     0     0]
 [   80    95    32    15  1697     0     0     0]
 [    1     0     0     2     3     0     0     0]
 [    3     0     5     0     1     0     0     0]
 [    1     4     0     0     0     0     0     0]]
F1 Score  0.964421474565
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
