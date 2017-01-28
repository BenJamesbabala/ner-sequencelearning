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

Then on top of that file, you can run the keras model training, which does a five fold CV:

> python3 train.py

The model
---------

The model is an LSTM over a convolutional layer which itself trains over a sequence of five glove embedding vectors (two previous words, word for the current label, two following words). The last layer is a softmax over all output classes.

CV categorical accuracy is about 96.4%, class weighted F1 score is about 96.2%.

License
-------

Since I am Apache committer, I consider everything inside of this repository 
licensed by Apache 2.0 license, although I haven't put the usual header into the source files.

If something is not licensed via Apache 2.0, there is a reference or an additional licence header included in the specific source file.
