#!/bin/bash

java -jar vectorizer.jar -c 3
java -jar vectorizer.jar -o data_test_a/ -c 3 -i data/eng.testa.txt -l data/meta.yaml
java -jar vectorizer.jar -o data_test_b/ -c 3 -i data/eng.testb.txt -l data/meta.yaml

