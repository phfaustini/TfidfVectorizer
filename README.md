# TfidfVectorizer
Convert raw documents to a matrix of TF-IDF features.


## Requirements:
* Armadillo, g++, boost
```
sudo apt install g++ libboost-all-dev libarmadillo-dev
```

## Compiling and running example in main.cc:

```
g++ main.cc src/tfidf_vectorizer.cc -larmadillo -std=c++11 && ./a.out
```


## Features:
* Tokenizes raw documents.
* Work with both tf-idf and binary values
* Can use a selected number of features (the ones with highest idf).

## Notes:
* Features are in rows, documents (objects) are in columns.
* This behavior is opposed to what is normally done in Python, but it is the default in C++ libraries such as MLPack.