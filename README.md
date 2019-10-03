# C++ TfidfVectorizer
Convert raw documents to a matrix of TF-IDF features.

For ease of use, all necessary implementation is in .h file. Hence, you just have to 

```
#include "tfidf_vectorizer.h"
```

the file in your code. An example is given in the file main.cc.


## Requirements:
* Armadillo, g++, boost
```
sudo apt install g++ libboost-all-dev libarmadillo-dev
```

## Compiling and running example in main.cc:

```
g++ main.cc -larmadillo -std=c++11 && ./a.out
```


## Features:
* Tokenizes raw documents.
* Work with both tf-idf and binary values.
* Can use a selected number of features (the ones with highest idf).
* Similar interface to sklearn: _fit_, _transform_ and _fit\_transform_ methods, as well as _idf\__ and _vocabulary\__ members. However, this is not a port from sklearn TfidfVectorizer. However, it tries to mimic sklearn. The example given here produces the same tfidf matrix as sklearn in https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

## Notes:
* Features are in rows, documents (objects) are in columns.
* This behavior is opposed to what is normally done in Python, but it is the default in C++ libraries such as MLPack.
