//g++ main.cc src/tfidf_vectorizer.cc -larmadillo -std=c++11
#include "include/tfidf_vectorizer.h"

int main()
{
    std::vector<std::string> documents = {"this is the first document.", 
                                          "this document is the second document.", 
                                          "and this is the third one.", 
                                          "is this the first document?"};
    TfIdfVectorizer tfidfvectorizer;
    arma::mat X = tfidfvectorizer.fit_transform(documents);
    X.print("Matrix");
    std::cout << X.n_rows << " " << X.n_cols;

    return 0;
}