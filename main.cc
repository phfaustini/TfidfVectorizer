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
    X.print("TF-IDF Matrix");

    std::map<std::string, double> idfs = tfidfvectorizer.get_idf_();
    std::map<std::string, size_t> vocab = tfidfvectorizer.get_vocabulary_();

    std::cout << "Training vocabulary:" << std::endl;
    for (auto const& x : vocab)
    {
        std::cout << x.first << " ";
    }std::cout << std::endl;


    std::cout << "IDF values:" << std::endl;
    for (auto const& x : idfs)
    {
        std::cout << x.first << " = " << x.second << std::endl;;
    }

    return 0;
}