#ifndef TFIDF_VECTORISER_H
#define TFIDF_VECTORISER_H
#include <iostream>
#include <boost/tokenizer.hpp>
//#include <boost/algorithm/string.hpp>
#include <string>
#include <vector>
#include <armadillo>
#include <map>
#include <cmath>
#include <set>

class TfIdfVectoriser
{
    public:
        /**
         * Constructor.
         * 
         * @param binary: whether features are 1/0 (true) or tfidf values.
         * @param max_features: use max_features words with the highest tfidf values.
         *                      If negative, uses all words.
         */
        TfIdfVectoriser(bool binary=false, int max_features=-1);

        /**
         * Fit the model by computing idf of training data.
         * 
         * @param documents: a list of strings. Each string is a document (raw text).
         */
        void fit(std::vector<std::string>& documents);

        /**
         * Convert raw documents to a binary/tfidf representation.
         * 
         * @param documents: a list of strings. Each string is a document (raw text).
         * 
         * @return matrix with numerical features. 
         *         Each row is a feature. 
         *         Each column is a document.
         */
        arma::mat transform(std::vector<std::string>& documents);

        /**
         * Fit, followed by transform over the same argument.
         * 
         * @param documents: a list of strings. Each string is a document (raw text).
         * 
         * @return matrix with numerical features. 
         *         Each row is a feature. 
         *         Each column is a document.
         */
        arma::mat fit_transform(std::vector<std::string>& documents);

    protected:
        std::vector<std::string> tokenise_document(std::string& document);
        std::vector<std::vector<std::string>> tokenise_documents(std::vector<std::string>& documents);
        std::vector<std::map<std::string, int>> word_count(std::vector<std::vector<std::string>>& documents_tokenised);
        std::vector<std::map<std::string, double>> tf(std::vector<std::vector<std::string>>& documents_tokenised);
        std::map<std::string, double> idf(std::vector<std::map<std::string, int>>& documents_word_counts);

    private:
        std::map<std::string, double> idf_;
        std::map<std::string, size_t> vocabulary_;
        bool binary;
        int max_features;
};

#endif