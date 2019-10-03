#ifndef TFIDF_VECTORISER_H
#define TFIDF_VECTORISER_H

/*
Copyright <2019> <Pedro Faustini>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <iostream>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <string>
#include <vector>
#include <armadillo>
#include <map>
#include <cmath>
#include <set>

class TfIdfVectorizer
{
    public:
        /**
         * Constructor.
         * 
         * @param binary: If True, all non-zero term counts are set to 1. 
         *                This does not mean outputs will have only 0/1 values, 
         *                only that the tf term in tf-idf is binary. 
         *                (Set binary to true and use_idf to false to get 0/1 outputs.)
         * @param use_idf: Enable inverse-document-frequency reweighting.
         * @param lowercase: Convert all characters to lowercase before tokenizing.
         * @param max_features: use max_features words with the highest tfidf values.
         *                      If negative, uses all words.
         * @param norm: Each output will have unit norm. 
         *              None: no normalization.
         *              ‘l2’: Sum of squares of vector elements is 1.
         *              ‘l1’: Sum of absolute values of vector elements is 1. 
         * @param sublinear_tf: Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
         */
        TfIdfVectorizer(bool binary=false, bool lowercase=true, bool use_idf=true, int max_features=-1, std::string norm="l2", bool sublinear_tf=false);

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

        std::map<std::string, double> get_idf_();
        std::map<std::string, size_t> get_vocabulary_();

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
        double p;
        bool lowercase;
        bool use_idf;
        bool sublinear_tf;
};

#endif
