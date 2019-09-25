#include "../include/tfidf_vectorizer.h"

//https://stackabuse.com/python-for-nlp-creating-tf-idf-model-from-scratch/


TfIdfVectorizer::TfIdfVectorizer(bool binary, int max_features)
{
    this->binary = binary;
    this->max_features = max_features; // -1 uses all words.
}


std::vector<std::string> TfIdfVectorizer::tokenise_document(std::string& document)
{
    std::vector<std::string> tokens;
    std::string s;

    boost::tokenizer<> tok(document);
    for(boost::tokenizer<>::iterator beg=tok.begin(); beg!=tok.end();++beg)
    {
        s = *beg;
        //boost::algorithm::to_lower(s);
        tokens.push_back(s);
    }
    return tokens;
}


std::vector<std::vector<std::string>> TfIdfVectorizer::tokenise_documents(std::vector<std::string>& documents)
{
    std::vector<std::vector<std::string>> documents_tokenised;
    for (size_t i = 0; i < documents.size(); i++)
        documents_tokenised.push_back(tokenise_document(documents[i]));
    return documents_tokenised;
}

std::vector<std::map<std::string, int>> TfIdfVectorizer::word_count(std::vector<std::vector<std::string>>& documents_tokenised)
{
    std::vector<std::map<std::string, int>> documents_word_counts;
    std::string word;
    std::set<std::string> words_set;
    for (size_t d = 0; d < documents_tokenised.size(); d++) // d: document index.
    {
        std::map<std::string, int> wc;
        documents_word_counts.push_back(wc);
        for (size_t w = 0; w < documents_tokenised[d].size(); w++) // w: word index.
        {
            word = documents_tokenised[d][w];
            documents_word_counts[d][word] += 1;
            words_set.insert(word);
        }
    }

    size_t i = 0;
    for (std::set<std::string>::iterator it = words_set.begin(); it != words_set.end(); ++it)
    {
        word = *it;
        this->vocabulary_[word] = i;
        i++;
    }
    
    return documents_word_counts;
}

void TfIdfVectorizer::fit(std::vector<std::string>& documents)
{
    this->vocabulary_.clear();
    this->idf_.clear();
    std::vector<std::vector<std::string>> documents_tokenised = tokenise_documents(documents);
    std::vector<std::map<std::string, int>> documents_word_counts = word_count(documents_tokenised);
    idf(documents_word_counts);
}

std::map<std::string, double> TfIdfVectorizer::idf(std::vector<std::map<std::string, int>>& documents_word_counts)
{
    std::string key;
    int value;
    size_t i;
    size_t documents = documents_word_counts.size();
    double d_documents = (double)documents;
    double temp_idf;
    for (auto it = this->vocabulary_.cbegin(); it != this->vocabulary_.cend(); ++it)
    {
        key = (*it).first;
        value = 0;//(*it).second;
        for (i = 0; i < documents; i++)
        {
            if (documents_word_counts[i][key] > 0)
                value++;
        }
        temp_idf = std::log(d_documents / (value + 1));
        this->idf_[key] = temp_idf;
    }

    /*Get only the words with highest idf.*/
    if (this->max_features > 0)
    {
        /*Ordered set, reversed order by value*/
        std::set<std::pair<int, std::string>> temp_idf;
        for (auto const& x : this->idf_)
        {
            std::pair<int, std::string> elem (x.second, x.first);
            temp_idf.insert(elem);
        }

        /*Get the most important words from the ordered set */
        this->idf_.clear();
        this->vocabulary_.clear();
        int i = 0;
        std::set<std::pair<int, std::string>>::reverse_iterator rit;
        for (rit = temp_idf.rbegin(); rit != temp_idf.rend(); rit++)
        {
            if (i == this->max_features)
                break; 
            this->idf_[rit->second] = rit->first;
            this->vocabulary_[rit->second] = i;
            i++;
        }
    }
    return this->idf_;
}

std::vector<std::map<std::string, double>> TfIdfVectorizer::tf(std::vector<std::vector<std::string>>& documents_tokenised)
{
    std::vector<std::map<std::string, double>> documents_word_frequency;
    std::string word;
    for (size_t d = 0; d < documents_tokenised.size(); d++) // d: document index.
    {
        std::map<std::string, double> wf;
        documents_word_frequency.push_back(wf);
        for (size_t w = 0; w < documents_tokenised[d].size(); w++) // w: word index.
        {
            word = documents_tokenised[d][w];
            documents_word_frequency[d][word] += 1;
        }
        for (auto& s : documents_word_frequency[d])
        {
            documents_word_frequency[d][s.first] /= documents_tokenised[d].size();
        } 
    }
    return documents_word_frequency;
}

arma::mat TfIdfVectorizer::fit_transform(std::vector<std::string>& documents)
{
    fit(documents);
    return transform(documents);
}

arma::mat TfIdfVectorizer::transform(std::vector<std::string>& documents)
{
    std::vector<std::vector<std::string>> documents_tokenised = tokenise_documents(documents);
    std::vector<std::map<std::string, double>> documents_word_counts = tf(documents_tokenised);
    arma::mat X_transformed(this->vocabulary_.size(), documents.size());

    std::map<std::string, double> tf_hash;
    std::string word;
    size_t w;
    double idf;
    for (size_t d = 0; d < documents.size(); d++)
    {
        tf_hash = documents_word_counts[d];
        for (auto& s : this->idf_)
        {
            word = s.first;
            w = this->vocabulary_[word];
            idf = s.second;
            if (this->binary)
                X_transformed(w, d) = (tf_hash[word] > 0) ? 1 : 0;
            else
                X_transformed(w, d) = tf_hash[word] * idf;
        } 
    }
    return X_transformed;
}
