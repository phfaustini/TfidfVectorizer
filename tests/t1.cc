// g++ t1.cc ../src/tfidf_vectorizer.cc -larmadillo -std=c++11 -o tests
#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#define protected public

#include "../include/tfidf_vectorizer.h"


TEST_CASE("tokenise_document: lowercase")
{
    std::vector<std::string> documents = {"This is the first document.", 
                                      "This document is the second document.", 
                                      "And this is the third one.", 
                                      "Is this the first document?",
                                      "A more-elaborated sentence my de@r, isn't it?!"};
    TfIdfVectorizer tfidfvectorizer(/*binary=*/false, /*lowercase=*/true);
    std::vector<std::string> tokens = tfidfvectorizer.tokenise_document(documents[0]);
    std::vector<std::string> answer = {"this", "is", "the", "first", "document"};
    REQUIRE(tokens == answer);

    tokens = tfidfvectorizer.tokenise_document(documents[1]);
    answer = {"this", "document", "is", "the", "second", "document"};
    REQUIRE(tokens == answer);

    tokens = tfidfvectorizer.tokenise_document(documents[2]);
    answer = {"and", "this", "is", "the", "third", "one"};
    REQUIRE(tokens == answer);

    tokens = tfidfvectorizer.tokenise_document(documents[3]);
    answer = {"is", "this", "the", "first", "document"};
    REQUIRE(tokens == answer);

    tokens = tfidfvectorizer.tokenise_document(documents[4]);
    answer = {"a", "more", "elaborated", "sentence", "my", "de", "r", "isn", "t", "it"};
    REQUIRE(tokens == answer);

}

TEST_CASE("tokenise_documents: lowercase")
{
    std::vector<std::string> documents = {"This is the first document.", 
                                      "This document is the second document.", 
                                      "And this is the third one.", 
                                      "Is this the first document?"};
    TfIdfVectorizer tfidfvectorizer(/*binary=*/false, /*lowercase=*/true);
    std::vector<std::vector<std::string>> tokens = tfidfvectorizer.tokenise_documents(documents);
    std::vector<std::vector<std::string>> answer = {{"this", "is", "the", "first", "document"},{"this", "document", "is", "the", "second", "document"},{"and", "this", "is", "the", "third", "one"},{"is", "this", "the", "first", "document"}};
    REQUIRE(tokens == answer);
}

TEST_CASE("tokenise_document: no lowercase")
{
    std::vector<std::string> documents = {"This is the first document.", 
                                      "This document is the second document.", 
                                      "And this is the third one.", 
                                      "Is this the first docUment?"};
    TfIdfVectorizer tfidfvectorizer(/*binary=*/false, /*lowercase=*/false);
    std::vector<std::vector<std::string>> tokens = tfidfvectorizer.tokenise_documents(documents);
    std::vector<std::vector<std::string>> answer = {{"This", "is", "the", "first", "document"},{"This", "document", "is", "the", "second", "document"},{"And", "this", "is", "the", "third", "one"},{"Is", "this", "the", "first", "docUment"}};
    REQUIRE(tokens == answer);
}

template<typename T>
bool isEqual(std::vector<T> a, std::vector<T> b)
{
    return (a.size()==b.size() && std::equal(a.begin(), a.end(), b.begin()));
}

TEST_CASE("word count")
{
    TfIdfVectorizer tfidfvectorizer(/*binary=*/false, /*lowercase=*/true);
    std::vector<std::vector<std::string>> tokenised = {{"this", "is", "the", "first", "document"},{"This", "there", "this", "an", "end", "tokens", "a", "document", "this"}};
    std::vector<std::map<std::string, int>> wc = tfidfvectorizer.word_count(tokenised);
    std::map<std::string, int> d1; d1["this"] = 1; d1["is"] = 1; d1["the"] = 1; d1["first"] = 1; d1["document"] = 1;
    std::map<std::string, int> d2; d2["This"] = 1; d2["this"] = 2; d2["there"] = 1; d2["an"] = 1; d2["end"] = 1; d2["tokens"] = 1; d2["a"] = 1; d2["document"] = 1;
    std::vector<std::map<std::string, int>> answer = {d1, d2};
    REQUIRE(isEqual(wc, answer));
}

