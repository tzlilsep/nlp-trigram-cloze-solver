# nlp-trigram-cloze-solver

Natural Language Processing - Cloze Task Solver
This project is designed to solve a Cloze task using a trigram language model. The task involves filling in missing words in a sentence, where the correct words are selected from a list of candidates based on their surrounding context. The solution leverages a statistical language model built from a large corpus to predict the most probable words.

# Project Description
In this project, we build a trigram model using a corpus of text, then apply this model to predict missing words in a Cloze-style text. The solution is designed to handle missing words efficiently by calculating probabilities using the surrounding context of the missing words.

# Tasks Implemented:
Trigram Language Model: A trigram model is used to calculate the probability of a word given its two preceding words, using Add-1 smoothing.
Cloze Task Solver: The program fills in missing words (represented as __________) in a given text using the trigram model.
Chance Accuracy Estimation: An estimation of the accuracy when randomly selecting words is computed over multiple trials to provide a baseline for comparison.

# How It Works
N-gram Model:
Unigrams, Bigrams, and Trigrams: The model counts occurrences of unigrams, bigrams, and trigrams in the corpus.
Probability Calculation: For a given placeholder, the program computes the probability of each candidate word using the trigram model and selects the word with the highest probability.

# Functions:
build_ngram_model(corpus_path, candidates_set): This function builds the unigram, bigram, and trigram models from the provided corpus.
trigram_prob(word3, word1, word2): This calculates the probability of word3 given word1 and word2 using Add-1 smoothing.
predict_best_candidate(candidates, word_before1, word_before2, word_after1, word_after2): This function predicts the best candidate to fill in the blank based on the trigram model.
solve_cloze(input_path, candidates_path, corpus_path): This function solves the Cloze task by filling in the blanks with the predicted words.
estimate_chance_accuracy(input_path, candidates_path): This function estimates the accuracy of random word selection to be used as a baseline for comparison.
