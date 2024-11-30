import json
from collections import defaultdict
import random

# Initialize counters as defaultdict to handle missing keys
unigram_counts = defaultdict(int)
bigram_counts = defaultdict(lambda: defaultdict(int))  # Dictionary of dictionaries for bigrams
trigram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # Dictionary of dictionaries for trigrams

def trigram_prob(word3, word1, word2):
    """
    Calculate the probability of word3 given word1 and word2 using Add-1 smoothing.
    Formula: P(word3 | word1, word2) = (count(word1, word2, word3) + 1) / (count(word1, word2) + V)
    """
    count_trigram = trigram_counts[word1][word2][word3]
    count_bigram = bigram_counts[word1][word2]
    vocab_size = len(unigram_counts)  # Total number of unique words in the vocabulary

    # Apply Add-1 smoothing
    return (count_trigram + 1) / (count_bigram + vocab_size)


def build_ngram_model(corpus_path, candidates_set):
    """
    Build unigram, bigram, and trigram models from the given corpus file.
    Only processes words that are in the set of `candidates`.
    """
    print(f"Loading and processing lines from corpus...")

    with open(corpus_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):

            if i % 100000 == 0:
                print(i)

            # Split the line into words
            s = ["<s>"] + line.split() + ["</s>"]

            # Use zip to process unigrams, bigrams, and trigrams in one loop
            for w1, w2, w3 in zip(s, s[1:], s[2:]):
                # Process unigram (current word)
                if w1 in candidates_set or w2 in candidates_set or w3 in candidates_set:
                    unigram_counts[w1] += 1
                    bigram_counts[w1][w2] += 1
                    trigram_counts[w1][w2][w3] += 1


    print("N-gram models built successfully!")

def predict_best_candidate(candidates, word_before1, word_before2, word_after1, word_after2):
    """
    Predict the best candidate word to fill in the blank based on the trigram model.
    Uses the product of:
    P(candidate | word_before1, word_before2) * P(word_after1 | word_before2, candidate) * P(word_after2 | candidate, word_after1)
    """
    candidates_scores = {}

    for candidate in candidates:
        # Calculate trigram probabilities
        p1 = trigram_prob(candidate, word_before1, word_before2)  # P(candidate | word_before1, word_before2)
        p2 = trigram_prob(word_after1, word_before2, candidate)  # P(word_after1 | word_before2, candidate)
        p3 = trigram_prob(word_after2, candidate, word_after1)  # P(word_after2 | candidate, word_after1)

        # Combine the probabilities using a product
        candidates_scores[candidate] = (p1 * p2 * p3)


    # return the best candidate based on the highest score
    return  max(candidates_scores, key=candidates_scores.get)

def solve_cloze(input_path, candidates_path, corpus_path):
    """
    Solve the Cloze task by filling in the blanks in the input text.
    Uses a trigram language model to predict the most probable word for each blank.
    """
    # Load input cloze text
    with open(input_path, 'r', encoding='utf-8') as f:
        cloze_text = f.read()
    print(f"Loaded cloze text: {cloze_text[:200]}...")  # Debug: Show the first 200 characters

    # Load candidate words
    with open(candidates_path, 'r', encoding='utf-8') as f:
        candidates = f.read().splitlines()
    candidates_copy = candidates
    print(f"Loaded candidates: {candidates}")

    # Build the trigram model
    build_ngram_model(corpus_path, set(candidates))

    # Solve the cloze
    solution = []
    placeholders = cloze_text.split('__________')
    print(f"Found {len(placeholders) - 1} placeholders in the cloze text.")

    for i in range(len(placeholders) - 1):
        # Get context before the blank (two words before)
        context_before = placeholders[i].strip().split()
        word_before1 = context_before[-2] if len(context_before) > 1 else "<s>"
        word_before2 = context_before[-1] if len(context_before) > 0 else "<s>"

        # Get context after the blank (two words after)
        context_after = placeholders[i + 1].strip().split()
        word_after1 = context_after[0] if len(context_after) > 0 else "</s>"
        word_after2 = context_after[1] if len(context_after) > 1 else "</s>"

        # print(f" {word_before1} {word_before2} __________ {word_after1} {word_after2} ")

        # Predict the best candidate using the trigram model
        predicted_word = predict_best_candidate(candidates, word_before1, word_before2, word_after1, word_after2)
        if predicted_word != "UNKNOWN":
            candidates.remove(predicted_word)  # Remove used candidate
        solution.append(predicted_word)

    # Compare solution with correct answer assuming that the order of the candidates is the correct order:
    # compare_results(solution, candidates_copy)

    return solution


def compare_results(solution, correct_answer):
    """
    Compare the given solution to the correct answer and calculate accuracy.
    Displays the number of correctly completed blanks and the accuracy percentage.
    """
    correct_count = sum(1 for pred, actual in zip(solution, correct_answer) if pred == actual)
    total = len(correct_answer)
    accuracy = correct_count / total
    print(f"Correctly completed: {correct_count}/{total} words, Accuracy: {accuracy * 100:.2f}%")

def estimate_chance_accuracy(input_path, candidates_path):
    """
    Estimate the chance accuracy by randomly selecting words for the blanks and
    calculating the mean accuracy over 100 trials. The result is expressed as the
    average number of correctly filled blanks divided by the total number of blanks.
    """
    # Load input cloze text
    with open(input_path, 'r', encoding='utf-8') as f:
        cloze_text = f.read()

    # Load candidate words
    with open(candidates_path, 'r', encoding='utf-8') as f:
        candidates = f.read().splitlines()

    # Load correct answers (assuming they are stored in a separate file or provided in candidates)
    correct_answers = candidates[:]

    # Determine the number of placeholders in the cloze text
    placeholders = cloze_text.split('__________')
    num_placeholders = len(placeholders) - 1

    # Perform random selection over 100 trials
    num_trials = 100
    total_correct_per_trial = []

    for _ in range(num_trials):
        random_solution = [random.choice(candidates) for _ in range(num_placeholders)]
        correct_count = sum(1 for pred, actual in zip(random_solution, correct_answers) if pred == actual)
        total_correct_per_trial.append(correct_count)

    # Calculate mean accuracy
    chance_accuracy = sum(total_correct_per_trial) / (num_trials * num_placeholders)

    # Print details
    print(f"Estimated chance accuracy over {num_trials} random trials: {chance_accuracy * 100:.2f}%")
    print(f"Average Correctly completed: {sum(total_correct_per_trial) / num_trials:.2f} out of {num_placeholders}")

    return chance_accuracy

if __name__ == '__main__':
    with open('config.json', 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    solution = solve_cloze(config['input_filename'],
                           config['candidates_filename'],
                           config['corpus'])


    # estimate_chance_accuracy(config['input_filename'],config['candidates_filename'])

    print('cloze solution:', solution)

