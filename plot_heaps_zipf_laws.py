import json
import pickle
import os.path
from collections import defaultdict
from matplotlib import pyplot as plt
from math import log
import seaborn as sn
sn.set()

import matplotlib.ticker as mticker

def read_data(filename):
    word2freq = defaultdict(int)

    i = 0

    counter = 0
    with open(filename, 'r', encoding='utf-8') as fin:
        print('reading the text file...')
        for i, line in enumerate(fin):
            for word in line.split():
                word2freq[word] += 1
                counter=counter+1
            if i % 100000 == 0:
                print(i)

    total_words = sum(word2freq.values())
    word2nfreq = {w: word2freq[w]/total_words for w in word2freq}
    print("all worsd: ",counter)
    return word2nfreq


def plot_zipf_law(word2nfreq):
    y = sorted(word2nfreq.values(), reverse=True)
    x = list(range(1, len(y)+1))

    product = [a * b for a, b in zip(x, y)]
    print(product[:1000])  # todo: print and note the roughly constant value

    y = [log(e, 2) for e in y]
    x = [log(e, 2) for e in x]

    plt.plot(x, y)
    plt.xlabel('log(rank)')
    plt.ylabel('log(frequency)')
    plt.title("Zipf's law")
    plt.show()


def plot_heaps_law(filename):
    word_set = set()
    total_words = 0
    unique_words = []
    total_words_counts = []

    with open(filename, 'r', encoding='utf-8') as fin:
        print('Calculating Heaps\' law values...')
        for i, line in enumerate(fin):
            words = line.split()
            total_words += len(words)
            word_set.update(words)
            unique_words.append(len(word_set))
            total_words_counts.append(total_words)

            if i % 100000 == 0:
                print(f"Processed {i} lines: {total_words} total words, {len(word_set)} unique words.")

    # Plotting Heaps' law
    plt.plot(total_words_counts, unique_words)
    plt.xlabel('Total Words')
    plt.ylabel('Unique Words')
    plt.title("Heaps' Law")
    # Format axes to show full numbers

    plt.show()

if __name__ == '__main__':
    with open('config.json', 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    if not os.path.isfile('word2nfreq.pkl'):
        data = read_data(config['corpus'])
        pickle.dump(data, open('word2nfreq.pkl', 'wb'))

    plot_zipf_law(pickle.load(open('word2nfreq.pkl', 'rb')))

    # Plot Heaps' Law
    plot_heaps_law(config['corpus'])
