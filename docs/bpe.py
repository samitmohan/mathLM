# language model needs to be converted from text to numbers (token ids) before it can be trained or used for inference.

# originally a compression algorithm; which mreges most frequent adjacent pairs into new token

# every word is a sequence of characters

from collections import Counter, defaultdict

corpus = "low low low lowest lowest newer newer wider new"

vocab = Counter(corpus.split())

# convert each word to a character tuple (hashable)


def word_to_char(word):
    return tuple(list(word) + ["</w>"])  # marks end of word


char_vocab = {word_to_char(word): freq for word, freq in vocab.items()}
# print(char_vocab)
# {('l','o','w','</w>'): 3,
#  ('l','o','w','e','s','t','</w>'): 2,
#  ('n','e','w','e','r','</w>'): 2,
#  ('w','i','d','e','r','</w>'): 1,
#  ('n','e','w','</w>'): 1}

# count all adjacent pairs (weighted by freqency)
# Which pair of symbols occurs most frequently?


def get_adjacent_pairs(vocab):
    pairs = defaultdict(int)
    for characterTuples, freq in vocab.items():
        for i in range(len(characterTuples) - 1):
            pair = (characterTuples[i], characterTuples[i + 1])
            pairs[pair] += freq  # weight by how often this word appears

    return pairs


pairs = get_adjacent_pairs(char_vocab)
# print(sorted(pairs.items(), key=lambda x: -x[1]))  # sort by max
# [(('l','o'), 5), (('o','w'), 5), (('e','r'), 3), ...]

# merge most frequent pair: in this case ('l', 'o') = 5 means lo
# before ('l','o','w','</w>') and now ('lo','w','</w>')


def mergePairs(best_pair, vocab):
    new_vocab = {}
    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            # find the pair and u merge
            if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                new_word.append(word[i] + word[i + 1])  # merge
                i += 2  # skip both
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq

    return new_vocab


best = ("l", "o")
vocab = mergePairs(best, char_vocab)
# print(vocab)

# Train!!

num_merges = 10
merges = []  # store merge rules

for i in range(num_merges):
    pairs = get_adjacent_pairs(vocab)

    if not pairs:
        break

    # pick most frequent pair
    best_pair = max(pairs, key=pairs.get)
    merges.append(best_pair)

    print(f"Step {i + 1}: merging {best_pair} (freq={pairs[best_pair]})")

    vocab = mergePairs(best_pair, vocab)

print("\nFinal vocab:")
for k, v in vocab.items():
    print(k, ":", v)

print("\nLearned merges:")
print(merges)

# encode and decode

# apply these learned merges to text the model hasn't seen
