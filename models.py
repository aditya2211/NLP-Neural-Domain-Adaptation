# models.py

from nerdata import *
from utils import *

import numpy as np
import time, pickle
from scipy import misc
import copy, json


# Scoring function for sequence models based on conditional probabilities.
# Scores are provided for three potentials in the model: initial scores (applied to the first tag),
# emissions, and transitions. Note that CRFs typically don't use potentials of the first type.
class ProbabilisticSequenceScorer(object):
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence, tag_idx):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence, prev_tag_idx, curr_tag_idx):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    '''
    def score_emission(self, sentence, tag_idx, word):
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.get_index("UNK")
        return self.emission_log_probs[tag_idx, word_idx]
    '''
# utils.py


# Bijection between objects and integers starting at 0. Useful for mapping
# labels, features, etc. into coordinates of a vector space.
class Indexer(object):
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in xrange(0, len(self))])

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        return self.index_of(object) != -1

    # Returns -1 if the object isn't present, index otherwise
    def index_of(self, object):
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    # Adds the object to the index if it isn't present, always returns a nonnegative index
    def get_index(self, object, add=True):
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]


# Map from objects to doubles that has a default value of 0 for all elements
# Relatively inefficient (dictionary-backed); shouldn't be used for anything very large-scale,
# instead use an Indexer over the objects and use a numpy array to store the values
class Counter(object):
    def __init__(self):
        self.counter = {}

    def __repr__(self):
        return str([str(key) + ": " + str(self.get_count(key)) for key in self.counter.keys()])

    def __len__(self):
        return len(self.counter)

    def keys(self):
        return self.counter.keys()

    def get_count(self, key):
        if self.counter.has_key(key):
            return self.counter[key]
        else:
            return 0

    def increment_count(self, obj, count):
        if self.counter.has_key(obj):
            self.counter[obj] = self.counter[obj] + count
        else:
            self.counter[obj] = count

    def increment_all(self, objs_list, count):
        for obj in objs_list:
            self.increment_count(obj, count)

    def set_count(self, obj, count):
        self.counter[obj] = count

    def add(self, otherCounter):
        for key in otherCounter.counter.keys():
            self.increment_count(key, otherCounter.counter[key])

    # Bad O(n) implementation right now
    def argmax(self):
        best_key = None
        for key in self.counter.keys():
            if best_key is None or self.get_count(key) > self.get_count(best_key):
                best_key = key
        return best_key


# Beam data structure. Maintains a list of scored elements like a Counter, but only keeps the top n
# elements after every insertion operation. Insertion can sometimes be slow (list is maintained in
# sorted order), access is O(1)
class Beam(object):
    def __init__(self, size):
        self.size = size
        self.elts = []
        self.scores = []

    def __repr__(self):
        return "Beam(" + repr(self.get_elts_and_scores()) + ")"

    def __len__(self):
        return len(self.elts)

    # Adds the element to the beam with the given score if the beam has room or if the score
    # is better than the score of the worst element currently on the beam
    def add(self, elt, score):
        if len(self.elts) == self.size and score < self.scores[-1]:
            # Do nothing because this element is the worst
            return
        # If the list is empty, just insert the item
        if len(self.elts) == 0:
            self.elts.insert(0, elt)
            self.scores.insert(0, score)
        # Otherwise, find the insertion point with binary search
        else:
            lb = 0
            ub = len(self.scores) - 1
            # We're searching for the index of the first element with score less than score
            while lb < ub:
                m = (lb + ub) // 2
                # Check > because the list is sorted in descending order
                if self.scores[m] > score:
                    # Put the lower bound ahead of m because all elements before this are greater
                    lb = m + 1
                else:
                    # m could still be the insertion point
                    ub = m
            # lb and ub should be equal and indicate the index of the first element with score less than score.
            # Might be necessary to insert at the end of the list.
            if self.scores[lb] > score:
                self.elts.insert(lb + 1, elt)
                self.scores.insert(lb + 1, score)
            else:
                self.elts.insert(lb, elt)
                self.scores.insert(lb, score)
            # Drop and item from the beam if necessary
            if len(self.scores) > self.size:
                self.elts.pop()
                self.scores.pop()

    def get_elts(self):
        return self.elts

    def get_elts_and_scores(self):
        return zip(self.elts, self.scores)

    def head(self):
        return self.elts[0]


# Indexes a string feat using feature_indexer and adds it to feats.
# If add_to_indexer is true, that feature is indexed and added even if it is new
# If add_to_indexer is false, unseen features will be discarded
def maybe_add_feature(feats, feature_indexer, add_to_indexer, feat):
    if add_to_indexer:
        feats.append(feature_indexer.get_index(feat))
    else:
        feat_idx = feature_indexer.index_of(feat)
        if feat_idx != -1:
            feats.append(feat_idx)


# Computes the dot product over a list of features (i.e., a sparse feature vector)
# and a weight vector (numpy array)
def score_indexed_features(feats, weights):
    score = 0.0
    for feat in feats:
        score += weights[feat]
    return score


##################
# Tests
def test_counter():
    print "TESTING COUNTER"
    ctr = Counter()
    ctr.increment_count("a", 5)
    ctr.increment_count("b", 3)
    print str(ctr.counter)
    for key in ctr.counter.keys():
        print key
    ctr2 = Counter()
    ctr2.increment_count("a", 3)
    ctr2.increment_count("c", 4)
    ctr.add(ctr2)
    print repr(ctr) + " should be ['a: 8', 'c: 4', 'b: 3']"


def test_beam():
    print "TESTING BEAM"
    beam = Beam(3)
    beam.add("a", 5)
    beam.add("b", 7)
    beam.add("c", 6)
    beam.add("d", 4)
    print "Should contain b, c, a: " + repr(beam)
    beam.add("e", 8)
    beam.add("f", 6.5)
    print "Should contain e, b, f: " + repr(beam)

    beam = Beam(5)
    beam.add("a", 5)
    beam.add("b", 7)
    beam.add("c", 6)
    beam.add("d", 4)
    print "Should contain b, c, a, d: " + repr(beam)
    beam.add("e", 8)
    beam.add("f", 6.5)
    print "Should contain e, b, f, c, a: " + repr(beam)

if __name__ == '__main__':
    test_counter()
    test_beam()

    def score_emission(self, sentence, tag_idx, word_posn):
        word = sentence.tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.get_index("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the HMM model. See BadNerModel for an example implementation
    def decode(self,sentence):
        pred_tags = []
        if len(sentence) > 0:
            viterbi_mat = np.zeros((len(self.tag_indexer),len(sentence)), dtype = float)
            backpointer_mat = np.zeros((len(self.tag_indexer),len(sentence)), dtype = float)
            #print self.word_indexer.objs_to_ints

            for word,word_pos in zip(sentence.tokens,range(len(sentence))):
                #print self.word_indexer.get_index(word)
       
                word_idx = self.word_indexer.index_of(str(word)) if self.word_indexer.contains(str(word)) else self.word_indexer.get_index("UNK")
                #print word_idx
                if word_pos == 0:
                    for tag_idx in range(len(self.tag_indexer)):
                        viterbi_mat[tag_idx][word_pos] = self.init_log_probs[tag_idx] + self.emission_log_probs[tag_idx][word_idx]
                        backpointer_mat[tag_idx][word_pos] = 0
                else:
                    for tag_idx in range(len(self.tag_indexer)):
                        viterbi_mat[tag_idx][word_pos] = max([viterbi_mat[s][word_pos-1] + self.transition_log_probs[s][tag_idx] + self.emission_log_probs[tag_idx][word_idx] for s in range(len(self.tag_indexer))])
                        backpointer_mat[tag_idx][word_pos] = np.argmax(np.asarray([viterbi_mat[s][word_pos-1] + self.transition_log_probs[s][tag_idx] for s in range(len(self.tag_indexer))]))
                
            backpointer = np.argmax(np.asarray([viterbi_mat[:,len(sentence) - 1]]))
            pred_tags.append(self.tag_indexer.get_object(backpointer))

            for i in range(len(sentence) - 1):
                pred_tags.append(self.tag_indexer.get_object(backpointer_mat[backpointer][len(sentence) - i - 1]))
                backpointer = backpointer_mat[backpointer][len(sentence) - i - 1]

            pred_tags.reverse()

        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))


# Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
# Any word that only appears once in the corpus is replaced with UNK. A small amount
# of additive smoothing is applied to
def train_hmm_model(sentences):
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter.increment_count(token.word, 1.0)
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in xrange(0, len(sentence)):
            tag_idx = tag_indexer.get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_indexer.get_index(bio_tags[i])] += 1.0
            else:
                transition_counts[tag_indexer.get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print repr(init_counts)
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    '''
    print "Tag indexer: " + repr(tag_indexer)
    print "Initial state log probabilities: " + repr(init_counts)
    print "Transition log probabilities: " + repr(transition_counts)
    print "Emission log probs too big to print..."
    print "Emission log probs for India: " + repr(emission_counts[:,word_indexer.get_index("India")])
    print "Emission log probs for Phil: " + repr(emission_counts[:,word_indexer.get_index("Phil")])
    print "   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)"
    '''
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


# Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
# At test time, unknown words will be replaced by UNKs.
def get_word_index(word_indexer, word_counter, word):
    if word_counter.get_count(word) < 1.5:
        return word_indexer.get_index("UNK")
    else:
        return word_indexer.get_index(word)



class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights, weight_transitions, weight_initial_transitions):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.weight_transitions = weight_transitions
        self.weight_initial_transitions = weight_initial_transitions
 
    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the CRF model. See BadNerModel for an example implementation
    def decode_transition(self, sentence):
        f = open("wordClusters_1000_top2_hashed.json")
        wordClusters = json.load(f)

        pred_tags = []
        if len(sentence) > 0:
            feature_cache = [[[] for k in xrange(0, len(self.tag_indexer))] for j in xrange(0, len(sentence))]
            for word_idx in xrange(0, len(sentence)):
                for tag_idx in xrange(0, len(self.tag_indexer)):
                    feature_cache[word_idx][tag_idx] = extract_emission_features(sentence, word_idx, self.tag_indexer.get_object(tag_idx), self.feature_indexer, add_to_indexer = False)
                    #feature_cache[word_idx][tag_idx] = extract_emission_features_wordClusters(sentence, word_idx, self.tag_indexer.get_object(tag_idx), self.feature_indexer, wordClusters, add_to_indexer = False)


            scorer_mat = np.zeros((len(self.tag_indexer),len(sentence)), dtype = float)
            backpointer_mat = np.zeros((len(self.tag_indexer),len(sentence)), dtype = float)
            for word_idx in xrange(0, len(sentence)):
                if word_idx == 0:
                    for tag_idx in range(len(self.tag_indexer)):
                        scorer_mat[tag_idx][word_idx] = score_indexed_features(feature_cache[word_idx][tag_idx], self.feature_weights) + self.weight_initial_transitions[tag_idx]
                        backpointer_mat[tag_idx][word_idx] = 0
                else:
                    for tag_idx in range(len(self.tag_indexer)):
                        scorer_mat[tag_idx][word_idx] = max([scorer_mat[s][word_idx - 1] +  score_indexed_features(feature_cache[word_idx][tag_idx], self.feature_weights) + self.weight_transitions[s,tag_idx] for s in range(len(self.tag_indexer))])
                        backpointer_mat[tag_idx][word_idx] = np.argmax(np.asarray([scorer_mat[s][word_idx - 1] + self.weight_transitions[s,tag_idx] for s in range(len(self.tag_indexer))]))

            backpointer = np.argmax(np.asarray([scorer_mat[:,len(sentence) - 1]]))
            pred_tags.append(self.tag_indexer.get_object(backpointer))

            for i in range(len(sentence) - 1):
                pred_tags.append(self.tag_indexer.get_object(backpointer_mat[backpointer][len(sentence) - i - 1]))
                backpointer = backpointer_mat[backpointer][len(sentence) - i - 1]

            pred_tags.reverse()
            return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))

    def decode(self, sentence):
        pred_tags = []
        if len(sentence) > 0:
            feature_cache = [[[] for k in xrange(0, len(self.tag_indexer))] for j in xrange(0, len(sentence))]
            for word_idx in xrange(0, len(sentence)):
                for tag_idx in xrange(0, len(self.tag_indexer)):
                    feature_cache[word_idx][tag_idx] = extract_emission_features(sentence, word_idx, self.tag_indexer.get_object(tag_idx), self.feature_indexer, add_to_indexer = False)

            scorer_mat = np.ones((len(self.tag_indexer),len(sentence)), dtype = float) * (- float("inf"))
            backpointer_mat = np.zeros((len(self.tag_indexer),len(sentence)), dtype = float)
            for word_idx in xrange(0, len(sentence)):
                if word_idx == 0:
                    for tag_idx in range(len(self.tag_indexer)):
                        #print self.tag_indexer.get_object(tag_idx)
                        if isI(self.tag_indexer.get_object(tag_idx)):
                            continue
                        else:
                            #scorer_mat[tag_idx][word_idx] = score_indexed_features(feature_cache[word_idx][tag_idx], self.feature_weights) 
                            scorer_mat[tag_idx][word_idx] = np.sum(self.feature_weights[feature_cache[word_idx][tag_idx]])
                        backpointer_mat[tag_idx][word_idx] = 0
                else:
                    for tag_idx in range(len(self.tag_indexer)):
                        temp_array = [(s,scorer_mat[s][word_idx - 1])  for s in range(len(self.tag_indexer))]
                        curr_tag = self.tag_indexer.get_object(tag_idx)
                        maxVal = -float("inf")
                        maxIndex = -1
                        for t in temp_array:
                            if isI(curr_tag) and isO(self.tag_indexer.get_object(t[0])):
                                continue
                            elif isI(curr_tag) and not (get_tag_label(self.tag_indexer.get_object(t[0])) == get_tag_label(curr_tag)):
                                continue
                            else:
                                if t[1] > maxVal:
                                    maxVal = t[1]
                                    maxIndex = t[0]

                        #scorer_mat[tag_idx][word_idx] = maxVal +  score_indexed_features(feature_cache[word_idx][tag_idx], self.feature_weights)
                        scorer_mat[tag_idx][word_idx] = maxVal +  np.sum(self.feature_weights[feature_cache[word_idx][tag_idx]])
                        backpointer_mat[tag_idx][word_idx] = maxIndex


            backpointer = np.argmax(np.asarray([scorer_mat[:,len(sentence) - 1]]))
            pred_tags.append(self.tag_indexer.get_object(backpointer))

            for i in range(len(sentence) - 1):
                pred_tags.append(self.tag_indexer.get_object(backpointer_mat[backpointer][len(sentence) - i - 1]))
                backpointer = backpointer_mat[backpointer][len(sentence) - i - 1]

            pred_tags.reverse()

        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))


def forward_backword_transition(sentence, tag_indexer, weight, emission_features_sentence, weight_transitions, weight_initial_transitions):
    alpha_mat = np.zeros((len(tag_indexer),len(sentence)), dtype = float)
    sentence_tags = sentence.get_bio_tags()
    for word_idx in range(len(sentence)):
            if word_idx == 0:
                for tag_idx in range(len(tag_indexer)):
                    alpha_mat[tag_idx][word_idx] = np.sum(weight[emission_features_sentence[word_idx][tag_idx]]) + weight_initial_transitions[tag_idx]
                #alpha_mat[tag_indexer.get_index(sentence_tags[word_idx])][word_idx] += weight_initial_transitions[tag_indexer.get_index(sentence_tags[word_idx])]
            else:
                for tag_idx in range(len(tag_indexer)):
                    #array_temp = (np.asarray([alpha_mat[s][word_idx-1] + np.sum(weight[emission_features_sentence[word_idx][tag_idx]]) + log_transitions[s][tag_idx] for s in range(len(tag_indexer))]))
                    array_temp = (alpha_mat[:,word_idx-1] + weight_transitions[:, tag_idx]).T
                    alpha_mat[tag_idx][word_idx] = misc.logsumexp(array_temp) + np.sum(weight[emission_features_sentence[word_idx][tag_idx]])
                #alpha_mat[tag_indexer.get_index(sentence_tags[word_idx])][word_idx] += weight_transitions[tag_indexer.get_index(sentence_tags[word_idx - 1])][tag_indexer.get_index(sentence_tags[word_idx])]

    beta_mat = np.zeros((len(tag_indexer),len(sentence)), dtype = float)
    for word_idx in range(len(sentence))[::-1]:
        if word_idx == len(sentence) - 1:
            for tag_idx in range(len(tag_indexer)):
                beta_mat[tag_idx][word_idx] = 0
        else:
            for tag_idx in range(len(tag_indexer)):
                #print weight_transitions[tag_idx,:]
                array_temp = np.asarray([np.sum(weight[emission_features_sentence[word_idx + 1][s]]) + weight_transitions[tag_idx,s] + beta_mat[s,word_idx + 1] for s in range(len(tag_indexer))]).T
                #array_temp  = beta_mat[:,word_idx + 1] + a
                
                beta_mat[tag_idx][word_idx] = misc.logsumexp(array_temp)
            #beta_mat[tag_indexer.get_index(sentence_tags[word_idx + 1])][word_idx] += weight_transitions[tag_indexer.get_index(sentence_tags[word_idx])][tag_indexer.get_index(sentence_tags[word_idx + 1])]


    array_temp = alpha_mat[:,len(sentence) - 1]
    Z = array_temp[0]
    for t in array_temp[1:]:
        Z = np.logaddexp(t, Z)
    #print alpha_mat,beta_mat,Z
    #print alpha_mat
    return [alpha_mat,beta_mat, Z]


def forward_backword_transition_naive(sentence, tag_indexer, weight, emission_features_sentence, weight_transitions, weight_initial_transitions):
    alpha_mat = np.zeros((len(tag_indexer),len(sentence)), dtype = float)
    for word_idx in range(len(sentence)):
            if word_idx == 0:
                for tag_idx in range(len(tag_indexer)):
                    alpha_mat[tag_idx][word_idx] = np.sum(weight[emission_features_sentence[word_idx][tag_idx]]) + weight_initial_transitions[tag_idx]
                    #alpha_mat[tag_idx][word_idx] = np.sum(weight[emission_features_sentence[word_idx][tag_idx]])
            else:
                for tag_idx in range(len(tag_indexer)):
                    array_temp = (np.asarray([alpha_mat[s][word_idx-1] + weight_transitions[s][tag_idx] for s in range(len(tag_indexer))]))
                    #array_temp = (alpha_mat[:,word_idx-1]).T
                    alpha_mat[tag_idx][word_idx] = misc.logsumexp(array_temp) + np.sum(weight[emission_features_sentence[word_idx][tag_idx]])

    beta_mat = np.zeros((len(tag_indexer),len(sentence)), dtype = float)
    for word_idx in range(len(sentence))[::-1]:
        if word_idx == len(sentence) - 1:
            for tag_idx in range(len(tag_indexer)):
                beta_mat[tag_idx][word_idx] = 0
        else:
            for tag_idx in range(len(tag_indexer)):
                print weight_transitions[tag_idx,:]
                array_temp  = np.asarray([beta_mat[s][word_idx + 1] + np.sum(weight[emission_features_sentence[word_idx + 1][s]]) + weight_transitions[tag_idx][s] for s in range(len(tag_indexer))])
                #a = np.asarray([np.sum(weight[emission_features_sentence[word_idx + 1][s]]) for s in range(len(tag_indexer))]).T
                #array_temp  = beta_mat[:,word_idx + 1] + a 
                beta_mat[tag_idx][word_idx] = misc.logsumexp(array_temp)
                '''
                for t in array_temp:
                    beta_mat[tag_idx][word_idx] = np.logaddexp(t, beta_mat[tag_idx][word_idx])
                '''

    array_temp = alpha_mat[:,len(sentence) - 1]
    Z = array_temp[0]
    for t in array_temp[1:]:
        Z = np.logaddexp(t, Z)
    #print alpha_mat,beta_mat,Z
    #print alpha_mat
    return [alpha_mat,beta_mat, Z]


def forward_backword(sentence, tag_indexer, weight, emission_features_sentence):
    alpha_mat = np.zeros((len(tag_indexer),len(sentence)), dtype = float)
    for word_idx in range(len(sentence)):
            if word_idx == 0:
                for tag_idx in range(len(tag_indexer)):
                    alpha_mat[tag_idx][word_idx] = np.sum(weight[emission_features_sentence[word_idx][tag_idx]])
            else:
                for tag_idx in range(len(tag_indexer)):
                    array_temp = (alpha_mat[:,word_idx-1]).T
                    alpha_mat[tag_idx][word_idx] = misc.logsumexp(array_temp) + np.sum(weight[emission_features_sentence[word_idx][tag_idx]])

    beta_mat = np.zeros((len(tag_indexer),len(sentence)), dtype = float)
    for word_idx in range(len(sentence))[::-1]:
        if word_idx == len(sentence) - 1:
            for tag_idx in range(len(tag_indexer)):
                beta_mat[tag_idx][word_idx] = 0
        else:
            for tag_idx in range(len(tag_indexer)):
                #array_temp  = np.asarray([beta_mat[s][word_idx + 1] + np.sum(weight[emission_features_sentence[word_idx + 1][s]]) + log_transitions[tag_idx][s] for s in range(len(tag_indexer))])
                a = np.asarray([np.sum(weight[emission_features_sentence[word_idx + 1][s]]) for s in range(len(tag_indexer))]).T
                array_temp  = beta_mat[:,word_idx + 1] + a 
                beta_mat[tag_idx][word_idx] = misc.logsumexp(array_temp)


    array_temp = alpha_mat[:,len(sentence) - 1]
    Z = array_temp[0]
    for t in array_temp[1:]:
        Z = np.logaddexp(t, Z)

    return [alpha_mat,beta_mat, Z]


def calculate_weight_gradient(feature_cache, alpha_mat,beta_mat, Z, weight, sentence, tag_indexer):
    sentence_tags = sentence.get_bio_tags()
    learning_rate = 0.1
    for word_idx in range(len(sentence)):
        tag_idx_gold = tag_indexer.get_index(sentence_tags[word_idx])
        for feat_idx in feature_cache[word_idx][tag_idx_gold]:
            weight[feat_idx] = weight[feat_idx] + learning_rate * 1

        for tag_idx in range(len(tag_indexer)):
            for feat_idx in feature_cache[word_idx][tag_idx]:
                temp = alpha_mat[tag_idx][word_idx] + beta_mat[tag_idx][word_idx] - Z
                weight[feat_idx] = weight[feat_idx] - learning_rate * np.exp(temp)
    return weight


def calculate_weight_transition_gradient(alpha_mat, beta_mat, Z, weight_transitions, sentence, tag_indexer, weight, feature_cache):
    sentence_tags = sentence.get_bio_tags()
    learning_rate = 0.1
    #wI = copy.deepcopy(weight_initial_transitions)
    wT = copy.deepcopy(weight_transitions)
    for word_idx in range(0,len(sentence)):
        tag_idx_gold = tag_indexer.get_index(sentence_tags[word_idx])
        if word_idx == 0:
            continue
            #weight_initial_transitions[tag_idx_gold] += learning_rate * 1
        else:
            tag_idx_gold_prev = tag_indexer.get_index(sentence_tags[word_idx - 1])
            weight_transitions[tag_idx_gold_prev][tag_idx_gold] += learning_rate * 1

         
        for tag_curr in range(len(tag_indexer)):
            feat_idx = feature_cache[word_idx][tag_curr]
            if word_idx == 0:
                continue
                '''
                temp = wI[tag_curr] + np.sum(weight[feat_idx]) + beta_mat[tag_curr][word_idx] - Z
                print temp
                weight_initial_transitions[tag_curr] -= learning_rate * np.exp(temp)
                '''
            for tag_prev in range(len(tag_indexer)):
                temp = alpha_mat[tag_prev][word_idx-1] + wT[tag_prev][tag_curr] + beta_mat[tag_curr][word_idx] + np.sum(weight[feat_idx]) - Z
                weight_transitions[tag_prev][tag_curr] -= learning_rate * np.exp(temp)

    return weight_transitions


def initialize_weight_transitions(sentences, tag_indexer):
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.00000001
    init_counts = np.ones(len(tag_indexer), dtype=float) * 0.00000001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in xrange(0, len(sentence)):
            tag_idx = tag_indexer.get_index(bio_tags[i])
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.get_index(bio_tags[i - 1])][tag_idx] += 1.0

    init_counts = np.log(init_counts / init_counts.sum())
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    
    return init_counts,transition_counts


# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences):
    tag_indexer = Indexer()
    pos_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    print "Extracting features"
    feature_indexer = Indexer()

    '''
    bigram_indexer = Indexer()
    bigram_indexer.get_index("UNK")
    bigram_counter = Counter()
    for sentence in sentences:
        sentence_tokens = sentence.tokens
        for word_idx in range(1,len(sentence.tokens)):
            bigram_counter.increment_count(str(sentence_tokens[word_idx - 1].word) + "###" + str(sentence_tokens[word_idx].word), 1.0)
    for sentence in sentences:
        sentence_tokens = sentence.tokens
        for word_idx in range(1,len(sentence.tokens)):
            get_word_index(bigram_indexer, bigram_counter, str(sentence_tokens[word_idx - 1].word) + "###" + str(sentence_tokens[word_idx].word))
    '''

    
    #f = open("wordClusters_1000_top2_hashed.json")
    
    #wordClusters = json.load(f)

    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in xrange(0, len(tag_indexer))] for j in xrange(0, len(sentences[i]))] for i in xrange(0, len(sentences))]
    for sentence_idx in xrange(0, len(sentences)):
        '''
        if sentence_idx % 100 == 0:
            print "Ex " + repr(sentence_idx) + "/" + repr(len(sentences))
        '''
        for word_idx in xrange(0, len(sentences[sentence_idx])):
            for tag_idx in xrange(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx], word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
                #feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features_wordClusters(sentences[sentence_idx], word_idx, tag_indexer.get_object(tag_idx), feature_indexer, wordClusters, add_to_indexer=True)
   
    weight_initial_transitions, weight_transitions = initialize_weight_transitions(sentences, tag_indexer)

    weight = np.ones(len(feature_indexer), dtype = float)

    epoch = 2
    for e in range(epoch):
        print "Epoch:" + str(e)
        for sentence_idx in xrange(0,len(sentences)):
            [alpha_mat,beta_mat, Z] = forward_backword_transition(sentences[sentence_idx], tag_indexer, weight, feature_cache[sentence_idx], weight_transitions, weight_initial_transitions)
            #[alpha_mat,beta_mat, Z] = forward_backword(sentences[sentence_idx], tag_indexer, weight, feature_cache[sentence_idx])
            #[alpha_mat,beta_mat, Z] = forward_backword_transition_naive(sentences[sentence_idx], tag_indexer, weight, feature_cache[sentence_idx], weight_transitions, weight_initial_transitions)
            weight_transitions = calculate_weight_transition_gradient(alpha_mat, beta_mat, Z, weight_transitions,sentences[sentence_idx], tag_indexer, weight, feature_cache[sentence_idx])
            weight = calculate_weight_gradient(feature_cache[sentence_idx], alpha_mat, beta_mat, Z, weight, sentences[sentence_idx], tag_indexer)
        print "Epoch:" + str(e) + " Done"
    #print weight_transitions

    return CrfNerModel(tag_indexer, feature_indexer, weight, weight_transitions, weight_initial_transitions)


# Extracts emission features for tagging the word at word_index with tag.
# add_to_indexer is a boolean variable indicating whether we should be expanding the indexer or not:
# this should be True at train time (since we want to learn weights for all features) and False at
# test time (to avoid creating any features we don't have weights for).
def extract_emission_features(sentence, word_index, tag, feature_indexer, add_to_indexer):
    feats = []
    curr_word = sentence.tokens[word_index].word

    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in xrange(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence):
            active_word = "</s>"
        else:
            active_word = sentence.tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence):
            active_pos = "</S>"
        else:
            active_pos = sentence.tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    
    ## Bigram features
    for idx_offset in xrange(-1, 2):
        if word_index + idx_offset < 0:
            continue
        elif word_index + idx_offset >= len(sentence):
            continue
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Bigram" + repr(idx_offset) + "=" + str(sentence.tokens[word_index + idx_offset].word) + "###" + str(sentence.tokens[word_index].word))
    

    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in xrange(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in xrange(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)

def extract_emission_features_wordClusters(sentence, word_index, tag, feature_indexer, wordClusters, add_to_indexer):
    feats = []
    curr_word = sentence.tokens[word_index].word

    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in xrange(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence):
            active_word = "</s>"
        else:
            active_word = sentence.tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence):
            active_pos = "</S>"
        else:
            active_pos = sentence.tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    
    '''
    ## Bigram features
    for idx_offset in xrange(-1, 2):
        if word_index + idx_offset < 0:
            continue
        elif word_index + idx_offset >= len(sentence):
            continue
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Bigram" + repr(idx_offset) + "=" + str(sentence.tokens[word_index + idx_offset].word) + "###" + str(sentence.tokens[word_index].word))
    
    '''

    cluster = False
    if curr_word.lower()[0] in wordClusters.keys():
        if curr_word in wordClusters[curr_word.lower()[0]].keys():
            cluster = wordClusters[curr_word.lower()[0]][curr_word]
            maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordCluster=" + cluster)
            
    
    #prefixes of the clusters
    if cluster:
        prefix_sizes = [3, 4, 5, 7]
        for size in prefix_sizes:        
            if size > len(cluster):
                continue
            else:
                maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordCluster" + str(size) + "=" + cluster[0:size])

    
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in xrange(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in xrange(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)
