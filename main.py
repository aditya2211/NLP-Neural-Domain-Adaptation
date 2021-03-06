import sys
from nerdata import *
import time
from os import listdir
from utils import *
from models import *
from models_lstm import *

class IndexedSentence:
    def __init__(self, indexed_words, labels, pos, domain):
        self.indexed_words = indexed_words
        self.label = labels
        self.pos = pos
        self.domain = domain

    def __repr__(self):
        return repr(self.indexed_words) + "; labels=" + repr(self.label)

    def get_indexed_words_reversed(self):
        return [self.indexed_words[len(self.indexed_words) - 1 - i] for i in xrange(0, len (self.indexed_words))]

def read_and_index(sentences, indexer, tag_indexer, domain, add_to_indexer=False, word_counter=None):
    
    exs = []
    for sent_idx in xrange(0,len(sentences)):
        sent = []
        pos = []
        #print sent
        tags=sentences[sent_idx].get_bio_tags()
        
        for word_idx in xrange(0,len(sentences[sent_idx])):
            sent.append(sentences[sent_idx].tokens[word_idx].word)
            pos.append(sentences[sent_idx].tokens[word_idx].pos)
        
        indexed_sent = [indexer.get_index(word) if indexer.contains(word) or add_to_indexer else indexer.get_index("UNK") for word in sent]
        indexed_label = [tag_indexer.get_index(word) for word in tags]
        
        exs.append(IndexedSentence(indexed_sent,indexed_label, pos, domain))
    return exs
         



if __name__ == '__main__':

	#trainData = read_data_conll("./Datasets/conll-2012-en/train/a2e_0001.v4_gold_conll")
    
    word_vectors = read_word_embeddings("./Datasets/glove.6B.300d-relativized.txt")
    trainData1 = []
    trainData2 = []
    tag_indexer = Indexer()
    
    maxlen=0
    path = "./Datasets/conll-2012-en/train/"
    for f in listdir(path):
    	if str(f)[0:3] == "a2e":
    		if 'gold' in str(f):
    			trainData1 += read_data_conll(path + "/" + f)
        if str(f)[0:3] == "wsj":
            if 'gold' in str(f):
                trainData2 += read_data_conll(path + "/" + f)
            
    devData = []
    path = "./Datasets/conll-2012-en/dev/"
    for f in listdir(path):
    	if str(f)[0:3] == "a2e":
    		if 'gold' in str(f):
    			devData += read_data_conll(path + "/" + f)


    #devData = trainData[1800:]
    #trainData = trainData[:1800]  
    trainData = trainData1 + trainData2      
    for sent_idx in xrange(0,len(trainData)):
        
        if len(trainData[sent_idx])>maxlen:
            maxlen=len(trainData[sent_idx])
        tags = trainData[sent_idx].get_bio_tags()
        for tag in tags:
            tag_indexer.get_index(tag)
        
    for sent_idx in xrange(0,len(devData)):
        if len(devData[sent_idx])>maxlen:
            maxlen=len(devData[sent_idx])
        tags = devData[sent_idx].get_bio_tags()
        for tag in tags:
            tag_indexer.get_index(tag) 

    train_exs = read_and_index(trainData1,word_vectors.word_indexer,tag_indexer, 0)
    train_exs += read_and_index(trainData2,word_vectors.word_indexer,tag_indexer, 1)
    dev_exs = read_and_index(devData,word_vectors.word_indexer,tag_indexer, 0)


    
    for sent_idx in xrange(0,len(devData)):
        for word_idx in xrange(0,len(devData[sent_idx])):
            dev_exs[sent_idx].label[word_idx] = 0
            
            
    dev_results = train_fancy(train_exs,dev_exs,devData,tag_indexer,word_vectors)
    #dev_results = dev_exs
    devDecoded=[]
    for sent_idx in xrange(0,len(devData)):
        tags = []        
        for word_idx in xrange(0,len(devData[sent_idx])):
            tags.append(tag_indexer.get_object(dev_results[sent_idx].label[word_idx]))
        
        devDecoded.append(LabeledSentence(devData[sent_idx].tokens, chunks_from_bio_tag_seq(tags)))
        
    print_evaluation(devData, devDecoded)
    
    
    """
    LSTM
    """
    
     