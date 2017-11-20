import sys
from nerdata import *
import time
from os import listdir
from utils import *
from models import *



if __name__ == '__main__':

	#trainData = read_data_conll("./Datasets/conll-2012-en/train/a2e_0001.v4_gold_conll")

    trainData = []
    path = "./Datasets/conll-2012-en/train/"
    for f in listdir(path):
    	if str(f)[0:3] == "a2e":
    		if 'gold' in str(f):
    			trainData += read_data_conll(path + "/" + f)

    crf_model = train_crf_model(trainData[1:1000])
    dev_decoded = [crf_model.decode_transition(test_ex) for test_ex in trainData]

    print_evaluation(dev, dev_decoded)
