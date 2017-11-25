# models.py

import tensorflow as tf
import numpy as np
import random
from utils import *
from models import *
from nerdata import *


# Returns a new numpy array with the data from np_arr padded to be of length length. If length is less than the
# length of the base array, truncates instead.
def pad_to_length(np_arr, length):
    result = np.zeros(length)
    
    result[0:np_arr.shape[0]] = np_arr
    return result


# Train a feedforward neural network on the given training examples, using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value). Returned predictions
# should be SentimentExample objects with predicted labels and the same sentences as input (but these won't be
# read for evaluation anyway)
def train_ffnn(train_exs, dev_exs, test_exs, word_vectors):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])
    
    wordvectors = np.array(word_vectors.vectors, dtype="float32")
    #print wordvectors[0]
    
    #train_xs = np.array([[0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1]])
    #train_ys = np.array([0, 1, 1, 1, 1, 0])
    
    train_xs = train_mat
    dev_xs = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    test_xs = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    train_ys = train_labels_arr
    dev_ys = np.array([ex.label for ex in dev_exs])
    
    # Define some constants
    # Inputs are of size 2
    feat_vec_size = 300
    # Let's use 10 hidden units
    embedding_size1 =300
    embedding_size2 = 300
    embedding_size3 = 300
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    indices = tf.placeholder(tf.int32)
    #fx = tf.get_variable("fx", feat_vec_size)
    word_embeds = tf.Variable(wordvectors,trainable=True,name="word_embeds")
    fx = tf.reduce_mean(tf.nn.embedding_lookup(word_embeds,indices),0)
    keep_prob1 = tf.placeholder(tf.float32)  # DROP-OUT here
    keep_prob2 = tf.placeholder(tf.float32)
    # DROP-OUT here
    # Other initializers like tf.random_normal_initializer are possible too
    
    fx_dropout = tf.nn.dropout(fx, keep_prob1)
    
    
    V1 = tf.get_variable("V1", [embedding_size1, feat_vec_size], initializer=tf.contrib.layers.xavier_initializer(seed=0)   )
    z1 = tf.nn.relu(tf.tensordot(V1, (fx_dropout), 1))
    z1_dropout = tf.nn.dropout(z1, keep_prob2) 
    V2 = tf.get_variable("V2", [embedding_size2, embedding_size1], initializer=tf.contrib.layers.xavier_initializer(seed=0)  )
    z2 = tf.nn.relu(tf.tensordot(V2, z1_dropout, 1))
    z2_dropout = tf.nn.dropout(z2, keep_prob2) 
    V3 = tf.get_variable("V3", [embedding_size3, embedding_size2] )
    z3 = tf.nn.relu(tf.tensordot(V3, z2_dropout, 1))
    z3_dropout = tf.nn.dropout(z3, keep_prob2) 
    
    W = tf.get_variable("W", [num_classes, embedding_size3])
    
    probs = tf.nn.softmax(tf.tensordot(W, z1_dropout, 1))
    # This is the actual prediction -- not used for training but used for inference
    one_best = tf.argmax(probs)

    # Input for the gold label so we can compute the loss
    label = tf.placeholder(tf.int32, 1)
    # Convert a value-based representation (e.g., [2]) into the one-hot representation ([0, 0, 1])
    # Because label is a tensor of dimension one, the one-hot is actually [[0, 0, 1]], so
    # we need to flatten it.
    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes])
    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))


    # TRAINING ALGORITHM CUSTOMIZATION
    # Decay the learning rate by a factor of 0.99 every 10 gradient steps (for larger datasets you'll want a slower
    # weight decay schedule
    decay_steps = 10
    learning_rate_decay_factor = .999995
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.0001
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer(lr)
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    #tf.reset_default_graph()
    num_epochs = 25
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    max_acc=0
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_xs)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                count=0
                averaged_word_vector_input = np.zeros(300,dtype=float)
                for j in xrange(0,train_seq_lens[ex_idx]):
                    averaged_word_vector_input+= wordvectors[train_xs[ex_idx][j].astype(int)]
                       
                
                averaged_word_vector_input = averaged_word_vector_input/train_seq_lens[ex_idx]
                #print averaged_word_vector_input
                #print len(averaged_word_vector_input)
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {indices: train_xs[ex_idx][:train_seq_lens[ex_idx]].astype(int),
                                                                                  label: np.array([train_ys[ex_idx]]), keep_prob1:.8,keep_prob2:.5})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        # Evaluate on the train set
            train_correct = 0
            for ex_idx in xrange(0, len(train_xs)):
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and
                
                averaged_word_vector_input = np.zeros(300,dtype=float)
                for j in xrange(0,train_seq_lens[ex_idx]):
                    averaged_word_vector_input+= wordvectors[train_xs[ex_idx][j].astype(int)]
                
                averaged_word_vector_input = averaged_word_vector_input/train_seq_lens[ex_idx]
                [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                                      feed_dict={indices: train_xs[ex_idx][:train_seq_lens[ex_idx]].astype(int), keep_prob1:1,keep_prob2:1})
                if (train_ys[ex_idx] == pred_this_instance):
                    train_correct += 1
    #            print "Example " + repr(train_xs[ex_idx]) + "; gold = " + repr(train_ys[ex_idx]) + "; pred = " +\
    #                  repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
                #print "  Hidden layer activations for this example: " + repr(z_this_instance)
            print "training accuracy: "+repr(float(train_correct)/float(len(train_ys))*100) + " correct after training"
           
            dev_correct = 0
            for ex_idx in xrange(0, len(dev_xs)):
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                averaged_word_vector_input = np.zeros(300,dtype=float)
                for j in xrange(0,dev_seq_lens[ex_idx]):
                    averaged_word_vector_input+= wordvectors[dev_xs[ex_idx][j].astype(int)]
                
                averaged_word_vector_input = averaged_word_vector_input/dev_seq_lens[ex_idx]
                [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                                      feed_dict={indices: dev_xs[ex_idx][:dev_seq_lens[ex_idx]].astype(int), keep_prob1:1,keep_prob2:1})
                if (dev_ys[ex_idx] == pred_this_instance):
                    dev_correct += 1
    #            print "Example " + repr(dev_xs[ex_idx]) + "; gold = " + repr(dev_ys[ex_idx]) + "; pred = " +\
    #                  repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
    #            #print "  Hidden layer activations for this example: " + repr(z_this_instance)
            print "dev accuracy: "+repr(float(dev_correct)/float(len(dev_ys))*100) + " correct after training"
            
            if(float(dev_correct)/float(len(dev_ys))*100 > max_acc):
                print "Writing Test Labels"
                for ex_idx in xrange(0, len(test_xs)):
                    if(ex_idx==100):
                        print "Wrote 100th Test Label"
                    [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                                      feed_dict={indices: test_xs[ex_idx][:test_seq_lens[ex_idx]].astype(int), keep_prob1:1,keep_prob2:1})
                    test_exs[ex_idx].label = pred_this_instance
                max_acc = float(dev_correct)/float(len(dev_ys))*100


    #raise Exception("Not implemented")


# Analogous to train_ffnn, but trains your fancier model.
    
    
def get_matrix_illegal(tag_indexer):
    trans = np.zeros((len(tag_indexer), len(tag_indexer)))
    for i in range(len(tag_indexer)):
        label1 = tag_indexer.get_object(i)
        for j in range(len(tag_indexer)):
            label2 = tag_indexer.get_object(j)
            if isB(label1) and isI(label2):
                if get_tag_label(label1) != get_tag_label(label2):
                    trans[i,j] = - np.inf
            if isO(label1) and isI(label2):
                trans[i,j] = - np.inf
            if isI(label1) and isI(label2):
                if get_tag_label(label1) != get_tag_label(label2):
                    trans[i,j] = - np.inf

    #print trans
    return trans

    
        
def train_fancy(train_exs, dev_exs, devData, tag_indexer, word_vectors):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 200
    feat_vec_size = 300
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    
    #print test_exs[0].label
    # Also store the sequence lengths -- this could be useful for training LSTMs
    
    
    # Labels
    
    filters = np.zeros([3,feat_vec_size,feat_vec_size], dtype ='float32')


    for i in xrange(0,3):
        for j in xrange(0,feat_vec_size):
            for k in xrange(0,feat_vec_size):
                if j==k:
                    filters[i][j][k]=1
                    
    wordvectors = np.array(word_vectors.vectors, dtype="float32")
    #print wordvectors[0]
    
    #train_xs = np.array([[0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1]])
    #train_ys = np.array([0, 1, 1, 1, 1, 0])
    
    train_xs = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    dev_xs = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    #test_xs = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    #test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    train_ys =  np.array([pad_to_length(np.array(ex.label), seq_max_len) for ex in train_exs])
    dev_ys = np.array([pad_to_length(np.array(ex.label), seq_max_len) for ex in dev_exs])
    
    # Define some constants
    # Inputs are of size 2
    
    # Let's use 10 hidden units
    embedding_size1 = 300
    embedding_size2 = 300
    cell_size=50
    fclayer = 100
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = len(tag_indexer)

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    indices = tf.placeholder(tf.int32)
    #fx1 = tf.placeholder(tf.float64,[1,60,300])
    #fx = tf.placeholder(tf.float32, [1,60,300])
    #fx1 = tf.Variable(tf.zeros([1, 60, 300]),dtype=tf.float32)
    word_embeds = tf.Variable(wordvectors, name="word_embeds")
    fx1 = tf.nn.embedding_lookup(wordvectors,indices)
    #fx1 = tf.nn.embedding_lookup(word_embeds,indices)
    fx1 = tf.reshape(fx1,[1,seq_max_len,feat_vec_size])
    #fx2 = tf.Variable(tf.zeros([1, 200, 300]),dtype=tf.float32)
    #filters = tf.get_variable("filters", [3, 300, 100], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    fx2 = tf.nn.conv1d(fx1,filters,stride=1, padding = 'SAME')
    
    
    zeros_pad = np.zeros([1,1,feat_vec_size],dtype='float32')
    fx1 = tf.concat([zeros_pad,fx1],1)
    fx1 = tf.concat([fx1,zeros_pad],1)
    
    print fx1.shape
    
    
    left_concat = tf.concat([fx1[0][0:seq_max_len],fx1[0][1:seq_max_len+1]],1)
    print left_concat.shape
    right_concat = tf.concat([left_concat,fx1[0][2:seq_max_len+2]],1)
    print right_concat.shape
    
    fx1_concat = tf.reshape(right_concat,[1,seq_max_len,3*feat_vec_size])
    
    
    x_lengths =  tf.placeholder(tf.int32)
    #fx2 = tf.reduce_mean(tf.nn.embedding_lookup(word_embeds,indices[:x_lengths[0]]),0)
    #x_lengths = tf.reshape(x_lengths,[1])
    #fx = tf.reduce_mean(tf.nn.embedding_lookup(word_embeds,indices),0)
    keep_prob1 = tf.placeholder(tf.float32)  # DROP-OUT here
    keep_prob2 = tf.placeholder(tf.float32)
    #transitions = tf.Variable(get_matrix_illegal(tag_indexer),name = "transitions",dtype = tf.float32)
    transitions = tf.get_variable("transitions", [len(tag_indexer),len(tag_indexer)], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # DROP-OUT here
    # Other initializers like tf.random_normal_initializer are possible too
    
    #fx_dropout = tf.nn.dropout(fx, keep_prob1)
    """LSTM"""
    """ 
    #cell1 = tf.contrib.rnn.LSTMBlockCell(num_units=cell_size)
    #cell2 = tf.contrib.rnn.LSTMBlockCell(num_units=cell_size)
    cell1 = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, state_is_tuple=True)
    cell2 = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, state_is_tuple=True)
    #cell3 = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, use_peepholes=True,state_is_tuple=True)
    stacked = tf.contrib.rnn.MultiRNNCell([cell1,cell2])
    outputs, last_states = tf.nn.dynamic_rnn(cell1,fx1, dtype=tf.float32,sequence_length=x_lengths)
    """    

    
    """BiLSTM"""
    
    cellf1 = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, state_is_tuple=True)
    cellb1 = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, state_is_tuple=True)
    cellf2 = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, state_is_tuple=True)
    cellb2 = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, state_is_tuple=True)

    cellf = tf.contrib.rnn.MultiRNNCell([cellf1,cellf2])
    cellb = tf.contrib.rnn.MultiRNNCell([cellb1,cellb2])
    

    cellf_do = tf.contrib.rnn.DropoutWrapper(cellf,keep_prob1,keep_prob2,keep_prob2) 
    cellb_do = tf.contrib.rnn.DropoutWrapper(cellb,keep_prob1,keep_prob2,keep_prob2) 
    bi_outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cellf1,cellb1,fx1_concat, dtype=tf.float32,sequence_length=x_lengths)
    
    outputs = tf.concat([bi_outputs[0],bi_outputs[1]],2)
    print outputs.shape
    
    #V1 = tf.get_variable("V1", [embedding_size1, feat_vec_size], initializer=tf.contrib.layers.xavier_initializer(seed=0)   )
    #z1 = tf.nn.relu(tf.tensordot(V1, fx_dropout, 1))
    #z1_dropout = tf.nn.dropout(z1, keep_prob2) 
    #V2 = tf.get_variable("V2", [embedding_size2, embedding_size1] )
    #z2 = tf.nn.relu(tf.tensordot(V2, z1_dropout, 1))
    #z2_dropout = tf.nn.dropout(z2, keep_prob2) 
    
    #W = tf.get_variable("W", [num_classes, cell_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=0)  )
    #z1 = tf.nn.relu(tf.tensordot(W, outputs[0][x_lengths[0]-1], 1))
    """
    fx2_dropout = tf.nn.dropout(fx2, keep_prob1) 
    V1 = tf.get_variable("V1", [embedding_size1, feat_vec_size], initializer=tf.contrib.layers.xavier_initializer(seed=0)   )
    z1 = tf.nn.relu(tf.tensordot(V1, (fx2_dropout), 1))
    z1_dropout = tf.nn.dropout(z1, keep_prob2) 
    V2 = tf.get_variable("V2", [embedding_size2, embedding_size1], initializer=tf.contrib.layers.xavier_initializer(seed=0)  )
    z2 = tf.nn.relu(tf.tensordot(V2, z1_dropout, 1))
    z2_dropout = tf.nn.dropout(z2, keep_prob2) 
    """
    
    
    
    
    #V1 = tf.get_variable("V1", [num_classes, fclayer], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=0)  )
    #b = tf.get_variable("b", num_classes, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=0)  )
    #concat_output = tf.Variable(tf.zeros(2*cellsize), dtype=tf.float32)
    #concat_output = tf.concat([outputs[0][0][x_lengths-1] outputs[0][1][0]],0)
    #W = tf.get_variable("W", [num_classes, cell_size*2], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    #probs = tf.nn.softmax(tf.tensordot(W, tf.concat([outputs[0][0][x_lengths[0]-1],outputs[1][0][0]],0), 1))
    #concat_output = tf.concat([last_states[0],last_states[1]],1)
    #concat_output = tf.concat([outputs[0][x_lengths[0]-1],z2_dropout],0)
    #concat_output = tf.reduce_mean(tf.concat(outputs,2)[0],0)
    #concat_output_do = tf.nn.dropout(concat_output,keep_prob2)
    #z1 = tf.nn.relu(tf.tensordot(V1, concat_output_do, 1))
    #z1_dropout = tf.nn.dropout(z1,keep_prob2)
    label_orig = tf.placeholder(tf.int32)
    W = tf.get_variable("W", [ cell_size*2, num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    #b = tf.get_variable("b", [num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    """
    lstm_outputs = tf.reshape(outputs[0][:x_lengths[0]], [-1, cell_size])
    logits = tf.matmul(lstm_outputs, W)# + b
    softmax = tf.nn.softmax(logits)
    preds = tf.argmax(softmax, 1)
    """
    
    #lstm_outputs = tf.reshape(outputs, [1,200, cell_size])
    label = tf.reshape(label_orig,[1,200])
    outputpred = tf.reshape(outputs[0], [-1, cell_size*2])
    logits =tf.matmul(outputpred, W)# + b
    preds = logits[:x_lengths[0]]
    
    #print transitions.shape
  
    
    loss,_ = tf.contrib.crf.crf_log_likelihood(tf.reshape(logits,[1,200,num_classes]),label,x_lengths,transitions)
    loss = tf.negative(tf.reduce_mean(loss))
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label[:x_lengths[0]],logits=logits))
    
    #print loss.shape
    #print lstm_outputs.shape
    #b = tf.get_variable("b", [num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    #probs = tf.nn.softmax(tf.tensordot(V1, concat_output, 1))
    #inputfinal = tf.reduce_mean(outputs[0][x_lengths[0]-1],0)
    #probs = tf.nn.softmax(tf.tensordot(W, tf.concat([tf.reduce_mean(outputs[0][:x_lengths[0]],0),tf.reshape(concat_output,[2*cell_size])],0), 1))
    #probs = tf.nn.softmax(tf.tensordot(W, tf.nn.sigmoid(outputs[0][x_lengths[0]-1]), 1))
    
    #probs = tf.nn.softmax(tf.tensordot(W, tf.transpose(outputs[0][:x_lengths[0]]), 1))
    
    #label_onehot = tf.reshape(tf.one_hot(label[i], num_classes), shape=[num_classes])
    
    
    #loss=0
    
    # This is the actual prediction -- not used for training but used for inference
    
    #b = tf.get_variable("b", [1,num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    
   

    # Input for the gold label so we can compute the loss
    
    
    # Convert a value-based representation (e.g., [2]) into the one-hot representation ([0, 0, 1])
    # Because label is a tensor of dimension one, the one-hot is actually [[0, 0, 1]], so
    # we need to flatten it.
    
   


    

    # TRAINING ALGORITHM CUSTOMIZATION
    # Decay the learning rate by a factor of 0.99 every 10 gradient steps (for larger datasets you'll want a slower
    # weight decay schedule
    decay_steps = 10
    learning_rate_decay_factor = .999995
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.001
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer(lr)
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')


    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    #tf.reset_default_graph()
    num_epochs = 20
    ensemble_num=1
    devres= np.zeros((ensemble_num,len(dev_xs),2),dtype=float)
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    max_acc = 0
 
        
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            print "Epoch# "+str(i) + " started"
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            #sample = np.random.choice(len(train_xs), size=len(train_xs), replace=False)
            for ex_idx in xrange(0, len(train_xs)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                
                #print averaged_word_vector_input
                #print len(averaged_word_vector_input)
                #print train_ys[0].shape
                [_, loss_this_instance, summary] = sess.run([ train_op, loss, merged], feed_dict = {indices: train_xs[ex_idx].astype(int),
                                                                                  label: np.reshape(train_ys[ex_idx],[1,200]), keep_prob1:0.8,keep_prob2:0.8,x_lengths:[train_seq_lens[ex_idx].astype(int)]})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_instance
                
                #print loss_this_instance
                #if math.isnan(loss_this_instance):
                #print str(ex_idx) + " "+str(p1)

            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
            for ex_idx in xrange(0, len(dev_xs)):
            
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                [pred_this_instance,trans] = sess.run([preds,transitions],feed_dict={indices: dev_xs[ex_idx].astype(int), keep_prob1:1,keep_prob2:1,x_lengths:[dev_seq_lens[ex_idx].astype(int)]})
            #print pred_this_instance
                a,b = tf.contrib.crf.viterbi_decode(pred_this_instance, trans)
                dev_exs[ex_idx].label = a
            devDecoded=[]
            dev_results = dev_exs
            for sent_idx in xrange(0,len(devData)):
                tags = []        
                for word_idx in xrange(0,len(devData[sent_idx])):
                    tags.append(tag_indexer.get_object(dev_results[sent_idx].label[word_idx]))
        
                devDecoded.append(LabeledSentence(devData[sent_idx].tokens, chunks_from_bio_tag_seq(tags)))
        
            print_evaluation(devData, devDecoded)
  
           
    return dev_exs

  
