# models.py

import tensorflow as tf
import numpy as np
import random
from utils import *
from models import *
from nerdata import *

'''
def decode_transition(pred_this_instance, sentence, tag_indexer):

    pred_tags = []
    if len(sentence) > 0:

        scorer_mat = np.zeros((len(tag_indexer),len(sentence)), dtype = float)
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
'''

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

def pad_to_length(np_arr, length):
    result = np.zeros(length)
    
    result[0:np_arr.shape[0]] = np_arr
    return result
    
def train_fancy(train_exs, dev_exs, devData, tag_indexer, word_vectors):


    pos_indexer = Indexer()
    for sent_idx in range(len(train_exs)):
        for pos in train_exs[sent_idx].pos:
            pos_indexer.get_index(pos)

    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 250
    
    wordvectors = np.array(word_vectors.vectors, dtype="float32")
    #print wordvectors[0]
    
    #train_xs = np.array([[0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1]])
    #train_ys = np.array([0, 1, 1, 1, 1, 0])
    
    train_xs = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    dev_xs = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])

    train_xs_pos = [ex.pos for ex in train_exs]
    dev_xs_pos = [ex.pos for ex in dev_exs]

    #test_xs = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    #test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    train_ys =  np.array([pad_to_length(np.array(ex.label), seq_max_len) for ex in train_exs])
    dev_ys = np.array([pad_to_length(np.array(ex.label), seq_max_len) for ex in dev_exs])
    
    pos_size = len(pos_indexer)
    feat_vec_size = 300 + pos_size
    embedding_size1 = 300
    embedding_size2 = 300
    cell_size=50
    fclayer = 100
    num_classes = len(tag_indexer)

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    indices = tf.placeholder(tf.int32)
    fx1_words = tf.nn.embedding_lookup(wordvectors,indices)
    fx1_words = tf.reshape(fx1_words,[1,seq_max_len,300])
    #fx2 = tf.Variable(tf.zeros([1, 200, 300]),dtype=tf.float32)

    '''
    filters = tf.get_variable("filters", [3, 300, 300], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    fx2 = tf.nn.conv1d(fx1,filters,stride=1, padding = 'SAME')
    '''

    fx0 = tf.placeholder(tf.float32, [seq_max_len, pos_size])
    V = tf.get_variable("V", [pos_size, pos_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    fx1_pos = tf.nn.relu(tf.tensordot(fx0, V, 1))
    fx1_pos = tf.reshape(fx1_pos,[1,seq_max_len,pos_size])

    fx1 = tf.concat([fx1_words, fx1_pos], axis = 2)

    x_lengths =  tf.placeholder(tf.int32)
    #fx2 = tf.reduce_mean(tf.nn.embedding_lookup(word_embeds,indices[:x_lengths[0]]),0)
    #x_lengths = tf.reshape(x_lengths,[1])
    #fx = tf.reduce_mean(tf.nn.embedding_lookup(word_embeds,indices),0)
    keep_prob1 = tf.placeholder(tf.float32)  # DROP-OUT here
    keep_prob2 = tf.placeholder(tf.float32)
    
    ill_trans = get_matrix_illegal(tag_indexer)
    transitions = tf.Variable(ill_trans, name="transitions", dtype=tf.float32)

    cell1 = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, state_is_tuple=True)
    cell2 = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, state_is_tuple=True)
    #cell3 = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, use_peepholes=True,state_is_tuple=True)
    stacked = tf.contrib.rnn.MultiRNNCell([cell1,cell2])
    outputs, last_states = tf.nn.dynamic_rnn(cell1,fx1_words, dtype=tf.float32,sequence_length=x_lengths)
    
    label_orig = tf.placeholder(tf.int32)
    W = tf.get_variable("W", [ cell_size, num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    #b = tf.get_variable("b", [num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))



    """
    lstm_outputs = tf.reshape(outputs[0][:x_lengths[0]], [-1, cell_size])
    logits = tf.matmul(lstm_outputs, W)# + b
    softmax = tf.nn.softmax(logits)
    preds = tf.argmax(softmax, 1)
    """

    
    #lstm_outputs = tf.reshape(outputs, [1,200, cell_size])
    label = tf.reshape(label_orig,[1,seq_max_len])
    outputpred = tf.reshape(outputs[0], [-1, cell_size])
    logits =tf.matmul(outputpred, W)# + b
    preds = logits[:x_lengths[0]]
    
    #print transitions.shape
  
    loss,_ = tf.contrib.crf.crf_log_likelihood(tf.reshape(logits,[1,seq_max_len,num_classes]),label,x_lengths,transitions)
    loss = tf.negative(tf.reduce_mean(loss))
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label[:x_lengths[0]],logits=logits))
    
    

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
                pos_table = np.zeros((seq_max_len, pos_size))
                for pos in train_xs_pos[ex_idx]:
                    pos_curr = pos_indexer.get_index(pos)
                    pos_table[i, pos_curr] = 1

                [_, loss_this_instance, summary] = sess.run([ train_op, loss, merged], feed_dict = {indices: train_xs[ex_idx].astype(int),
                                                                                  label: np.reshape(train_ys[ex_idx],[1,seq_max_len]), fx0:pos_table ,keep_prob1:0.8,keep_prob2:0.5,x_lengths:[train_seq_lens[ex_idx].astype(int)]})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_instance
                
                #print loss_this_instance
                #if math.isnan(loss_this_instance):
                #print str(ex_idx) + " "+str(p1)

            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
            for ex_idx in xrange(0, len(dev_xs)):
                pos_table = np.zeros((seq_max_len, pos_size))
                for pos in dev_xs_pos[ex_idx]:
                    pos_curr = pos_indexer.get_index(pos)
                    pos_table[i, pos_curr] = 1
            
                [pred_this_instance,trans] = sess.run([preds,transitions],feed_dict={indices: dev_xs[ex_idx].astype(int), fx0:pos_table, keep_prob1:1,keep_prob2:1,x_lengths:[dev_seq_lens[ex_idx].astype(int)]})
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

def create_graph(indices, fx1_words, fx0,  x_lengths, keep_prob1, keep_prob2, label, tag_indexer, pos_indexer, seq_max_len, ident):
    pos_size = len(pos_indexer)
    feat_vec_size = 300 + pos_size
    embedding_size1 = 300
    embedding_size2 = 300
    cell_size=50
    fclayer = 100
    num_classes = len(tag_indexer)


    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    with tf.variable_scope(ident):
        
        #fx2 = tf.Variable(tf.zeros([1, 200, 300]),dtype=tf.float32)

        '''
        filters = tf.get_variable("filters", [3, 300, 300], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        fx2 = tf.nn.conv1d(fx1,filters,stride=1, padding = 'SAME')
        '''

        
        V = tf.get_variable("V", [pos_size, pos_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        fx1_pos = tf.nn.relu(tf.tensordot(fx0, V, 1))
        fx1_pos = tf.reshape(fx1_pos,[1,seq_max_len,pos_size])

        fx1 = tf.concat([fx1_words, fx1_pos], axis = 2)

        
        
        ill_trans = get_matrix_illegal(tag_indexer)
        transitions = tf.Variable(ill_trans, name="transitions", dtype=tf.float32)

        cell1 = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, state_is_tuple=True)
        cell2 = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, state_is_tuple=True)
        #cell3 = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, use_peepholes=True,state_is_tuple=True)
        stacked = tf.contrib.rnn.MultiRNNCell([cell1,cell2])
        outputs, last_states = tf.nn.dynamic_rnn(cell1,fx1_words, dtype=tf.float32,sequence_length=x_lengths)
        
        
        W = tf.get_variable("W", [ cell_size, num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        #b = tf.get_variable("b", [num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))



        """
        lstm_outputs = tf.reshape(outputs[0][:x_lengths[0]], [-1, cell_size])
        logits = tf.matmul(lstm_outputs, W)# + b
        softmax = tf.nn.softmax(logits)
        preds = tf.argmax(softmax, 1)
        """

        
        #lstm_outputs = tf.reshape(outputs, [1,200, cell_size])
        
        outputpred = tf.reshape(outputs[0], [-1, cell_size])
        logits =tf.matmul(outputpred, W)# + b
        preds = logits[:x_lengths[0]]

        return transitions, logits, preds    
 
def train_fancy_da(train_exs, dev_exs, devData, tag_indexer, word_vectors):


    pos_indexer = Indexer()
    for sent_idx in range(len(train_exs)):
        for pos in train_exs[sent_idx].pos:
            pos_indexer.get_index(pos)

    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 250
    
    wordvectors = np.array(word_vectors.vectors, dtype="float32")
    
    train_xs = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    dev_xs = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])

    train_xs_pos = [ex.pos for ex in train_exs]
    dev_xs_pos = [ex.pos for ex in dev_exs]

    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])

    train_ys =  np.array([pad_to_length(np.array(ex.label), seq_max_len) for ex in train_exs])
    dev_ys = np.array([pad_to_length(np.array(ex.label), seq_max_len) for ex in dev_exs])

    pos_size = len(pos_indexer)
    feat_vec_size = 300 + pos_size
    embedding_size1 = 300
    embedding_size2 = 300
    cell_size=50
    fclayer = 100
    num_classes = len(tag_indexer)

    indices = tf.placeholder(tf.int32)
    fx1_words = tf.nn.embedding_lookup(wordvectors,indices)
    fx1_words = tf.reshape(fx1_words,[1,seq_max_len,300])
    fx0 = tf.placeholder(tf.float32, [seq_max_len, pos_size])

    x_lengths =  tf.placeholder(tf.int32)
    #fx2 = tf.reduce_mean(tf.nn.embedding_lookup(word_embeds,indices[:x_lengths[0]]),0)
    #x_lengths = tf.reshape(x_lengths,[1])
    #fx = tf.reduce_mean(tf.nn.embedding_lookup(word_embeds,indices),0)
    keep_prob1 = tf.placeholder(tf.float32)  # DROP-OUT here
    keep_prob2 = tf.placeholder(tf.float32)

    label_orig = tf.placeholder(tf.int32)
    label = tf.reshape(label_orig,[1,seq_max_len])
    
    transitions_a, logits_a, preds_a = create_graph(indices, fx1_words, fx0, x_lengths, keep_prob1, keep_prob2, label,  tag_indexer, pos_indexer, seq_max_len, "a")
    
    transitions_b, logits_b, preds_b = create_graph(indices,fx1_words, fx0,x_lengths, keep_prob1, keep_prob2, label, tag_indexer, pos_indexer, seq_max_len, "b")

    transitions_c, logits_c, preds_c = create_graph(indices,fx1_words, fx0,x_lengths, keep_prob1, keep_prob2, label, tag_indexer, pos_indexer,seq_max_len, "c")

    #print transitions.shape
    indomain = tf.placeholder(tf.int32)


    if indomain == 0:
        logits = tf.reduce_mean([logits_a, logits_c], axis = 0)
        transitions = tf.reduce_mean([transitions_a, transitions_c], axis = 0)
        preds = logits[:x_lengths[0]]
    else:
        logits = tf.reduce_mean([logits_b, logits_c], axis = 0)
        transitions = tf.reduce_mean([transitions_b, transitions_c], axis = 0)
        preds = logits[:x_lengths[0]]

    loss,_ = tf.contrib.crf.crf_log_likelihood(tf.reshape(logits,[1,seq_max_len,num_classes]),label,x_lengths,transitions)
    loss = tf.negative(tf.reduce_mean(loss))
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label[:x_lengths[0]],logits=logits))
    
    

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
                pos_table = np.zeros((seq_max_len, pos_size))
                for pos in train_xs_pos[ex_idx]:
                    pos_curr = pos_indexer.get_index(pos)
                    pos_table[i, pos_curr] = 1

                indomain_flag = 0
                if train_exs[ex_idx].domain == 1:
                    indomain_flag = 1

                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {indices: train_xs[ex_idx].astype(int), indomain: indomain_flag,
                                                                                  label: np.reshape(train_ys[ex_idx],[1,seq_max_len]), fx0:pos_table ,keep_prob1:0.8,keep_prob2:0.5,x_lengths:[train_seq_lens[ex_idx].astype(int)]})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_instance
                
                #print loss_this_instance
                #if math.isnan(loss_this_instance):
                #print str(ex_idx) + " "+str(p1)

            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
            for ex_idx in xrange(0, len(dev_xs)):
                pos_table = np.zeros((seq_max_len, pos_size))
                for pos in dev_xs_pos[ex_idx]:
                    pos_curr = pos_indexer.get_index(pos)
                    pos_table[i, pos_curr] = 1
            
                [pred_this_instance,trans] = sess.run([preds,transitions],feed_dict={indices: dev_xs[ex_idx].astype(int), fx0:pos_table, indomain: 0, keep_prob1:1,keep_prob2:1,x_lengths:[dev_seq_lens[ex_idx].astype(int)]})
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