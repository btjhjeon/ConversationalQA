"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from sklearn import cross_validation, metrics
from itertools import chain
from six.moves import range, reduce
from pprint import pprint
import time

import tensorflow as tf
import numpy as np

from memn2n import MemN2N
import panda_utils

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 40, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 100, "Maximum size of memory.")
# tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_integer("word_vector_size", 50, "glove embeding size (50, 100, 200, 300 only)")
tf.flags.DEFINE_integer("sentence_vector_size", 128, "sentence vector size")
tf.flags.DEFINE_integer("rnn_neuron_size", 50, "GRU embeding size")
tf.flags.DEFINE_integer("rnn_layer_size", 3, "GRU layer size")
# tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_string("data_dir", "data/pinkData/dataset_new/", "Directory containing Panda tasks")
tf.flags.DEFINE_string("summary_dir", "result/summary/", "Directory containing summary")
tf.flags.DEFINE_string("task_story", "HanselAndGretel", "Panda task story name")
# tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS

print("Started Task:", FLAGS.task_story)

# task data
train, test = panda_utils.load_data(FLAGS.data_dir, FLAGS.task_story)
data = train + test
# print("print data (")
# pprint(data)
# print(")")

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q) for s, q, _ in data)))
# print("print vocab (")
# pprint(vocab)
# print(")")
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
# print("print word_idx (")
# pprint(word_idx)
# print(")")
glove = panda_utils.load_glove(FLAGS.word_vector_size)
# print("print glove (")
# pprint(len(glove))
# print(")")

max_story_size = max(map(len, (s for s, _, _ in data)))
print("<max_story_size> : ", max_story_size)
mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
print("<mean_story_size> : ", mean_story_size)
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
print("<story_sentence_size> : ", sentence_size)
ans_sentence_size = max(map(len, [panda_utils.tokenize(a) for _, _, a in data]))
print("<ans_sentence_size> : ", ans_sentence_size)
query_size = max(map(len, (q for _, q, _ in data)))
print("<query_size> : ", query_size)
memory_size = min(FLAGS.memory_size, max_story_size)
print("<memory_size> : ", memory_size)
vocab_size = len(word_idx) + 1 # +1 for nil word
print("<vocab_size> : ", vocab_size)
glove_size = len(glove)
print("<glove_size> : ", glove_size)
sentence_size = max(query_size, sentence_size) # for the position
print("<sentence_size> : ", sentence_size)

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
S, Q, A, A_org = panda_utils.vectorize_data(train, word_idx, glove, sentence_size, memory_size, ans_sentence_size, FLAGS.word_vector_size)
trainS, valS, trainQ, valQ, trainA, valA, trainA_org, valA_org = cross_validation.train_test_split(S, Q, A, A_org, test_size=.1, random_state=FLAGS.random_state)
testS, testQ, testA, testA_org = panda_utils.vectorize_data(test, word_idx, glove, sentence_size, memory_size, ans_sentence_size, FLAGS.word_vector_size)

allA = np.append(A, testA, axis=0)

print(testS[0])

print("Training set shape", trainS.shape)

# params
n_total = len(data)
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)
print("Training Answer Size", trainA.shape)
print("Validation Answer Size", valA.shape)
print("Test Answer Size", testA.shape)

# train_labels = np.argmax(trainA, axis=1)
# test_labels = np.argmax(testA, axis=1)
# val_labels = np.argmax(valA, axis=1)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)

batches = zip(range(0, n_total-batch_size, batch_size), range(batch_size, n_total, batch_size))
batches = [(start, end) for start, end in batches]
train_batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
train_batches = [(start, end) for start, end in train_batches]
test_batches = zip(range(0, n_test-batch_size, batch_size), range(batch_size, n_test, batch_size))
test_batches = [(start, end) for start, end in test_batches]

with tf.Graph().as_default() as graph:
    with tf.Session() as sess:
        model = MemN2N(batch_size, vocab_size, ans_sentence_size, sentence_size, memory_size, 
                        FLAGS.embedding_size, FLAGS.word_vector_size, FLAGS.rnn_neuron_size, 
                        FLAGS.rnn_layer_size, session=sess, hops=FLAGS.hops, 
                        max_grad_norm=FLAGS.max_grad_norm, optimizer=optimizer)

        rep_labels = []
        for start, end in batches:
            a = allA[start:end]
            label = model.desire(a)
            rep_labels += list(label)

        print("rep_labels")
        print(rep_labels)

        for t in range(1, FLAGS.epochs+1):
            np.random.shuffle(train_batches)
            total_cost = 0.0
            start_time = time.time()

            for start, end in train_batches:
                s = trainS[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]
                cost_t = model.batch_fit(s, q, a)
                total_cost += cost_t

            duration = time.time() - start_time

            if t % FLAGS.evaluation_interval == 0:
                # model.update_summary(t, trainS, trainQ, trainA)

                train_preds_vec = []
                train_labels_vec = []
                for start in range(0, n_train, batch_size):
                    end = start + batch_size
                    s = trainS[start:end]
                    q = trainQ[start:end]
                    a = trainA[start:end]
                    pred = model.predict_proba(s, q)
                    train_preds_vec += list(pred)
                    label = model.desire(a)
                    train_labels_vec += list(label)

                answer2vec = panda_utils.sentence_vec(data, rep_labels)
                train_preds = panda_utils.match_label(answer2vec, train_preds_vec, train_labels_vec)

                val_preds_vec = model.predict_proba(valS, valQ)
                val_labels_vec = model.desire(valA)
                val_preds = panda_utils.match_label(answer2vec, val_preds_vec, val_labels_vec)
                
                train_acc = panda_utils.calculate_bleu(train_preds, trainA_org)
                val_acc = panda_utils.calculate_bleu(val_preds, valA_org)
                # train_acc = metrics.accuracy_score(np.array(train_preds), np.array(train_labels))
                # val_acc = metrics.accuracy_score(val_preds, val_labels)

                print('-----------------------')
                print('Epoch', t)
                print('Total Cost:', total_cost)
                print('Training Accuracy:', train_acc)
                print('Validation Accuracy:', val_acc)
                print('Duration Time: %.3f sec' % (duration))
                print('-----------------------')

                # print('train_preds : ', np.array(train_preds).shape)
                # print(train_preds)

                if t == 200:
                    print('train_preds : ', np.array(train_preds).shape)
                    print(train_preds)
                    # print('train_labels_vec : ', np.array(train_labels_vec).shape)
                    # print(train_labels_vec)

        test_preds_vec = []
        test_labels_vec = []
        for start, end in test_batches:
            s = testS[start:end]
            q = testQ[start:end]
            a = testA[start:end]
            pred = model.predict_proba(s, q)
            test_preds_vec += list(pred)
            label = model.desire(a)
            test_labels_vec += list(label)

        test_preds = panda_utils.match_label(answer2vec, test_preds_vec, test_labels_vec)
        test_acc = panda_utils.calculate_bleu(test_preds, testA_org)
        # test_acc = metrics.accuracy_score(test_preds, test_labels)
        print("Testing Accuracy:", test_acc)
