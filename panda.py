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
from xlsxexporter import XlsxExporter
import panda_utils
from panda_utils import ProgressBar

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 40, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 2000, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 100, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_integer("memn2n_vector_size", 1000, "sentence vector size")
tf.flags.DEFINE_integer("loss_norm", 2, "normalization of loss function (only 1 or 2)")
tf.flags.DEFINE_string("data_dir", "data/pinkData/dataset_new/", "Directory containing Panda tasks")
tf.flags.DEFINE_string("summary_dir", "results/summary/", "Directory containing summary")
# tf.flags.DEFINE_string("task_story", "TheWolfAndTheSevenSheep", "Panda task story name")
tf.flags.DEFINE_boolean("use_testset", False, "Using some part of data as test set")
FLAGS = tf.flags.FLAGS

story_dic = ['HanselAndGretel',
    'SnowWhiteAndTheSevenDwarves',
    'TheLittleMermaid',
    'TheThreeLittlePigs',
    'TheWolfAndTheSevenSheep']

# train/validation/test sets
sent2vec_model = panda_utils.load_sent2vec_model()

for story in story_dic:

    print("Started Task:", story)
    # task data
    if FLAGS.use_testset:
        train, test = panda_utils.load_data(FLAGS.data_dir, story)
        Q_org, testQ_org = panda_utils.load_queries(FLAGS.data_dir, story)
        data = train + test
    else:
        train, val = panda_utils.load_data(FLAGS.data_dir, story)
        trainQ_org, valQ_org = panda_utils.load_queries(FLAGS.data_dir, story)
        data = train + val

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q) for s, q, _ in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    ans_sentence_size = max(map(len, [panda_utils.tokenize(a) for _, _, a in data]))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = min(FLAGS.memory_size, max_story_size)
    vocab_size = len(word_idx) + 1                  # +1 for nil word
    sentence_size = max(query_size, sentence_size)  # for the position

    print("Longest sentence length", sentence_size)
    print("Longest story length", max_story_size)
    print("Average story length", mean_story_size)

    if(FLAGS.use_testset):
        S, Q, A, A_org = panda_utils.vectorize_data(train, word_idx, sent2vec_model, sentence_size, memory_size, ans_sentence_size)
        trainS, valS, trainQ, valQ, trainQ_org, valQ_org, trainA, valA, trainA_org, valA_org = cross_validation.train_test_split(
            S, Q, Q_org, A, A_org, test_size=.1, random_state=FLAGS.random_state)
        testS, testQ, testQ_org, testA, testA_org = panda_utils.vectorize_data(test, word_idx, sent2vec_model, sentence_size, memory_size, ans_sentence_size)
    else:
        trainS, trainQ, trainA, trainA_org = panda_utils.vectorize_data(train, word_idx, sent2vec_model, sentence_size, memory_size, ans_sentence_size)
        valS, valQ, valA, valA_org = panda_utils.vectorize_data(val, word_idx, sent2vec_model, sentence_size, memory_size, ans_sentence_size)

    ans_vec_size = trainA.shape[1]

    if(FLAGS.use_testset):
        allA = np.append(A, testA, axis=0)
        allA_org = np.append(A_org, testA_org, axis=0)
    else:
        allA = np.append(trainA, valA, axis=0)
        allA_org = np.append(trainA_org, valA_org, axis=0)

    sent_vec = dict((a, v) for a, v in zip(allA_org, allA))

    print(trainS[0])

    print("Training Set Shape", trainS.shape)

    # params
    n_total = len(data)
    n_train = trainS.shape[0]
    n_val = valS.shape[0]

    if(FLAGS.use_testset):
        n_test = testS.shape[0]

    print("Training Size", n_train)
    print("Validation Size", n_val)
    if(FLAGS.use_testset):
        print("Testing Size", n_test)
    print("Training Answer Size", trainA.shape)
    print("Validation Answer Size", valA.shape)
    if(FLAGS.use_testset):
        print("Test Answer Size", testA.shape)
    print("Answer Vector Size", ans_vec_size)
    print("Learning Rate", FLAGS.learning_rate)

    tf.set_random_seed(FLAGS.random_state)
    batch_size = FLAGS.batch_size
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)

    batches = zip(range(0, n_total-batch_size+1, batch_size), range(batch_size, n_total+1, batch_size))
    batches = [(start, end) for start, end in batches]
    train_batches = zip(range(0, n_train-batch_size+1, batch_size), range(batch_size, n_train+1, batch_size))
    train_batches = [(start, end) for start, end in train_batches]
    val_batches = zip(range(0, n_val-batch_size+1, batch_size), range(batch_size, n_val+1, batch_size))
    val_batches = [(start, end) for start, end in val_batches]
    if(FLAGS.use_testset):
        test_batches = zip(range(0, n_test-batch_size+1, batch_size), range(batch_size, n_test+1, batch_size))
        test_batches = [(start, end) for start, end in test_batches]

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            model = MemN2N(batch_size, vocab_size, ans_vec_size, sentence_size, memory_size, FLAGS.embedding_size, 
                            FLAGS.memn2n_vector_size, FLAGS.loss_norm,
                            session=sess, hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm, optimizer=optimizer)
            saver = XlsxExporter(story)

            header = {}
            header['Story'] = story
            header['Learning Rate'] = str(FLAGS.learning_rate)
            header['Training Size'] = str(n_train)
            header['Validation Size'] = str(n_val)
            header['Sentence Size'] = str(sentence_size)
            header['Answer Vector Size'] = str(ans_vec_size)
            header['MemN2N Vector Size'] = str(FLAGS.memn2n_vector_size)
            saver.set_header(header)

            bar = ProgressBar('Train : ' + story, max=FLAGS.epochs)

            for t in range(1, FLAGS.epochs+1):
                bar.next()
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
                    for start in range(0, n_train-batch_size+1, batch_size):
                        end = start + batch_size
                        s = trainS[start:end]
                        q = trainQ[start:end]
                        pred = model.predict_proba(s, q)
                        train_preds_vec += list(pred)

                    train_preds = panda_utils.match_label(sent_vec, train_preds_vec, trainA)
                    train_acc = panda_utils.calculate_bleu(train_preds, trainA_org[:end])
                    
                    val_preds_vec = []
                    for start, end in val_batches:
                        s = valS[start:end]
                        q = valQ[start:end]
                        preds = model.predict_proba(s, q)
                        val_preds_vec += list(preds)

                    val_preds = panda_utils.match_label(sent_vec, val_preds_vec, valA)
                    val_acc = panda_utils.calculate_bleu(val_preds, valA_org[:end])
                    # train_acc = metrics.accuracy_score(np.array(train_preds), np.array(train_labels))
                    # val_acc = metrics.accuracy_score(val_preds, val_labels)

                    result = {}
                    result['Epoch'] = t
                    result['Total Cost'] = total_cost
                    result['Training Accuracy'] = train_acc
                    result['Validation Accuracy'] = val_acc
                    result['Duration Time'] = duration
                    saver.add_result(result)

                    # print('-----------------------')
                    # print('Epoch', t)
                    # print('Total Cost:', total_cost)
                    # print('Training Accuracy:', train_acc)
                    # print('Validation Accuracy:', val_acc)
                    # print('Duration Time: %.3f sec' % (duration))
                    # print('-----------------------')

            bar.finish()

            if FLAGS.use_testset:
                test_preds_vec = []
                for start, end in test_batches:
                    s = testS[start:end]
                    q = testQ[start:end]
                    pred = model.predict_proba(s, q)
                    test_preds_vec += list(pred)

                test_preds = panda_utils.match_label(sent_vec, test_preds_vec, testA)
                test_acc = panda_utils.calculate_bleu(test_preds, testA_org[:end])
                # test_acc = metrics.accuracy_score(test_preds, test_labels)
                saver.set_accuracy(test_acc)

                print("Testing Accuracy:", test_acc)
                

            # panda_utils.save_result(results)
            saver.set_answers(valQ_org, valA_org, val_preds)
            saver.make_file()