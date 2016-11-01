from __future__ import absolute_import

import os
import re

import time
import numpy as np
from progress.bar import Bar
from nltk.translate import bleu_score

import skipthoughts

# from pprint import pprint

def load_data(data_dir, task_story):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    story_dic = ['HanselAndGretel',
        'SnowWhiteAndTheSevenDwarves',
        'TheLittleMermaid',
        'TheThreeLittlePigs',
        'TheWolfAndTheSevenSheep']

    assert task_story in story_dic

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = '{}_dataset_'.format(task_story)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file)
    test_data = get_stories(test_file)
    return train_data, test_data


def load_queries(data_dir, task_story):
    story_dic = ['HanselAndGretel',
        'SnowWhiteAndTheSevenDwarves',
        'TheLittleMermaid',
        'TheThreeLittlePigs',
        'TheWolfAndTheSevenSheep']

    assert task_story in story_dic

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = '{}_dataset_'.format(task_story)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_queries(train_file)
    test_data = get_queries(test_file)
    return train_data, test_data


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a = line.split('\t')
            q = tokenize(q)
            # a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            # Provide all the substories
            substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def parse_queries(lines):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a = line.split('\t')

            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            # Provide all the substories
            substory = [x for x in story if x]

            data.append(q)
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def get_stories(f):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines())


def get_queries(f):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_queries(f.readlines())


def load_sent2vec_model():
    
    return skipthoughts.load_model()


def vectorize_data(data, word_idx, sent2vec_model, sentence_size, memory_size, ans_sentence_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    A_org = []

    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        S.append(ss)
        Q.append(q)
        A_org.append(answer)

    answers = [a for _, _, a in data]
    A = skipthoughts.encode(sent2vec_model, answers, verbose=False)

    return np.array(S), np.array(Q), A, np.array(A_org)


def match_label(dict, sources, references):
    answers = []

    for i in range(len(sources)):
        diff = np.square(np.subtract(references, sources[i]))
        errors = np.sum(diff, axis=1)
        ans_idx = np.argmin(errors)
        
        for answer, vec in dict.iteritems():
            if np.array_equal(vec, references[ans_idx]):
                answers.append(answer)
                break;

        if i != len(answers) - 1:
            answers.append('')

    return answers


def calculate_bleu(sources, references):
    assert len(sources) == len(references)

    # src_token = [[tokenize(src)] for src in sources]
    # ref_token = [tokenize(ref) for ref in references]
    src_token = [[src] for src in sources]
    ref_token = references

    # print('src_token : ', max(map(len, src_token)))
    # print(np.array(src_token).shape)
    # print(src_token)
    # print("ref_token : ", max(map(len, ref_token)))
    # print(np.array(ref_token).shape)
    # print(ref_token)

    return bleu_score.corpus_bleu(src_token, ref_token)


def save_result(results, directory='./results/'):
    now = time.time()
    filepath = directory + str(now) + '.txt'

    with open(filepath, 'w') as thefile:
        for result in results:
            thefile.write('%s\n' % result)


class ProgressBar(Bar):
    message = 'Loading'
    fill = '#'
    suffix = '%(percent).1f%% | ETA: %(eta)ds'