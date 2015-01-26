import email
import math
import os


def load_tokens(email_path):
    tokens = []
    file = open(email_path)
    message = email.message_from_file(file)
    for line in email.iterators.body_line_iterator(message):
        to_add = line.split()
        if "Subject:" in to_add:
            for i in range(len(to_add)):
                to_add[i] = "Subject*" + to_add[i]
        if "To:" in to_add:
            for i in range(len(to_add)):
                to_add[i] = "To*" + to_add[i]
        if "From:" in to_add:
            for i in range(len(to_add)):
                to_add[i] = "From*" + to_add[i]
        if "Return-Path:" in to_add:
            for i in range(len(to_add)):
                to_add[i] = "Return-Path*" + to_add[i]
        tokens = tokens + to_add
    file.close()
    return tokens


def log_probs(email_paths, smoothing):
    tokens = []
    counts = {}
    vocabulary = set()
    probs = {}
    for path in email_paths:
        tokens = tokens + load_tokens(path)
    for token in tokens:
        vocabulary.add(token)
        value = counts.get(token)
        if value is not None:
            counts.update({token: value + 1})
        else:
            counts.update({token: 1})
    bottom = len(tokens) + smoothing*(len(vocabulary) + 1)
    unkp = smoothing/bottom
    unkp = math.log(unkp)
    probs.update({"<UNK>": unkp})
    for word in vocabulary:
        p = 0
        top = counts.get(word) + smoothing
        p = top/bottom
        p = math.log(p)
        probs.update({word: p})
    return probs


class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir):
        smoothing = 1e-5
        spam = []
        ham = []
        for f in os.listdir(spam_dir):
            doc = spam_dir + "/" + f
            spam.append(doc)
        for f in os.listdir(ham_dir):
            doc = ham_dir + "/" + f
            ham.append(doc)

        self.ham_dict = log_probs(ham, smoothing)
        self.spam_dict = log_probs(spam, smoothing)
        spam_count = len(spam)
        ham_count = len(ham)

        self.p_spam = math.log(spam_count/float(spam_count + ham_count))
        self.p_not_spam = math.log(ham_count/float(spam_count + ham_count))
    
    def is_spam(self, email_path):
        words = load_tokens(email_path)
        p_s = 0
        p_ns = 0
        for word in words:
            value = self.ham_dict.get(word)
            if value is not None:
                p_ns = p_ns+value
            else:
                value = self.ham_dict.get("<UNK>")
                p_ns = p_ns+value
        p_ns = p_ns + self.p_not_spam
        for word in words:
            value = self.spam_dict.get(word)
            if value is not None:
                p_s = p_s+value
            else:
                value = self.ham_dict.get("<UNK>")
                p_s = p_s+value
            if len(word) > 20:
                p_s = p_s - math.log(0.9)
        p_s = p_s + self.p_spam
        return p_s > p_ns

    def indicative_spam(self, word):
        value = math.log(0.5)
        if word in self.spam_dict and self.ham_dict.get(word) is not None:
            p_ws = math.exp(self.spam_dict.get(word))
            p_wns = math.exp(self.ham_dict.get(word))
            p_w = p_ws*math.exp(self.p_spam) + p_wns*math.exp(self.p_not_spam)
            value = math.log(p_ws/p_w)
        return (value)

    def indicative_ham(self, word):
        value = math.log(0.5)
        if word in self.spam_dict and self.ham_dict.get(word) is not None:
            p_ws = math.exp(self.spam_dict.get(word))
            p_wns = math.exp(self.ham_dict.get(word))
            p_w = p_ws*math.exp(self.p_spam) + p_wns*math.exp(self.p_not_spam)
            value = math.log(p_wns/p_w)
        return (value)
