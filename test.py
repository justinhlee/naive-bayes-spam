from spamfilter import *
import unittest


class is_legal_moves_tests(unittest.TestCase):

    def testOne(self):
        # ham_dir = "data/train/ham/"
        # print load_tokens(ham_dir+"ham2")[110:114]
        # spam_dir = "data/train/spam/"
        # print load_tokens(spam_dir+"spam1")[1:5]
        # print load_tokens(spam_dir+"spam2")[:4]
        paths = ["data/train/ham/ham%d" % i for i in range(1, 11)]
        paths = ["data/train/spam/spam%d" %i for i in range(1, 11)]

        # p = log_probs(paths, 1e-5)
        # print p["<UNK>"]

        sf = SpamFilter("data/train/spam", "data/train/ham")
        dev = "data/dev/ham"
        dev2 = "data/dev/spam"
        count = 0
        errors = 0
        for f in os.listdir(dev):
            doc = dev + "/" + f
            result = sf.is_spam(doc)
            count = count + 1
            if result is True:
                print f
                errors = errors + 1
        print (count - errors)/float(count)

        count = 0
        errors = 0
        
        for f in os.listdir(dev2):
            doc = dev2 + "/" + f
            count = count + 1

            if not sf.is_spam(doc):
                print f
                errors = errors + 1
        print (count - errors)/float(count)
        
        
if __name__ == '__main__':
    unittest.main()

