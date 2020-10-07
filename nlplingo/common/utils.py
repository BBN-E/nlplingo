from __future__ import absolute_import
from __future__ import division

int_type = 'int32'
float_type = 'float32'

DEPREL_TO_ID = {'PAD': 0, 'no_label': 1, 'nsubj': 2, 'obj': 3, 'iobj': 4, 'csubj': 5, 'ccomp': 6,
                'xcomp': 7, 'obl': 8, 'vocative': 9, 'expl': 10, 'dislocated': 11, 'advcl': 12,
                'advmod': 13, 'discourse': 14,  'aux': 15, 'cop': 16, 'mark': 17, 'nmod': 18,
                'appos': 19, 'nummod': 20, 'acl': 21, 'amod': 22, 'det': 23, 'clf': 24, 'case': 25,
                'conj': 26, 'cc': 27, 'fixed': 28, 'flat': 29, 'compound': 30, 'list': 31, 'parataxis': 32,
                'orphan': 33, 'goeswith': 34, 'reparandum': 35, 'punct': 36, 'root': 37, 'dep': 38}

# only accept tokens of following parts-of-speech as trigger candidates
# trigger_pos_category = set([u'NOUN', u'VERB', u'ADJ', u'PROPN'])
# trigger_pos_category = set([u'NOUN', u'VERB', u'ADJ'])
trigger_pos_category = set(['NOUN', 'VERB'])


def only1(l):
    """
    Checks if the list 'l' of booleans has one and only one True value
    :param l: list of booleans
    :return: True if list has one and only one True value, False otherwise
    """
    true_found = False
    for v in l:
        if v:
            if true_found:
                return False
            else:
                true_found = True
    return true_found

def split_offsets(line, _len=len):
    """
    Retain character offsets from line, splitting the string line.
    :param line: any string
    :param _len: length function
    :return:
    """
    words = line.split()
    index = line.index
    offsets = []
    append = offsets.append
    running_offset = 0
    for word in words:
        word_offset = index(word, running_offset)
        word_len = _len(word)
        running_offset = word_offset + word_len
        append((word, word_offset, running_offset - 1))
    return offsets

class IntPair(object):
    """A utility class to store a pair of integers
   
    Attributes:
        first: first integer
        second: second integer
    """

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def has_same_boundary(self, other):
        """
        :type other: IntPair
        """
        if self.first == other.first and self.second == other.second:
            return True
        else:
            return False

    def contains(self, other):
        """
        :type other: IntPair
        """
        if self.first <= other.first and other.second <= self.second:
            return True
        else:
            return False

    def has_overlapping_boundary(self, other):
        """
        :type other: IntPair
        """
        if (self.first <= other.first and other.first <= self.second) or \
                (other.first <= self.first and self.first <= other.second) or \
                self.has_same_boundary(other) or self.contains(other) or other.contains(self):
            return True
        else:
            return False

    def to_string(self):
        return '({},{})'.format(self.first, self.second)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.first == other.first and self.second == other.second
        return False

    def __ne__(self, other):
        return self.first != other.first or self.second != other.second

    def __hash__(self):
        return hash((self.first, self.second))


class Struct:
    """A structure that can have any fields defined

    Example usage:
    options = Struct(answer=42, lineline=80, font='courier')
    options.answer (prints out 42)
    # adding more
    options.cat = 'dog'
    options.cat (prints out 'dog')
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)


class F1Score(object):
    def __init__(self, c, num_true, num_predict, class_label='class_label'):
        self.c = c
        self.num_true = num_true
        self.num_predict = num_predict
        self.class_label = class_label
        self.calculate_score()

    def calculate_score(self):
        if self.c > 0 and self.num_true > 0:
            self.recall = float(self.c) / self.num_true
        else:
            self.recall = 0

        if self.c > 0 and self.num_predict > 0:
            self.precision = float(self.c) / self.num_predict
        else:
            self.precision = 0

        if self.recall > 0 and self.precision > 0:
            self.f1 = (2 * self.recall * self.precision) / (self.recall + self.precision)
        else:
            self.f1 = 0

    def to_string(self):
        return '%s #C=%d,#R=%d,#P=%d R,P,F=%.2f,%.2f,%.6f' % (self.class_label, self.c, self.num_true, self.num_predict, self.recall, self.precision, self.f1)


