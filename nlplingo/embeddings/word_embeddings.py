from __future__ import absolute_import
from __future__ import with_statement
import six

import os
import codecs
import logging

import numpy as np

import abc

logger = logging.getLogger(__name__)


@six.add_metaclass(abc.ABCMeta)
class WordEmbeddingAbstract(object):
    @abc.abstractmethod
    def load_word_vec_file(self, embedding_filepath, vocab_size, vector_size):
        return None

    @abc.abstractmethod
    def get_none_vector(self):
        pass

    @abc.abstractmethod
    def get_missing_vector(self):
        pass

    @abc.abstractmethod
    def to_token_text(self, text, embedding_prefix):
        pass

    @abc.abstractmethod
    def to_lookup_id(self, sent, token):
        pass

    @abc.abstractmethod
    def to_lookup_id_sent(self, sent):
        pass

    @abc.abstractmethod
    def get_vector(self, token, embedding_prefix=None, try_lower=True, try_lemma=True, sent=None):
        pass

    @abc.abstractmethod
    def get_sent_vector(self, sent):
        pass

    @abc.abstractmethod
    def get_vector_by_index(self, idx):
        pass


class WordEmbedding(WordEmbeddingAbstract):

    def __init__(self, embedding_filepath, vocab_size, vector_size, none_token, missing_token):
        """
        :type params: nlplingo.common.parameters.Parameters
        """

        print('Loading embeddings file ', embedding_filepath)
        words, word_vec = self.load_word_vec_file(embedding_filepath, vocab_size, vector_size)
        self.words = words
        """:type: numpy.ndarray"""
        self.word_vec = word_vec
        """:type: numpy.ndarray"""

        self.word_lookup = dict((w, i) for i, w in enumerate(self.words))
        self.vector_length = vector_size
        self.base_size = len(self.words)
        #self.word_vecLength = None
        #self.token_map = dict()
        #self.accessed_indices = set()

        self.none_token = none_token
        text, self.none_index, vec = self.get_vector(self.none_token)
        assert self.none_index != -1

        self.missing_token = missing_token
        text, self.missing_index, vec = self.get_vector(self.missing_token)
        assert self.missing_index != -1

    class Factory(object):
        def create(self, embeddings_params):
            return WordEmbedding(
                embeddings_params['embedding_file'],
                embeddings_params['vocab_size'],
                embeddings_params['vector_size'],
                embeddings_params['none_token'],
                embeddings_params['missing_token']
            )

    def load_word_vec_file(self, embedding_filepath, vocab_size, vector_size):
        npz_file = embedding_filepath + '.npz'
        if os.path.exists(npz_file):
            data = np.load(npz_file)
            return (data['words'], data['word_vec'])

        words = []
        word_vec = np.empty((vocab_size, vector_size), dtype=np.float32)

        with codecs.open(embedding_filepath, 'r', encoding='utf8') as f:
            i = 0
            for line in f:
                fields = line.rstrip().split(' ')
                if len(fields) == 2:  # header in embeddings file
                    continue
                word = fields[0]
                words.append(word)
                word_vec[i, :] = np.asarray([float(x) for x in fields[1:]])
                i += 1

        words = np.asarray(words)
        np.savez(npz_file, words=words, word_vec=word_vec)
        return (words, word_vec)

    def get_none_vector(self):
        return self.get_vector(self.none_token)

    def get_missing_vector(self):
        return self.get_vector(self.missing_token)

    def to_token_text(self, text, embedding_prefix):
        if text is None:
            return None
        if embedding_prefix is not None:
            text = embedding_prefix + text
        return text

    def get_vector(self, token, embedding_prefix=None, try_lower=True, try_lemma=True, sent=None):
        """
        Returns:
            (str, int, np.array[float])
        :param token: could be a string or nlplingo.text.text_span.Token
        :param sent: Fitting the interface, not used
        :return: str==(original token string, lowercase, or lemma), int==embedding-index, np.array=embedding-vector
        """
        idx = -1
        vec = None
        if isinstance(token, six.string_types) or type(token) is np.unicode_:
            text = self.to_token_text(token, embedding_prefix)

            if text in self.word_lookup:
                idx = self.word_lookup[text]
                vec = self.word_vec[idx]
            if idx < 0:
                text = None
        else:
            text = self.to_token_text(token.text, embedding_prefix)
            if text in self.word_lookup:
                idx = self.word_lookup[text]
                vec = self.word_vec[idx]
            if idx < 0:
                text = self.to_token_text(token.text.lower(), embedding_prefix)
                if text in self.word_lookup:
                    idx = self.word_lookup[text]
                    vec = self.word_vec[idx]
            if idx < 0:
                text = self.to_token_text(token.lemma, embedding_prefix)
                if text in self.word_lookup:
                    idx = self.word_lookup[text]
                    vec = self.word_vec[idx]
            if idx < 0:
                text = None
        #if idx > -1:
        #    self.accessed_indices.add(idx)
        return (text, idx, vec)

    def get_vector_by_index(self, idx):
        if idx < self.base_size:
            return (self.words[idx], idx, self.word_vec[idx])
        else:
            raise ValueError('Given an idx which is out of bounds for embedding vocab')

    def get_sent_vector(self, token, embedding_prefix=None, try_lower=True, try_lemma=True, sent=None):
        return (None, None)

    def to_lookup_id(self, sent, token):
        return None

    def to_lookup_id_sent(self, sent):
        return None


class DocumentContextualEmbeddings(object):
    @classmethod
    def load_embeddings_into_doc(cls, doc, embedding_dict, average_embeddings=False):
        """
        BERT might break up a single word token into multiple word pieces.
        If average_embeddings is False, then for each token, we use the embeddings associated with the first word piece.
        If average_embeddings is True, we average the embeddings over the multiple word pieces, and use that as the
        embedding for the word token.

        :type doc: nlplingo.text.text_theory.Document
        :type embedding_dict: dict   ; Parsed npz file
        """
        doc_embeddings = embedding_dict['embeddings']
        doc_token_map = embedding_dict['token_map']

        try:
            assert (len(doc_embeddings) == len(doc_token_map)
                    == len(doc.sentences))
        except AssertionError:
            logger.error(
                'len(doc_embeddings)={} len(doc_token_map)={} len(doc.sentences)={}'.format(
                    len(doc_embeddings),
                    len(doc_token_map),
                    len(doc.sentences)))
        for i, sentence in enumerate(doc.sentences):
            try:
                assert len(sentence.tokens) == len(doc_token_map[i])
            except AssertionError:
                logger.warning(
                    '#tokens do not match, len(sentence.tokens)={} len(doc_token_map[i])={}\n'
                    'This likely means this particular sentence generated > max_number of BERT tokens.'
                    'We will still grab whatever BERT embeddings we can from this sentence'.format(
                        len(sentence.tokens), len(doc_token_map[i])))

            for j, token in enumerate(sentence.tokens):
                if j < len(doc_token_map[i]):
                    contextual_token_index = doc_token_map[i][j]

                    try:
                        assert contextual_token_index < len(doc_embeddings[i])

                        if average_embeddings:
                            if (j + 1) < len(doc_token_map[i]):
                                next_contextual_token_index = doc_token_map[i][j+1]
                            else:
                                next_contextual_token_index = len(doc_embeddings[i]) - 1    # NOTE: this assume that the very last embedding is a filler

                            info_string = 'At token {} of sentence which has {} tokens, ' \
                                            'len(doc_embeddings[i])={} ' \
                                            'contextual_token_index={}, next_contextual_token_index={}'.format(
                                            str(j), len(sentence.tokens),
                                            str(len(doc_embeddings[i])),
                                            str(contextual_token_index), str(next_contextual_token_index))

                            if next_contextual_token_index <= contextual_token_index:
                                logger.warning(info_string)
                                next_contextual_token_index = contextual_token_index + 1

                            values = list()
                            for index in range(contextual_token_index, next_contextual_token_index):
                                values.append(doc_embeddings[i][index])
                            token.word_vector = np.array(values).mean(axis=0)
                            token.has_vector = True
                        else:
                            v = doc_embeddings[i][contextual_token_index]
                            token.word_vector = v
                            token.has_vector = True
                    except AssertionError:
                        logger.warning('OUT-OF-RANGE: contextual_token_index={}, but len(doc_embeddings[i])={}'.format(
                            contextual_token_index, len(doc_embeddings[i])))



'''
Some process has determined the embeddings for each token in a document. This class matches those tokens to that embedding
'''


class ContextWordEmbedding(WordEmbeddingAbstract):

    def __init__(self, embedding_filepath, vocab_size, vector_size, none_token, missing_token):
        """
        :type params: nlplingo.common.parameters.Parameters
        """

        print('Loading embeddings file ', embedding_filepath)
        lookup_ids, word_vec, lookup_ids_sent, sent_vec = self.load_word_vec_file(embedding_filepath, vocab_size, vector_size)
        self.lookup_ids = lookup_ids
        """:type: numpy.ndarray"""
        self.word_vec = word_vec
        """:type: numpy.ndarray"""
        self.lookup_ids_sent = lookup_ids_sent

        self.word_lookup = dict((w, i) for i, w in enumerate(self.lookup_ids))
        self.vector_length = vector_size
        self.base_size = len(self.lookup_ids)
        self.sent_lookup = dict((w, i) for i, w in enumerate(self.lookup_ids_sent))
        self.sent_vec = sent_vec
        #self.word_vecLength = None
        #self.token_map = dict()
        #self.accessed_indices = set()


        self.none_token = none_token
        text, self.none_index, vec = self.get_none_vector()
        assert self.none_index != -1

        self.missing_token = missing_token
        #text, self.missing_index, vec = self.get_vector(self.missing_token)
        #assert self.missing_index != -1
        self.missing_index = 0

    class Factory(object):
        def create(self, embeddings_params):
            return ContextWordEmbedding(
                embeddings_params['embedding_file'],
                embeddings_params['vocab_size'],
                embeddings_params['vector_size'],
                embeddings_params['none_token'],
                embeddings_params['missing_token']
            )

    def load_word_vec_file(self, embedding_filepath, vocab_size, vector_size):
        # It is assumed that the embedding_filepath is an npz
        if os.path.exists(embedding_filepath):
            data = np.load(embedding_filepath)
        else:
            raise RuntimeError("Couldn't find embedding file {}".format(embedding_filepath))

        lookup_ids = [
            (docid[0], sent_tok_id[0], sent_tok_id[1])
            for docid, sent_tok_id in zip(data['docids'], data['sent_tok_id'])
        ]

        lookup_ids_sent = [
            (docid, sent_id)
            for docid, sent_id in zip(data['sent_docs'], data['sent_ids'])
        ]
        return (
            lookup_ids,
            data['word_vec'].astype(np.float32),
            lookup_ids_sent,
            data['sent_vec'].astype(np.float32)
        )

    def get_none_vector(self):
        return self.get_vector(self.none_token)

    def get_missing_vector(self):
        return self.get_vector(self.missing_token)

    def to_token_text(self, text, embedding_prefix):
        if text is None:
            return None
        if embedding_prefix is not None:
            text = embedding_prefix + text
        return text

    def to_lookup_id(self, sent, token):
        if sent is not None and token is not None:
            return (sent.docid, sent.index, token.index_in_sentence)
        else:
            return None

    def to_lookup_id_sent(self, sent):
        if sent is not None:
            return (sent.docid, sent.index)
        else:
            return None

    def get_vector(self, token, embedding_prefix=None, try_lower=True, try_lemma=True, sent=None):
        """
        Returns:
            (str, int, np.array[float])
        :param token: could be a string or nlplingo.text.text_span.Token
        :return: str==(original token string, lowercase, or lemma), int==embedding-index, np.array=embedding-vector
        """
        idx = -1
        vec = None
        text = None

        if isinstance(token, six.string_types) or type(token) is np.unicode_:
            if token == '_pad':
                idx = len(self.word_vec) - 1
                vec = self.word_vec[idx]
            else:
                return None
        else:
            text = self.to_token_text(token.text, embedding_prefix)

            lookup_id = self.to_lookup_id(sent, token)
            if lookup_id is not None:
                if lookup_id in self.word_lookup:
                    idx = self.word_lookup[lookup_id]
                    vec = self.word_vec[idx]

        return (text, idx, vec)

    def get_sent_vector(self, sent):
        """
        Returns:
            (int, np.array[float])
        :param sent: could be a string or nlplingo.text.text_span.Sentence
        :return: str==(original token string, lowercase, or lemma), int==embedding-index, np.array=embedding-vector
        """
        idx = -1
        vec = None

        lookup_id_sent = self.to_lookup_id_sent(sent)
        if lookup_id_sent is not None:
            if lookup_id_sent in self.sent_lookup:
                idx = self.sent_lookup[lookup_id_sent]
                vec = self.sent_vec[idx]

        return (idx, vec)


    def get_vector_by_index(self, idx):
        if idx < self.base_size:
            return (self.lookup_ids[idx], idx, self.word_vec[idx])
        else:
            raise ValueError('Given an idx which is out of bounds for embedding vocab')


class DependencyEmbedding(WordEmbeddingAbstract):

    def __init__(self, embedding_filepath, vocab_size, vector_size, none_token, missing_token):
        """
        :type params: nlplingo.common.parameters.Parameters
        """

        print('Loading embeddings file ', embedding_filepath)
        words, word_vec = self.load_word_vec_file(embedding_filepath, vocab_size, vector_size)
        self.words = words
        """:type: numpy.ndarray"""
        self.word_vec = word_vec
        """:type: numpy.ndarray"""

        self.word_lookup = dict((w, i) for i, w in enumerate(self.words))
        self.vector_length = vector_size
        self.base_size = len(self.words)
        #self.word_vecLength = None
        #self.token_map = dict()
        #self.accessed_indices = set()

        self.none_token = none_token
        text, self.none_index, vec = self.get_vector(self.none_token)
        assert self.none_index != -1

        self.missing_token = missing_token
        text, self.missing_index, vec = self.get_vector(self.missing_token)
        assert self.missing_index != -1

    class Factory(object):
        def create(self, embeddings_params):
            return DependencyEmbedding(
                embeddings_params['embedding_file'],
                embeddings_params['vocab_size'],
                embeddings_params['vector_size'],
                embeddings_params['none_token'],
                embeddings_params['missing_token']
            )

    def load_word_vec_file(self, embedding_filepath, vocab_size, vector_size):
        npz_file = embedding_filepath
        if os.path.exists(npz_file):
            data = np.load(npz_file)
            return (data['index'], data['embeddings'])


    def get_none_vector(self):
        return self.get_vector(self.none_token)

    def get_missing_vector(self):
        return self.get_vector(self.missing_token)

    def to_token_text(self, text, embedding_prefix):
        if text is None:
            return None
        if embedding_prefix is not None:
            text = embedding_prefix + text
        return text

    def get_vector(self, token, embedding_prefix=None, try_lower=True, try_lemma=True, sent=None):
        """
        Returns:
            (str, int, np.array[float])
        :param token: could be a string or nlplingo.text.text_span.Token
        :param sent: Fitting the interface, not used
        :return: str==(original token string, lowercase, or lemma), int==embedding-index, np.array=embedding-vector
        """
        idx = -1
        vec = None
        if isinstance(token, six.string_types) or type(token) is np.unicode_:
            text = self.to_token_text(token, embedding_prefix)

            if text in self.word_lookup:
                idx = self.word_lookup[text]
                vec = self.word_vec[idx]
            if idx < 0:
                text = None
        else:
            text = self.to_token_text(token.text, embedding_prefix)
            if text in self.word_lookup:
                idx = self.word_lookup[text]
                vec = self.word_vec[idx]
            if idx < 0:
                text = self.to_token_text(token.text.lower(), embedding_prefix)
                if text in self.word_lookup:
                    idx = self.word_lookup[text]
                    vec = self.word_vec[idx]
            if idx < 0:
                text = self.to_token_text(token.lemma, embedding_prefix)
                if text in self.word_lookup:
                    idx = self.word_lookup[text]
                    vec = self.word_vec[idx]
            if idx < 0:
                text = None
        #if idx > -1:
        #    self.accessed_indices.add(idx)
        return (text, idx, vec)

    def get_vector_by_index(self, idx):
        if idx < self.base_size:
            return (self.words[idx], idx, self.word_vec[idx])
        else:
            raise ValueError('Given an idx which is out of bounds for embedding vocab')

    def get_sent_vector(self, token, embedding_prefix=None, try_lower=True, try_lemma=True, sent=None):
        return (None, None)

    def to_lookup_id(self, sent, token):
        return None

    def to_lookup_id_sent(self, sent):
        return None


class WordEmbeddingFactory(object):
    factories = {
        'word_embeddings': WordEmbedding.Factory(),
        'context_word_embeddings': ContextWordEmbedding.Factory(),
        'dependency_embeddings': DependencyEmbedding.Factory()
    }

    @staticmethod
    def createWordEmbedding(id, embeddings_params):
        if id in WordEmbeddingFactory.factories:
            return WordEmbeddingFactory.factories[id].create(embeddings_params)
        else:
            raise RuntimeError(
                'Word Embedding type not supported: {}'.format(
                    id
                )
            )


def load_embeddings(params):
    """
    :return: dict[str : WordEmbeddingAbstract]
    """
    embeddings = dict()

    if 'embeddings' in params and 'embedding_file' in params['embeddings']:
        embeddings_params = params['embeddings']
        word_embeddings = WordEmbeddingFactory.createWordEmbedding(
            embeddings_params.get('type', 'word_embeddings'),
            embeddings_params
        )
        embeddings['word_embeddings'] = word_embeddings
        print('Word embeddings loaded')

    if 'dependency_embeddings' in params and 'embedding_file' in params['dependency_embeddings']:
        dep_embeddings_params = params['dependency_embeddings']
        dependency_embeddings = WordEmbeddingFactory.createWordEmbedding(
            dep_embeddings_params.get('type', 'dependency_embeddings'),
            dep_embeddings_params
        )
        embeddings['dependency_embeddings'] = dependency_embeddings
        print('Dependency embeddings loaded')

    return embeddings
