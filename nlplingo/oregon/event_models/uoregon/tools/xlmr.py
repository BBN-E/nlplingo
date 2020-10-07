from fairseq.models.roberta import XLMRModel
# from fairseq.models.roberta.alignment_utils import align_bpe_to_words
from .utils import ensure_dir
from .global_constants import WORKING_DIR
import os
from nlplingo.oregon.event_models.uoregon.define_opt import opt

#ensure_dir(os.path.join(WORKING_DIR, 'xlmr_resources'))	# <==


def align_bpe_to_words(roberta, bpe_tokens, other_tokens):
    """
    Helper to align GPT-2 BPE to other tokenization formats (e.g., spaCy).
    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        bpe_tokens (torch.LongTensor): GPT-2 BPE tokens of shape `(T_bpe)`
        other_tokens (List[str]): other tokens of shape `(T_words)`
    Returns:
        List[str]: mapping from *other_tokens* to corresponding *bpe_tokens*.
    """
    if not bpe_tokens.dim() == 1:
        return None
    if not bpe_tokens[0] == 0:
        return None

    def clean(text):
        return text.strip()

    """
    bpe_tokens: tensor([       0,     62,  14012,    111,   9907,  98809,  19175,     15, 186831,
                            1388,  92865,    765,  18782,    297,   7103,   8035,  52875,    297,
                             390,     70,   7082,  13918,      6,      5,      2])
    """

    # remove whitespaces to simplify alignment
    bpe_tokens = [roberta.task.source_dictionary.string([x]) for x in bpe_tokens]
    # bpe_tokens:   ['', '▁A', '▁number', '▁of', '▁National', '▁Football', '▁League', '▁(', '▁NFL', '▁)', '▁players', '▁have', '▁protest', 'ed', '▁after', '▁being', '▁attack', 'ed', '▁by', '▁the', '▁US', '▁president', '▁', '.', '']
    bpe_tokens = [clean(roberta.bpe.decode(x) if x not in {'<s>', ''} else x) for x in bpe_tokens]
    # bpe_tokens:   ['', 'A', 'number', 'of', 'National', 'Football', 'League', '(', 'NFL', ')', 'players', 'have', 'protest', 'ed', 'after', 'being', 'attack', 'ed', 'by', 'the', 'US', 'president', '', '.', '']
    other_tokens = [clean(str(o)) for o in other_tokens]
    # other_tokens: ['A', 'number', 'of', 'National', 'Football', 'League', '(', 'NFL', ')', 'players', 'have', 'protested', 'after', 'being', 'attacked', 'by', 'the', 'US', 'president', '.']

    # strip leading <s>
    bpe_tokens = bpe_tokens[1:]
    # bpe_tokens:   ['A', 'number', 'of', 'National', 'Football', 'League', '(', 'NFL', ')', 'players', 'have', 'protest', 'ed', 'after', 'being', 'attack', 'ed', 'by', 'the', 'US', 'president', '', '.', '']
    if not ''.join(bpe_tokens) == ''.join(other_tokens):
        return None

    # create alignment from every word to a list of BPE tokens
    alignment = []
    bpe_toks = filter(lambda item: item[1] != '', enumerate(bpe_tokens, start=1))   # let the starting index be 1, instead of starting index is 0
    # The filter() method returns an iterator that passed the function check for each element in the iterable.
    # empty strings are discarded from bpe_tokens to become bpe_toks

    j, bpe_tok = next(bpe_toks)
    for other_tok in other_tokens:
        bpe_indices = []
        while True:
            if other_tok.startswith(bpe_tok):
                bpe_indices.append(j)
                other_tok = other_tok[len(bpe_tok):]
                try:
                    j, bpe_tok = next(bpe_toks)
                except StopIteration:
                    j, bpe_tok = None, None
            elif bpe_tok.startswith(other_tok):
                # other_tok spans multiple BPE tokens
                bpe_indices.append(j)
                bpe_tok = bpe_tok[len(other_tok):]
                other_tok = ''
            else:
                print('Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
                return None
            if other_tok == '':
                break
        if not len(bpe_indices) > 0:
            return None
        alignment.append(bpe_indices)
    if not len(alignment) == len(other_tokens):
        return None

    # alignment [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12, 13], [14], [15], [16, 17], [18], [19], [20], [21], [23]]
    return alignment


class XLMRTool:
    def __init__(self, xlmr_version):
        assert xlmr_version in ['xlmr.large', 'xlmr.base']
        self.xlmr_version = xlmr_version

    def get_token_ids(self, xlmr_model, stanford_words, trigger_word=None):
        """
        stanford_words   ['A', 'number', 'of', 'National', 'Football', 'League', '(', 'NFL', ')', 'players', 'have', 'protested', 'after', 'being', 'attacked', 'by', 'the', 'US', 'president', '.']
        trigger_word    protested
        """
        xlmr_model.eval()
        text = ' '.join(stanford_words)
        if trigger_word is not None:
            xlmr_tokens = xlmr_model.encode(text)
            token_groups = align_bpe_to_words(xlmr_model, xlmr_tokens, stanford_words)
            if token_groups is None:
                return None, None
            retrieve_ids = [group[0] for group in token_groups]
            """
            xlmr_tokens     tensor([     0,     62,  14012,    111,   9907,  98809,  19175,     15, 186831,
                                      1388,  92865,    765,  18782,    297,   7103,   8035,  52875,    297,
                                       390,     70,   7082,  13918,      6,      5,      2]), len=25
            token_groups [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12, 13], [14], [15], [16, 17], [18], [19], [20], [21], [23]], len=20
            retrieve_ids [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23], len=20
            """

            xlmr_tokens = xlmr_model.encode(text, trigger_word)
            """
            xlmr_tokens     tensor([     0,     62,  14012,    111,   9907,  98809,  19175,     15, 186831,
                                      1388,  92865,    765,  18782,    297,   7103,   8035,  52875,    297,
                                       390,     70,   7082,  13918,      6,      5,      2,      2,  18782,
                                       297,      2])
            """
        else:
            xlmr_tokens = xlmr_model.encode(text)
            token_groups = align_bpe_to_words(xlmr_model, xlmr_tokens, stanford_words)
            if token_groups is None:
                return None, None
            retrieve_ids = [group[0] for group in token_groups]
        if not len(stanford_words) == len(retrieve_ids):
            return None, None
        return xlmr_tokens, retrieve_ids


xlmr_tokenizer = XLMRTool(opt['xlmr_version'])
