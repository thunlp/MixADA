from .base import CharSubstitute
from ..data_manager import DataManager
import numpy as np


disallowed = ['TAG', 'MALAYALAM', 'BAMUM', 'HIRAGANA', 'RUNIC', 'TAI', 'SUNDANESE', 'BATAK', 'LEPCHA', 'CHAM',
              'TELUGU', 'DEVANGARAI', 'BUGINESE', 'MYANMAR', 'LINEAR', 'SYLOTI', 'PHAGS-PA', 'CHEROKEE',
              'CANADIAN', 'YI', 'LYCIAN', 'HANGUL', 'KATAKANA', 'JAVANESE', 'ARABIC', 'KANNADA', 'BUHID',
              'TAGBANWA', 'DESERET', 'REJANG', 'BOPOMOFO', 'PERMIC', 'OSAGE', 'TAGALOG', 'MEETEI', 'CARIAN',
              'UGARITIC', 'ORIYA', 'ELBASAN', 'CYPRIOT', 'HANUNOO', 'GUJARATI', 'LYDIAN', 'MONGOLIAN', 'AVESTAN',
              'MEROITIC', 'KHAROSHTHI', 'HUNGARIAN', 'KHUDAWADI', 'ETHIOPIC', 'PERSIAN', 'OSMANYA', 'ELBASAN',
              'TIBETAN', 'BENGALI', 'TURKIC', 'THROWING', 'HANIFI', 'BRAHMI', 'KAITHI', 'LIMBU', 'LAO', 'CHAKMA',
              'DEVANAGARI', 'ITALIC', 'CJK', 'MEDEFAIDRIN', 'DIAMOND', 'SAURASHTRA', 'ADLAM', 'DUPLOYAN']
disallowed_codes = ['1F1A4', 'A7AF']  # 不允许编码


def get_hex_string(ch):
    return '{:04x}'.format(ord(ch)).upper()  # 获得字符16进制编码


class DCESSubstitute(CharSubstitute):
    """
    :Data Requirements: :py:data:`.AttackAssist.DCES`
    :Package Requirements: * **sklearn**
    
    An implementation of :py:class:`.CharSubstitute`.

    DCES substitute used in :py:class:`.VIPERAttacker`.

    """

    def __init__(self):
        self.descs, self.neigh = DataManager.load("AttackAssist.DCES")
        # load

    def __call__(self, char, threshold):
        """
        :param int threshold: Returns top k results (k = threshold).
        """
        c = get_hex_string(char)

        if c in self.descs:
            description = self.descs[c]["description"]
        else:
            return [char, 1]

        tokens = description.split(' ')
        case = 'unknown'
        identifiers = []

        for token in tokens:
            if len(token) == 1:
                identifiers.append(token)
            elif token == 'SMALL':
                case = 'SMALL'
            elif token == 'CAPITAL':
                case = 'CAPITAL'

        matches = []
        match_ids = []
        for i in identifiers:
            for idx, val in self.descs.items():
                desc_toks = val["description"].split(' ')
                if i in desc_toks and not np.any(np.in1d(desc_toks, disallowed)) and \
                        not np.any(np.in1d(idx, disallowed_codes)) and \
                        not int(idx, 16) > 30000:

                    desc_toks = np.array(desc_toks)
                    case_descriptor = desc_toks[(desc_toks == 'SMALL') | (desc_toks == 'CAPITAL')]

                    if len(case_descriptor) > 1:
                        case_descriptor = case_descriptor[0]
                    elif len(case_descriptor) == 0:
                        case = 'unknown'

                    if case == 'unknown' or case == case_descriptor:
                        match_ids.append(idx)
                        matches.append(val["vec"])

        if len(matches) == 0:
            return [(char, 1)]

        match_vecs = np.stack(matches)
        Y = match_vecs

        self.neigh.fit(Y)

        X = self.descs[c]["vec"].reshape(1, -1)

        if Y.shape[0] > threshold:
            dists, idxs = self.neigh.kneighbors(X, threshold, return_distance=True)
        else:
            dists, idxs = self.neigh.kneighbors(X, Y.shape[0], return_distance=True)
        probs = dists.flatten()

        charcodes = [match_ids[idx] for idx in idxs.flatten()]
        
        chars = []
        for charcode in charcodes:
            chars.append(chr(int(charcode, 16)))
        ret = list(zip(chars, probs))
        return list(filter(lambda x: x[1] < threshold, ret))
