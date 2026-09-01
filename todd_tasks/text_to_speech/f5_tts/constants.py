__all__ = [
    'MAX_DURATION',
    'EN_PUNCTUATIONS',
    'ZH_PUNCTUATIONS',
    'PUNCTUATIONS',
    'PUNCTUATION_PATTERN',
    'WHITESPACE_PATTERN',
    'SEGMENT_PATTERN',
    'TRANSLATION_TABLE',
]

import re

import regex

MAX_DURATION = 4096

EN_PUNCTUATIONS = '!,.:;?'
ZH_PUNCTUATIONS = '！，。：；？、'
PUNCTUATIONS = EN_PUNCTUATIONS + ZH_PUNCTUATIONS

PUNCTUATION_PATTERN = re.compile(
    rf'(?<=[{EN_PUNCTUATIONS}]\s)|(?<=[{ZH_PUNCTUATIONS}])',
)
WHITESPACE_PATTERN = re.compile(r'(?<=\s)')
SEGMENT_PATTERN = regex.compile(r'\p{ASCII}+|\P{ASCII}+')

TRANSLATION_TABLE = str.maketrans({
    ';': ',',
    '“': '"',
    '”': '"',
    '‘': "'",
    '’': "'",
})
