from utils import *

def get_unique_words(corpus):
    """
    Get all unique words from a corpus of text.

    Parameters
    ----------
    corpus : list of str
        A list of strings representing the text corpus.
    
    Returns
    -------
    np.ndarray
        A sorted array of unique words found in the corpus, converted to lowercase.
    """

    pattern = re.compile(r"\b\w+\b")
    unique = set()
    for text in corpus:
        unique.update(map(str.lower, pattern.findall(text)))
    return np.sort(list(unique))