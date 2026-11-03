from utils import *
import re

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

def save_md(md_text, file_path):
    """
    Save the given markdown text to a file at the specified file path.
    The resulting markdown file can be opened in any markdown viewer or editor.

    Parameters
    ----------
    md_text : str
        The markdown text to be saved.
    file_path : str
        The path to the file where the markdown text will be saved.   
        The file extension should be .md to indicate that it is a markdown file.
    """

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_text)