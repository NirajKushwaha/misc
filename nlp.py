from utils import *
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from cleantext import clean, clean_texts
from .texts import get_unique_words
from wordfreq import word_frequency

def custom_stop_words(include_these=None):
    """
    Get stop words list that includes some extra stop words that are due to the way sklearn's TfidfVectorizer tokenizes words.

    Parameters
    ----------
    include_these : list, None
        List of extra stop words to include. If None, no extra stop words are included.

    Returns
    -------
    list
        List of stop words.
    """

    stop_words = get_stop_words('en')

    base_vectorizer = TfidfVectorizer()
    analyzer = base_vectorizer.build_analyzer()

    processed_stop_words = []
    for word in stop_words:
        processed_stop_words.append(word)

        sklearn_tokenized_word = analyzer(word)

        if len(sklearn_tokenized_word) > 0:
            if sklearn_tokenized_word[0] != word:
                processed_stop_words.append(sklearn_tokenized_word[0])

    processed_stop_words.append("re")  ## From Sklearn's warning
    processed_stop_words.append("uh")  ## Seen in Data

    if include_these is not None:
        processed_stop_words.extend(include_these)

    return processed_stop_words


class Corpus:
    """
    Class for processing a corpus of text data.

    Parameters
    ----------
    corpus : list of str
        List of text documents.
    zipf_threshold : float, optional
        Threshold for word frequency based on Zipf's law. Words with frequency higher than this threshold
        will be considered stop words. Default is 1e-4.
    """

    def __init__(self, corpus,
                 zipf_threshold=1e-4):
        self.corpus = corpus
        self.cleaned_corpus = clean_texts(self.corpus,
                                            fix_unicode=True,
                                            no_digits=True,
                                            no_numbers=True,
                                            replace_with_digit="",
                                            replace_with_number="",
                                            n_jobs=-1)
        self.zipf_threshold = zipf_threshold
        self.stop_words = self.generate_stop_words()


    def generate_stop_words(self):
        """
        Generate a list of stop words based on the corpus.
        """

        unique_words = get_unique_words(self.cleaned_corpus)
        word_freqs = np.array([word_frequency(word, "en") for word in unique_words])

        ## removes all non english words
        non_en_ix = np.where(word_freqs==0)[0]
        add_to_stop_words = unique_words[non_en_ix]

        ## removes all words with frequency higher than the zipf_threshold
        high_freq_ix = np.where(word_freqs>self.zipf_threshold)[0]
        add_to_stop_words = np.concatenate([add_to_stop_words, unique_words[high_freq_ix]])

        return custom_stop_words(include_these=add_to_stop_words)

    def get_tfidf(self, df=False, use_cleaned_corpus=True, ngram_range=(1, 1)):
        """
        Get the TF-IDF matrix for the corpus.
        """

        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            ngram_range=ngram_range
        )

        if use_cleaned_corpus:
            tfidf_matrix = vectorizer.fit_transform(self.cleaned_corpus)
        else:
            tfidf_matrix = vectorizer.fit_transform(self.corpus)

        if df:
            feature_names = vectorizer.get_feature_names_out()
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=feature_names
            )

            return tfidf_df
        else:
            return vectorizer, tfidf_matrix

    def get_coarse_tfidf(self, n=10, use_cleaned_corpus=True):
        """
        Create a coarse-grained TF-IDF representation.

        The final vocabulary is the union of the top-n TF-IDF words
        from every document. For each document, words that are not
        among its top-n are set to zero.

        Parameters
        ----------
        n : int
            Number of top TF-IDF words to consider for each document.
        use_cleaned_corpus : bool
            Whether to use the cleaned corpus or the original corpus for TF-IDF calculation.

        Returns
        -------
        vectorizer : TfidfVectorizer
            Fitted vectorizer containing only the union vocabulary.

        tfidf_matrix : scipy.sparse.csr_matrix
            Coarse-grained sparse TF-IDF matrix.
        """

        corpus = self.cleaned_corpus if use_cleaned_corpus else self.corpus

        # First calculate the normal TF-IDF
        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words
        )

        tfidf_matrix = vectorizer.fit_transform(corpus)

        feature_names = vectorizer.get_feature_names_out()

        # Find top-n words for every document
        top_words_per_doc = []

        for row in tfidf_matrix:
            row = row.toarray().ravel()

            nonzero_indices = np.flatnonzero(row)

            # Sort by TF-IDF descending
            sorted_indices = nonzero_indices[
                np.argsort(row[nonzero_indices])[::-1]
            ]

            top_indices = sorted_indices[:n]

            top_words = set(feature_names[top_indices])
            top_words_per_doc.append(top_words)

        # Union of top-n words across all documents
        important_words = set().union(*top_words_per_doc)

        # Preserve the original vocabulary order
        important_words = [
            word for word in feature_names
            if word in important_words
        ]

        # Original word -> column index
        feature_to_index = {
            word: i for i, word in enumerate(feature_names)
        }

        important_indices = [
            feature_to_index[word]
            for word in important_words
        ]

        # Extract only the union vocabulary
        coarse_matrix = tfidf_matrix[:, important_indices].tocsr()

        # Mask words that aren't in the document's top-n
        coarse_matrix = coarse_matrix.tolil()

        for doc_idx, top_words in enumerate(top_words_per_doc):
            for col_idx, word in enumerate(important_words):
                if word not in top_words:
                    coarse_matrix[doc_idx, col_idx] = 0.0

        coarse_matrix = coarse_matrix.tocsr()

        # Create a new vectorizer with the reduced vocabulary.
        # This makes the feature names / vocabulary easy to work with.
        coarse_vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            vocabulary=important_words
        )

        return coarse_vectorizer, coarse_matrix



    def get_tf(self, df=False):
        """
        Get the term frequency matrix for the corpus.
        """

        vectorizer = CountVectorizer(
            stop_words=self.stop_words
        )

        tf_matrix = vectorizer.fit_transform(self.cleaned_corpus)

        if df:
            feature_names = vectorizer.get_feature_names_out()
            tf_df = pd.DataFrame(
                tf_matrix.toarray(),
                columns=feature_names
            )

            return tf_df
        else:
            return tf_matrix