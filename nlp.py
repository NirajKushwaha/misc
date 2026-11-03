from utils import *
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from cleantext import clean, clean_texts
from .texts import get_unique_words
from wordfreq import word_frequency
from scipy import sparse

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
        Generate a list of stop words based on the corpus using Zipf's law.
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
        IS NOT REALLY TESTED. LLM SLOP AT THE MOMENT. USE WITH CAUTION.

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

        # Select top-n entries directly from each sparse row. This avoids
        # materializing a dense document-by-vocabulary array.
        top_indices_per_doc = []
        top_data_per_doc = []
        important_column_mask = np.zeros(tfidf_matrix.shape[1], dtype=bool)

        for row_idx in range(tfidf_matrix.shape[0]):
            row_start, row_end = tfidf_matrix.indptr[row_idx:row_idx + 2]
            row_indices = tfidf_matrix.indices[row_start:row_end]
            row_data = tfidf_matrix.data[row_start:row_end]

            column_order = np.argsort(row_indices)
            sorted_order = column_order[
                np.argsort(row_data[column_order])[::-1]
            ]
            top_indices = row_indices[sorted_order[:n]]
            top_data = row_data[sorted_order[:n]]
            top_indices_per_doc.append(top_indices)
            top_data_per_doc.append(top_data)
            important_column_mask[top_indices] = True

        # Preserve the original vocabulary order.
        important_indices = np.flatnonzero(important_column_mask)
        important_words = feature_names[important_indices].tolist()
        column_positions = np.full(tfidf_matrix.shape[1], -1, dtype=np.intp)
        column_positions[important_indices] = np.arange(len(important_indices))

        # Build the coarse matrix directly in CSR format.
        top_columns_per_doc = []
        for top_indices in top_indices_per_doc:
            top_columns_per_doc.append(column_positions[top_indices])

        row_counts = np.array(
            [len(top_indices) for top_indices in top_indices_per_doc],
            dtype=np.intp
        )
        coarse_matrix = sparse.csr_matrix(
            (
                np.concatenate(top_data_per_doc),
                np.concatenate(top_columns_per_doc),
                np.concatenate([[0], np.cumsum(row_counts)])
            ),
            shape=(tfidf_matrix.shape[0], len(important_indices))
        )
        coarse_matrix.sort_indices()

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