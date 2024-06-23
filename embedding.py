from typing import List,Dict
import numpy as np
import string
import torch
from transformers import AutoTokenizer,AutoModel
from sentence_transformers import SentenceTransformer
from laserembeddings import Laser
import gensim.downloader as api
import spacy
from sklearn.decomposition import TruncatedSVD,PCA,KernelPCA,SparsePCA,MiniBatchSparsePCA,NMF,MiniBatchNMF,FactorAnalysis,FastICA
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.manifold import LocallyLinearEmbedding,Isomap
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")
nltk.download(['punkt', 'stopwords'])
nltk_stopwords_set = set(stopwords.words('english'))

class TextEmbedding():
    def __init__(self):
        pass

    def __repr__(self):
        return f"TextEmbedding()"

    def __str__(self):
        return "Class to embed text using various methods."

    def get_sklearn_embedding(self,texts:List[str] = [],custom_max_features:int = 5_000,custom_dtype:np.dtype = np.float64,is_lower:bool = True,use_stop_words:bool = True,custom_ngram_range:tuple = (1,1),vectorizer:str = 'tfidf',reduction_method:str = 'svd',embed_size:int = 50)->np.ndarray:
        """
        Convert a list of texts into embeddings using a selected vectorization method and dimensionality reduction method.

        Args:
        - texts (list): List of strings representing documents.
        - custom_max_features (int), default = 5000: Maximum number of features (terms) to be used in vectorization.
        - custom_dtype (np.dtype), default = np.float64: Data type of the matrix created by vectorization.
        - is_lower (bool), default = True: Convert text to lowercase.
        - use_stop_words (bool), default = True: Whether to use English stop words during vectorization.
        - custom_ngram_range (tuple), default = (1,1): Range for n-grams to be extracted.
          For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams.
        - vectorizer (str), default = 'tfidf': Vectorization method to use.
          Options: 'tfidf','count','binary'.
        - reduction_method (str), default = 'svd': Dimensionality reduction method to use.
          Options: 'svd','pca','kernel_pca','sparse_pca','mini_batch_sparse_pca','nmf','mini_batch_nmf',
          'factor_analysis','fast_ica','isomap','locally_linear_embedding','gaussian_random_projection',
          'sparse_random_projection'.
        - embed_size (int), default = 50: Number of components to keep in the reduced embedding space.

        Returns:
        - np.ndarray: Embeddings of the input texts.
        """

        if vectorizer == 'tfidf':
            if use_stop_words:
                vectorizer = TfidfVectorizer(decode_error = 'strict',use_idf = True,smooth_idf = True,binary = False,lowercase = is_lower,max_features = custom_max_features,dtype = custom_dtype,ngram_range = custom_ngram_range,stop_words = 'english')
            else:
                vectorizer = TfidfVectorizer(decode_error = 'strict',use_idf = True,smooth_idf = True,binary = False,lowercase = is_lower,max_features = custom_max_features,dtype = custom_dtype,ngram_range = custom_ngram_range)
        elif vectorizer == 'count':
            if use_stop_words:
                vectorizer = CountVectorizer(decode_error = 'strict',binary = False,lowercase = is_lower,max_features = custom_max_features,dtype = custom_dtype,ngram_range = custom_ngram_range,stop_words = 'english')
            else:
                vectorizer = CountVectorizer(decode_error = 'strict',binary = False,lowercase = is_lower,max_features = custom_max_features,dtype = custom_dtype,ngram_range = custom_ngram_range)
        elif vectorizer == 'binary':
            if use_stop_words:
                vectorizer = CountVectorizer(decode_error = 'strict',binary = True,lowercase = is_lower,max_features = custom_max_features,dtype = custom_dtype,ngram_range = custom_ngram_range,stop_words = 'english')
            else:
                vectorizer = CountVectorizer(decode_error = 'strict',binary = True,lowercase = is_lower,max_features = custom_max_features,dtype = custom_dtype,ngram_range = custom_ngram_range)
        else:
            raise ValueError("Vectorizer must be 'tfidf', 'count' or 'binary'")

        if reduction_method == 'svd':
            reduction_method = TruncatedSVD(n_components = embed_size)
        elif reduction_method == 'pca':
            reduction_method = PCA(n_components = embed_size)
        elif reduction_method == 'kernel_pca':
            reduction_method = KernelPCA(n_components = embed_size)
        elif reduction_method == 'sparse_pca':
            reduction_method = SparsePCA(n_components = embed_size)
        elif reduction_method == 'mini_batch_sparse_pca':
            reduction_method = MiniBatchSparsePCA(n_components = embed_size)
        elif reduction_method == 'nmf':
            reduction_method = NMF(n_components = embed_size)
        elif reduction_method == 'mini_batch_nmf':
            reduction_method = MiniBatchNMF(n_components = embed_size)
        elif reduction_method == 'factor_analysis':
            reduction_method = FactorAnalysis(n_components = embed_size)
        elif reduction_method == 'fast_ica':
            reduction_method = FastICA(n_components = embed_size)
        elif reduction_method == 'isomap':
            reduction_method = Isomap(n_components = embed_size)
        elif reduction_method == 'locally_linear_embedding':
            reduction_method = LocallyLinearEmbedding(n_components = embed_size)
        elif reduction_method == 'gaussian_random_projection':
            reduction_method = GaussianRandomProjection(n_components = embed_size)
        elif reduction_method == 'sparse_random_projection':
            reduction_method = SparseRandomProjection(n_components = embed_size)
        else:
            raise ValueError("Reduction Method must be 'svd', 'pca', 'kernel_pca', 'sparse_pca', 'mini_batch_sparse_pca', 'nmf', 'mini_batch_nmf', 'factor_analysis', 'fast_ica', 'isomap', 'locally_linear_embedding', 'gaussian_random_projection' or 'sparse_random_projection'")

        custom_pipeline = Pipeline([('vectorizer',vectorizer),('reduction',reduction_method)])

        try:
            sklearn_embeddings = custom_pipeline.fit_transform(texts)
        except Exception as e:
            raise RuntimeError(f"Pipeline not properly fitted. {e}")

        return sklearn_embeddings

    def processing_word_embedding(self,stxt:str = '',model_ = None)->np.ndarray:
        """
        Retrieve the word embedding vector for a given word or calculate an average embedding for out-of-vocabulary words.

        Args:
        - stxt (str): Input word or character.
        - model_name: Pre-trained word embedding model.

        Returns:
        - np.ndarray: Embeddings of the input texts.
        """

        if stxt in model_:
            return model_[stxt]
        else:
            embedding_container:list = []
            for char in stxt:
                if char in model_:
                    embedding_container.append(model_[char])
                else:
                    if ord(char)<91:
                        char:str = char.lower()
                        if char in model_:
                            embedding_container.append(model_[char])
                        else:
                            pass
                    else:
                        char:str = char.upper()
                        if char in model_:
                            embedding_container.append(model_[char])
                        else:
                            pass
            return np.mean(np.array(embedding_container),axis = 0)

    def get_word_embedding(self,texts:List[str] = [],model_name = None)->np.ndarray:
        """
        Compute word embeddings using a pre-trained word embedding model.

        Args:
        - texts (list): List of strings representing documents.
        - model_name: Pre-trained word embedding model.

        Returns:
        - np.ndarray: Embeddings of the input texts.
        """

        return np.array([np.mean([self.processing_word_embedding(stxt = word,model_ = model_name) for word in word_tokenize(doc)],axis = 0) for doc in texts])

    def get_laser_embeddings(self,texts:List[str] = [],model_name = None)->np.ndarray:
        """
        Compute LASER (Language-Agnostic SEntence Representations) embeddings for a list of texts.

        Args:
        - texts (list): List of strings representing documents.
        - model_name: Pre-trained word embedding model.

        Returns:
        - np.ndarray: LASER Embeddings of the input texts.
        """

        return model_name.embed_sentences(texts,lang = 'en')

    def get_spacy_embedding(self,texts:List[str] = [],model_name = None)->np.ndarray:
        """
        Compute spaCy word embeddings for a list of texts.

        Args:
        - texts (list): List of strings representing documents.
        - model_name: Pre-trained word embedding model.

        Returns:
        - np.ndarray: spaCy embeddings of the input texts.
        """

        return np.array([model_name(doc).vector for doc in texts])

    def get_sentence_transformers_embedding(self,texts:List[str] = [],model_name = None)->np.ndarray:
        """
        Compute Sentence Transformers embeddings for a list of texts.

        Args:
        - texts (list): List of strings representing documents.
        - model_name: Pre-trained word embedding model.

        Returns:
        - np.ndarray: Sentence Transformers embeddings of the input texts.
        """

        return model_name.encode(texts)

    def get_pre_trained_models_embedding(self,texts:List[str] = [],model_name = None,model_tokenizer = None,custom_max_length:int = 256)->np.ndarray:
        """
        Compute embeddings using a pre-trained transformer model (e.g.,BERT) and tokenizer.

        Args:
        - texts (list): List of strings representing documents.
        - model_name: Pre-trained transformer model instance.
        - model_tokenizer: Pre-trained tokenizer instance.
        - custom_max_length (int), default = 256: Maximum Length before truncation.

        Returns:
        - np.ndarray: Embeddings of the input texts using the pre-trained model.
        """

        model_encoded_inputs = model_tokenizer(texts,padding = True,truncation = True,max_length = custom_max_length,return_tensors = "pt")
        with torch.no_grad():
            model_encoded_outputs = model_name(**model_encoded_inputs)
        model_embeddings = model_encoded_outputs.last_hidden_state.mean(dim = 1)
        return model_embeddings.numpy()

    def remove_punctuation_strings(self,txt:str = '')->str:
        """
        Remove punctuation characters from a given string.

        Args:
        - txt (str): Input text to remove punctuation from.

        Returns:
        - str: Text with punctuation characters removed.
        """

        return txt.translate(str.maketrans('','',string.punctuation))

    def pre_processing_text(self,txt:str = '',is_lower:bool = True,stop_words_set:set = set(stopwords.words('english')),remove_stop_words:bool = True)->str:
        """
        Preprocesses the input text based on the specified options.

        Args:
        - txt (str): Input text to preprocess.
        - is_lower (bool), default = True: Flag indicating whether to convert text to lowercase (default = True).
        - stop_words_set (set), default = "nltk english stopwords": Set of stop words to remove from text (default = set()).
        - remove_stop_words (bool), default = True: Flag indicating whether to remove stop words (default = True).

        Returns:
        - str: Processed text after tokenization and preprocessing steps.
        """

        if is_lower:
            txt:str = str(txt).lower().strip()
        else:
            txt:str = str(txt).strip()

        if remove_stop_words:
            return ' '.join([x for x in word_tokenize(txt) if (x.isalnum() and x not in stop_words_set)])
        else:
            return ' '.join([x for x in word_tokenize(txt) if x.isalnum()])

# Example usage:
if __name__ == "__main__":

    laser_embeddings = Laser()

    nlp_spacy_model = spacy.load("en_core_web_sm",enable = ["tok2vec"])

    word_embedding_model = api.load("glove-wiki-gigaword-50")

    sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

    hf_model_name:str = "xlnet-base-cased"
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModel.from_pretrained(hf_model_name)

    test_file = pd.read_csv('Test1.csv',dtype='str',encoding='utf-8')
    print(f'{test_file.shape}')

    text_to_vector = TextEmbedding()

    # Single Instances

    test_sample = test_file.iloc[0]['Title']
    print(f'{test_sample = }')

    print(text_to_vector.remove_punctuation_strings.__doc__)
    print(text_to_vector.remove_punctuation_strings(txt = test_sample))

    print(text_to_vector.pre_processing_text.__doc__)
    print(text_to_vector.pre_processing_text(txt = test_sample))

    print(text_to_vector.get_sklearn_embedding.__doc__)
    print(text_to_vector.get_sklearn_embedding(texts = [test_sample],embed_size = 10).shape)

    print(text_to_vector.processing_word_embedding.__doc__)
    print(text_to_vector.processing_word_embedding(stxt = test_sample,model_ = word_embedding_model).shape)

    print(text_to_vector.get_word_embedding.__doc__)
    print(text_to_vector.get_word_embedding(texts = [test_sample],model_name = word_embedding_model).shape)

    print(text_to_vector.get_laser_embeddings.__doc__)
    print(text_to_vector.get_laser_embeddings(texts = [test_sample],model_name = laser_embeddings).shape)

    print(text_to_vector.get_spacy_embedding.__doc__)
    print(text_to_vector.get_spacy_embedding(texts = [test_sample],model_name = nlp_spacy_model).shape)

    print(text_to_vector.get_sentence_transformers_embedding.__doc__)
    print(text_to_vector.get_sentence_transformers_embedding(texts = [test_sample],model_name = sentence_transformer_model).shape)

    print(text_to_vector.get_pre_trained_models_embedding.__doc__)
    print(text_to_vector.get_pre_trained_models_embedding(texts = [test_sample],model_name = hf_model,model_tokenizer = hf_tokenizer,custom_max_length = 256).shape)


    # Multiple Instances
    test_file['Remove_Punctuation'] = test_file['Title'].apply(lambda x: text_to_vector.remove_punctuation_strings(txt=x))
    test_file['Pre_Processing'] = test_file['Title'].apply(lambda x: text_to_vector.pre_processing_text(txt=x))
    print(test_file.head(5).to_dict(orient='records'))

    print(f'{test_file.shape = }')
    print(text_to_vector.get_sklearn_embedding(texts=test_file['Title'].tolist(),embed_size=5).shape)
    print(text_to_vector.get_word_embedding(texts=test_file['Title'].tolist(),model_name=word_embedding_model).shape)
    print(text_to_vector.get_laser_embeddings(texts=test_file['Title'].tolist(),model_name = laser_embeddings).shape)
    print(text_to_vector.get_spacy_embedding(texts=test_file['Title'].tolist(),model_name = nlp_spacy_model).shape)
    print(text_to_vector.get_sentence_transformers_embedding(texts=test_file['Title'].tolist(),model_name = sentence_transformer_model).shape)
    print(text_to_vector.get_pre_trained_models_embedding(texts=test_file['Title'].tolist(),model_name = hf_model,model_tokenizer = hf_tokenizer,custom_max_length=256).shape)

