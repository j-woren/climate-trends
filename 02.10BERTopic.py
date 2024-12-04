from bertopic import BERTopic
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

# Set up stop words
stop_words = stopwords.words("english")

# Configure vectorizer and load data
vectorizer_model = CountVectorizer(stop_words=stop_words, ngram_range=(1, 3))
cdf_subs = pd.read_csv('/Users/trevor/Desktop/Research/climate-trends/unique_english_abstract_not_null.csv')
cdf_subs = cdf_subs.dropna(subset=['cleaned_abstract'])
cdf_subs['cleaned_abstract'] = cdf_subs['cleaned_abstract'].astype(str)

# Initialize embedding model and BERTopic
# use the SciBERT embedding model
word_embedding_model = models.Transformer("allenai/scibert_scivocab_uncased")
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
topic_model = BERTopic(embedding_model=embedding_model, vectorizer_model=vectorizer_model)

# Fit-transform and extract topics and probabilities
topics, probabilities = topic_model.fit_transform(cdf_subs['cleaned_abstract'].tolist())

# Add topics and probabilities to the DataFrame
cdf_subs['topic'] = topics
cdf_subs['probability'] = probabilities

# Get topic information and merge with the DataFrame
topic_info = topic_model.get_topic_info()
topic_info.rename(columns={'Topic': 'topic'}, inplace=True)
cdf_subs = cdf_subs.merge(topic_info[['topic', 'Name']], on='topic', how='left')

# Generate embeddings for the abstracts and add them to the DataFrame
embeddings = embedding_model.encode(cdf_subs['cleaned_abstract'].tolist(), show_progress_bar=True)
cdf_subs['embedding'] = embeddings.tolist()

topic_model.save("/Users/trevor/Desktop/Research/climate-trends/BERTopic_model")

# Save the DataFrame to a JSON file
cdf_subs.to_json('/Users/trevor/Desktop/Research/climate-trends/unique_english_abstract_not_null_bertopic_embeddings.json', 
                 orient='records', 
                 indent=4)