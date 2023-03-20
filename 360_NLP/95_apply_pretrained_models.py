#%% packages
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from scipy import spatial
import pandas as pd

#%% Data Import 
twitter_file = 'data/Tweets.csv'
df = pd.read_csv(twitter_file).dropna().sample(1000, random_state=123).reset_index(drop=True)
df.head()
# %% Sentiment Analysis
#----------------------
# sentiment_pipeline = pipeline("sentiment-analysis")
sentiment_pipeline = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")
data = df.loc[2, 'text']
data = 'Das finde ich ganz ok.'
sentiment_pipeline(data)

# %% Find similar Tweets
#-----------------------
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#%%
df=df.assign(embeddings=df['text'].apply(lambda x: model.encode(x)))

#%%
def closest_description(desc):
    data=df.copy()
    inp_vector=model.encode(desc)
    data['similarity'] = data['embeddings'].apply(lambda x: 1 - spatial.distance.cosine(x, inp_vector) )
    data = data.sort_values('similarity',ascending=False).head(3)
    return data[['text', 'sentiment']]
# %%
closest_description('this is amazing')
# %%
