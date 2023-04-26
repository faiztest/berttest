#import module
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from pprint import pprint
import spacy
import pickle
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from io import StringIO
from ipywidgets.embed import embed_minimal_html
from nltk.stem.snowball import SnowballStemmer
from bertopic import BERTopic
import plotly.express as px
from sklearn.cluster import KMeans


#===config===
st.set_page_config(
     page_title="Coconut",
     page_icon="🥥",
     layout="wide"
)
st.header("pyLDA")
st.subheader('Put your CSV file here ...')

#===upload file===
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    papers = pd.read_csv(uploaded_file)
    paper = papers.dropna(subset=['Abstract'])
    paper = paper[~paper.Abstract.str.contains("No abstract available")]
    paper = paper[~paper.Abstract.str.contains("STRAIT")]
        
    #===mapping===
    paper['Abstract_pre'] = paper['Abstract'].map(lambda x: re.sub('[:;\!?•-]=', '', x))
    paper['Abstract_pre'] = paper['Abstract_pre'].map(lambda x: x.lower())
    paper['Abstract_pre'] = paper['Abstract_pre'].map(lambda x: re.sub('©.*', '', x))
        
    #===lemmatize===
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(text):
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    paper['Abstract_lem'] = paper['Abstract_pre'].apply(lemmatize_words)
       
    #===stopword removal===
    stop = stopwords.words('english')
    paper['Abstract_stop'] = paper['Abstract_lem'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    

    method = st.selectbox(
         'Choose method',
         ('pyLDA', 'BERTopic'))
       
    #===topic===
    if method is 'pyLDA':
        topic_abs = paper.Abstract_stop.values.tolist()
        topic_abs = [t.split(' ') for t in topic_abs]
        id2word = Dictionary(topic_abs)
        corpus = [id2word.doc2bow(text) for text in topic_abs]
        num_topic = st.slider('Choose number of topics', min_value=2, max_value=50, step=1)
         
        #===LDA===
        lda_model = LdaModel(corpus=corpus,
                    id2word=id2word,
                    num_topics=num_topic, 
                    random_state=0,
                    chunksize=100,
                    alpha='auto',
                    per_word_topics=True)

        pprint(lda_model.print_topics())
        doc_lda = lda_model[corpus]

        #===coherence score===
        if st.button('📐 Calculate coherence'):
                    with st.spinner('Calculating, please wait ....'):    
                       coherence_model_lda = CoherenceModel(model=lda_model, texts=topic_abs, dictionary=id2word, coherence='c_v')
                       coherence_lda = coherence_model_lda.get_coherence()
                       st.write(coherence_lda)
        #===visualization===
        if st.button('📈 Generate visualization'):
                    with st.spinner('Creating pyLDAvis Visualization ...'):
                        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
                        py_lda_vis_html = pyLDAvis.prepared_data_to_html(vis)
                        components.html(py_lda_vis_html, width=1700, height=800)
                        st.markdown('Copyright (c) 2015, Ben Mabey. https://github.com/bmabey/pyLDAvis')
        
    elif method is 'BERTopic':
        num_btopic = st.slider('Choose number of topics', min_value=4, max_value=50, step=1)
        topic_abs = paper.Abstract_stop.values.tolist()
        topic_time = paper.Year.values.tolist()
        cluster_model = KMeans(n_clusters=num_btopic)
        topic_model = BERTopic(hdbscan_model=cluster_model).fit(topic_abs)
        topics, probs = topic_model.fit_transform(topic_abs)
        
        #===visualization===
        viz = st.selectbox(
         'Choose visualization',
         ('Visualize Topics', 'Visualize Documents', 'Visualize Document Hierarchy', 'Visualize Topic Similarity', 'Visualize Terms', 'Visualize Topics over Time'))
     
        if viz == 'Visualize Topics':
               fig1 = topic_model.visualize_topics()
               st.write(fig1)
               
        elif viz == 'Visualize Documents':
               fig2 = topic_model.visualize_documents(topic_abs)
               st.write(fig2)
        
        elif viz == 'Visualize Document Hierarchy':
               fig3 = topic_model.visualize_hierarchy(top_n_topics=num_btopic)
               st.write(fig3)
        
        elif viz == 'Visualize Topic Similarity':
               fig4 = topic_model.visualize_heatmap(n_clusters=num_btopic-1, width=1000, height=1000)
               st.write(fig4)
               
        elif viz == 'Visualize Terms':
               fig5 = topic_model.visualize_barchart(num_topic=num_btopic)
               st.write(fig5)
               
        elif viz == 'Visualize Topics over Time':
               topics_over_time = topic_model.topics_over_time(topic_abs, topic_time)
               fig6 = topic_model.visualize_topics_over_time(topics_over_time)
               st.write(fig6)
