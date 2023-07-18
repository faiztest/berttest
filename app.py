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
import bitermplus as btm
import tmplot as tmp
import tomotopy
import sys
import spacy
import en_core_web_sm
import pipeline
import plotly.graph_objects as go
from html2image import Html2Image
from PIL import Image
import vl_convert as vlc


#===config===
st.set_page_config(
     page_title="Coconut",
     page_icon="🥥",
     layout="wide"
)
st.header("Topic Modeling")
st.subheader('Put your file here...')

#========unique id========
@st.cache_resource(ttl=3600)
def create_list():
    l = [1, 2, 3]
    return l

l = create_list()
first_list_value = l[0]
l[0] = first_list_value + 1
uID = str(l[0])

@st.cache_data(ttl=3600)
def get_ext(uploaded_file):
    extype = uID+uploaded_file.name
    return extype

#===clear cache===

def reset_biterm():
     try:
          biterm_map.clear()
          biterm_bar.clear()
     except NameError:
          biterm_topic.clear()

def reset_all():
     st.cache_data.clear()
        
#===clean csv===
@st.cache_data(ttl=3600, show_spinner=False)
def clean_csv(extype):
    try:
        paper = papers.dropna(subset=['Abstract'])
    except KeyError:
        st.error('Error: Please check your Abstract column.')
        sys.exit(1)
    paper = paper[~paper.Abstract.str.contains("No abstract available")]
    paper = paper[~paper.Abstract.str.contains("STRAIT")]
            
        #===mapping===
    paper['Abstract_pre'] = paper['Abstract'].map(lambda x: re.sub('[,:;\.!-?•=]', '', x))
    paper['Abstract_pre'] = paper['Abstract_pre'].map(lambda x: x.lower())
    paper['Abstract_pre'] = paper['Abstract_pre'].map(lambda x: re.sub('©.*', '', x))
          
         #===stopword removal===
    stop = stopwords.words('english')
    paper['Abstract_stop'] = paper['Abstract_pre'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
     
        #===lemmatize===
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(text):
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    paper['Abstract_lem'] = paper['Abstract_stop'].apply(lemmatize_words)
     
    topic_abs = paper.Abstract_lem.values.tolist()
    return topic_abs, paper

#===upload file===
@st.cache_data(ttl=3600)
def upload(file):
    papers = pd.read_csv(uploaded_file)
    return papers

@st.cache_data(ttl=3600)
def conv_txt(extype):
    col_dict = {'TI': 'Title',
            'SO': 'Source title',
            'DT': 'Document Type',
            'AB': 'Abstract',
            'PY': 'Year'}
    papers = pd.read_csv(uploaded_file, sep='\t', lineterminator='\r')
    papers.rename(columns=col_dict, inplace=True)
    return papers


#===Read data===
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'], on_change=reset_all)

if uploaded_file is not None:
    extype = get_ext(uploaded_file)

    if extype.endswith('.csv'):
         papers = upload(extype) 
    elif extype.endswith('.txt'):
         papers = conv_txt(extype)
          
    topic_abs, paper=clean_csv(extype)
    c1, c2 = st.columns([5,5])
    method = c1.selectbox(
            'Choose method',
            ('Choose...', 'pyLDA', 'Biterm', 'BERTopic'), on_change=reset_all)
    c1.info("Don't do anything during the computing", icon="⚠️") 
    num_cho = c2.number_input('Choose number of topics', min_value=2, max_value=30, value=2)
    if c2.button("Submit", on_click=reset_all):
         num_topic = num_cho  
           
    #===topic===
    if method == 'Choose...':
        st.write('')

    elif method == 'pyLDA':       
         tab1, tab2, tab3 = st.tabs(["📈 Generate visualization & Calculate coherence", "📃 Reference", "📓 Recommended Reading"])

         with tab1:
         #===visualization===
              @st.cache_data(ttl=3600, show_spinner=False)
              def pylda(extype):
                 topic_abs_LDA = [t.split(' ') for t in topic_abs]
                 id2word = Dictionary(topic_abs_LDA)
                 corpus = [id2word.doc2bow(text) for text in topic_abs_LDA]
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
     
                 #===visualization===
                 coherence_model_lda = CoherenceModel(model=lda_model, texts=topic_abs_LDA, dictionary=id2word, coherence='c_v')
                 coherence_lda = coherence_model_lda.get_coherence()
                 vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
                 py_lda_vis_html = pyLDAvis.prepared_data_to_html(vis)
                 return py_lda_vis_html, coherence_lda, vis
                   
              with st.spinner('Performing computations. Please wait ...'):
                   try:
                        py_lda_vis_html, coherence_lda, vis = pylda(extype)
                        st.write('Coherence: ', (coherence_lda))
                        st.components.v1.html(py_lda_vis_html, width=1500, height=800)
                        st.markdown('Copyright (c) 2015, Ben Mabey. https://github.com/bmabey/pyLDAvis')
                       
                        @st.cache_data(ttl=3600, show_spinner=False)
                        def img_lda(vis):
                             pyLDAvis.save_html(vis, 'output.html')
                             hti = Html2Image()
                             hti.browser.flags = ['--default-background-color=ffffff', '--hide-scrollbars']
                             css = "body {background: white;}"
                             hti.screenshot(
                                  other_file='output.html', css_str=css, size=(1500, 800),
                                  save_as='ldavis_img.png'
                             )
                             
                        img_lda(vis)   
                        with open("ldavis_img.png", "rb") as file:
                              btn = st.download_button(
                                  label="Download image",
                                  data=file,
                                  file_name="ldavis_img.png",
                                  mime="image/png"
                                  )
                       
                   except NameError:
                        st.warning('🖱️ Please click Submit')

         with tab2:
             st.markdown('**Sievert, C., & Shirley, K. (2014). LDAvis: A method for visualizing and interpreting topics. Proceedings of the Workshop on Interactive Language Learning, Visualization, and Interfaces.** https://doi.org/10.3115/v1/w14-3110')

         with tab3:
             st.markdown('**Chen, X., & Wang, H. (2019, January). Automated chat transcript analysis using topic modeling for library reference services. Proceedings of the Association for Information Science and Technology, 56(1), 368–371.** https://doi.org/10.1002/pra2.31')
             st.markdown('**Joo, S., Ingram, E., & Cahill, M. (2021, December 15). Exploring Topics and Genres in Storytime Books: A Text Mining Approach. Evidence Based Library and Information Practice, 16(4), 41–62.** https://doi.org/10.18438/eblip29963')
             st.markdown('**Lamba, M., & Madhusudhan, M. (2021, July 31). Topic Modeling. Text Mining for Information Professionals, 105–137.** https://doi.org/10.1007/978-3-030-85085-2_4')
             st.markdown('**Lamba, M., & Madhusudhan, M. (2019, June 7). Mapping of topics in DESIDOC Journal of Library and Information Technology, India: a study. Scientometrics, 120(2), 477–505.** https://doi.org/10.1007/s11192-019-03137-5')
     
     #===Biterm===
    elif method == 'Biterm':            
             
        #===optimize Biterm===
        @st.cache_data(ttl=3600, show_spinner=False)
        def biterm_topic(extype):
            X, vocabulary, vocab_dict = btm.get_words_freqs(topic_abs)
            tf = np.array(X.sum(axis=0)).ravel()
            docs_vec = btm.get_vectorized_docs(topic_abs, vocabulary)
            docs_lens = list(map(len, docs_vec))
            biterms = btm.get_biterms(docs_vec)
            model = btm.BTM(
              X, vocabulary, seed=12321, T=num_topic, M=20, alpha=50/8, beta=0.01)
            model.fit(biterms, iterations=20)
            p_zd = model.transform(docs_vec)
            coherence = model.coherence_
            phi = tmp.get_phi(model)
            topics_coords = tmp.prepare_coords(model)
            totaltop = topics_coords.label.values.tolist()
            return topics_coords, phi, totaltop

        tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading"])
        with tab1:
             try:
               with st.spinner('Performing computations. Please wait ...'): 
                    topics_coords, phi, totaltop = biterm_topic(extype)            
                    col1, col2 = st.columns([4,6])
                  
                    @st.cache_data(ttl=3600)
                    def biterm_map(extype):
                         btmvis_coords = tmp.plot_scatter_topics(topics_coords, size_col='size', label_col='label', topic=numvis)
                         return btmvis_coords
                            
                    @st.cache_data(ttl=3600)
                    def biterm_bar(extype):
                         terms_probs = tmp.calc_terms_probs_ratio(phi, topic=numvis, lambda_=1)
                         btmvis_probs = tmp.plot_terms(terms_probs, font_size=12)
                         return btmvis_probs
                            
                    with col1:
                         numvis = st.selectbox(
                              'Choose topic',
                              (totaltop), on_change=reset_biterm)
                         btmvis_coords = biterm_map(extype)
                         st.altair_chart(btmvis_coords)
                         png_data = vlc.vegalite_to_png(vl_spec=btmvis_coords, scale=2)
                         with open("chart.png", "wb") as f:
                             f.write(png_data)
                         image = Image.open("chart.png")

                         st.image(image, caption='Sunrise by the mountains')
                              
                         
                    with col2:
                         btmvis_probs = biterm_bar(extype)
                         st.altair_chart(btmvis_probs, use_container_width=True)

             #except ValueError:
                   #st.error('🙇‍♂️ Please raise the number of topics and click submit')
             except NameError:
                   st.warning('🖱️ Please click Submit')

        with tab2: 
            st.markdown('**Yan, X., Guo, J., Lan, Y., & Cheng, X. (2013, May 13). A biterm topic model for short texts. Proceedings of the 22nd International Conference on World Wide Web.** https://doi.org/10.1145/2488388.2488514')
        with tab3:
            st.markdown('**Cai, M., Shah, N., Li, J., Chen, W. H., Cuomo, R. E., Obradovich, N., & Mackey, T. K. (2020, August 26). Identification and characterization of tweets related to the 2015 Indiana HIV outbreak: A retrospective infoveillance study. PLOS ONE, 15(8), e0235150.** https://doi.org/10.1371/journal.pone.0235150')
            st.markdown('**Chen, Y., Dong, T., Ban, Q., & Li, Y. (2021). What Concerns Consumers about Hypertension? A Comparison between the Online Health Community and the Q&A Forum. International Journal of Computational Intelligence Systems, 14(1), 734.** https://doi.org/10.2991/ijcis.d.210203.002')
            st.markdown('**George, Crissandra J., "AMBIGUOUS APPALACHIANNESS: A LINGUISTIC AND PERCEPTUAL INVESTIGATION INTO ARC-LABELED PENNSYLVANIA COUNTIES" (2022). Theses and Dissertations-- Linguistics. 48.** https://doi.org/10.13023/etd.2022.217')
            st.markdown('**Li, J., Chen, W. H., Xu, Q., Shah, N., Kohler, J. C., & Mackey, T. K. (2020). Detection of self-reported experiences with corruption on twitter using unsupervised machine learning. Social Sciences & Humanities Open, 2(1), 100060.** https://doi.org/10.1016/j.ssaho.2020.100060')
          
     #===BERTopic===
    elif method == 'BERTopic':
        @st.cache_data(ttl=3600, show_spinner=False)
        def bertopic_vis(extype):
          topic_time = paper.Year.values.tolist()
          cluster_model = KMeans(n_clusters=num_topic)
          nlp = en_core_web_sm.load(exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
          topic_model = BERTopic(embedding_model=nlp, hdbscan_model=cluster_model, language="multilingual").fit(topic_abs)
          topics, probs = topic_model.fit_transform(topic_abs)
          return topic_model, topic_time, topics, probs
        
        @st.cache_data(ttl=3600, show_spinner=False)
        def Vis_Topics(extype):
          fig1 = topic_model.visualize_topics()
          return fig1
        
        @st.cache_data(ttl=3600, show_spinner=False)
        def Vis_Documents(extype):
          fig2 = topic_model.visualize_documents(topic_abs)
          return fig2

        @st.cache_data(ttl=3600, show_spinner=False)
        def Vis_Hierarchy(extype):
          fig3 = topic_model.visualize_hierarchy(top_n_topics=num_topic)
          return fig3
    
        @st.cache_data(ttl=3600, show_spinner=False)
        def Vis_Heatmap(extype):
          global topic_model
          fig4 = topic_model.visualize_heatmap(n_clusters=num_topic-1, width=1000, height=1000)
          return fig4

        @st.cache_data(ttl=3600, show_spinner=False)
        def Vis_Barchart(extype):
          fig5 = topic_model.visualize_barchart(top_n_topics=num_topic, n_words=10)
          return fig5
    
        @st.cache_data(ttl=3600, show_spinner=False)
        def Vis_ToT(extype):
          topics_over_time = topic_model.topics_over_time(topic_abs, topic_time)
          fig6 = topic_model.visualize_topics_over_time(topics_over_time)
          return fig6

        @st.cache_data(ttl=3600, show_spinner=False)
        def img_bert(fig):
            my_saved_image = "fig.png"
            fig.write_image(my_saved_image)
            return my_saved_image
        
        tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading"])
        with tab1:
          try:
               with st.spinner('Performing computations. Please wait ...'):
                    topic_model, topic_time, topics, probs = bertopic_vis(extype)
                    #===visualization===
                    viz = st.selectbox(
                      'Choose visualization',
                      ('Visualize Topics', 'Visualize Documents', 'Visualize Document Hierarchy', 'Visualize Topic Similarity', 'Visualize Terms', 'Visualize Topics over Time'))
          
                    if viz == 'Visualize Topics':
                           with st.spinner('Performing computations. Please wait ...'):
                                fig1 = Vis_Topics(extype)
                                st.write(fig1)
                                my_saved_image = img_bert(fig1)
                                with open(my_saved_image, "rb") as file:
                                  btn = st.download_button(
                                       label="Download image",
                                       data=file,
                                       file_name="Vis_Topics.png",
                                       mime="image/png"
                                       )
          
                    elif viz == 'Visualize Documents':
                           with st.spinner('Performing computations. Please wait ...'):
                                fig2 = Vis_Documents(extype)
                                st.write(fig2)
                                my_saved_image = img_bert(fig2)
                                with open(my_saved_image, "rb") as file:
                                  btn = st.download_button(
                                       label="Download image",
                                       data=file,
                                       file_name="Vis_Documents.png",
                                       mime="image/png"
                                       )
          
                    elif viz == 'Visualize Document Hierarchy':
                           with st.spinner('Performing computations. Please wait ...'):
                                fig3 = Vis_Hierarchy(extype)
                                st.write(fig3)
                                my_saved_image = img_bert(fig3)
                                with open(my_saved_image, "rb") as file:
                                  btn = st.download_button(
                                       label="Download image",
                                       data=file,
                                       file_name="Vis_Hierarchy.png",
                                       mime="image/png"
                                       )
          
                    elif viz == 'Visualize Topic Similarity':
                           with st.spinner('Performing computations. Please wait ...'):
                                fig4 = Vis_Heatmap(extype)
                                st.write(fig4)
                                my_saved_image = img_bert(fig4)
                                with open(my_saved_image, "rb") as file:
                                  btn = st.download_button(
                                       label="Download image",
                                       data=file,
                                       file_name="Vis_Similarity.png",
                                       mime="image/png"
                                       )
          
                    elif viz == 'Visualize Terms':
                           with st.spinner('Performing computations. Please wait ...'):
                                fig5 = Vis_Barchart(extype)
                                st.write(fig5)
                                my_saved_image = img_bert(fig5)
                                with open(my_saved_image, "rb") as file:
                                  btn = st.download_button(
                                       label="Download image",
                                       data=file,
                                       file_name="Vis_Terms.png",
                                       mime="image/png"
                                       )
          
                    elif viz == 'Visualize Topics over Time':
                           with st.spinner('Performing computations. Please wait ...'):
                                fig6 = Vis_ToT(extype)
                                st.write(fig6)
                                my_saved_image = img_bert(fig6)
                                with open(my_saved_image, "rb") as file:
                                  btn = st.download_button(
                                       label="Download image",
                                       data=file,
                                       file_name="Vis_ToT.png",
                                       mime="image/png"
                                       )
                    
          except ValueError:
               st.error('🙇‍♂️ Please raise the number of topics and click submit')
          
          except NameError:
               st.warning('🖱️ Please click Submit')

        with tab2:
          st.markdown('**Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.** https://doi.org/10.48550/arXiv.2203.05794')
          
        with tab3:
          st.markdown('**Jeet Rawat, A., Ghildiyal, S., & Dixit, A. K. (2022, December 1). Topic modelling of legal documents using NLP and bidirectional encoder representations from transformers. Indonesian Journal of Electrical Engineering and Computer Science, 28(3), 1749.** https://doi.org/10.11591/ijeecs.v28.i3.pp1749-1755')
          st.markdown('**Yao, L. F., Ferawati, K., Liew, K., Wakamiya, S., & Aramaki, E. (2023, April 20). Disruptions in the Cystic Fibrosis Community’s Experiences and Concerns During the COVID-19 Pandemic: Topic Modeling and Time Series Analysis of Reddit Comments. Journal of Medical Internet Research, 25, e45249.** https://doi.org/10.2196/45249')
