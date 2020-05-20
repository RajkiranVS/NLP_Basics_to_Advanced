
# coding: utf-8

# In[1]:


def load_text(path):
    with open(path) as f:
        document = f.read()
    return document


# In[2]:


doc = load_text('Large_Text.text')


# In[3]:


import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[4]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[5]:


import warnings
warnings.filterwarnings("ignore")


# In[6]:


import spacy


# In[7]:


nlp = spacy.load('en_core_web_lg')


# In[8]:


from pprint import pprint


# In[9]:


doc = nlp(doc)


# In[10]:


pprint([(X.text, X.label_) for X in doc.ents])


# In[11]:


sentences = [x for x in doc.sents]
print(sentences[:10])


# In[12]:


from spacy import displacy


# In[13]:


displacy.render(nlp(str(sentences[:10])), jupyter=True, style='ent')


# In[14]:


[(x.orth_,x.pos_, x.lemma_) for x in [y 
                                      for y
                                      in nlp(str(sentences[:10])) 
                                      if not y.is_stop and y.pos_ != 'PUNCT']]


# In[15]:


dict([(str(x), x.label_) for x in nlp(str(sentences[:10])).ents])


# In[16]:


displacy.render(nlp(str(sentences)), jupyter=True, style='ent')


# In[17]:


from wordcloud import WordCloud, STOPWORDS


# In[18]:


stopwords = set(STOPWORDS)


# In[19]:


wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stopwords, 
                min_font_size = 10).generate(str(sentences[:25])) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

