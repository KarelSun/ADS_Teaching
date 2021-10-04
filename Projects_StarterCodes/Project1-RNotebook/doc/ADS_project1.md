# ADS_Project1

# Abstract

In today's life, ideology has become an indispensable part of life. It is precisely because of the difference in ideology that our way of dealing with life and the way of thinking about problems are different. Therefore, the quantitative analysis of textual data on philosophy may be a topic worthy of consideration. This article will use python to ask questions about philosophers, philosophical schools, and the philosophical discourse itself and the possible relationships between them. And use the knowledge of data analysis with relevant data to give analysis and answers.


(Key Words: Philosophy, Data Analysis, Python)

# Introduction

We may wonder what the ideology is. There are a lot of different kinds of ideologies, including political, social, epistemological, and ethical. Recent analysis tends to posit that ideology is a 'coherent system of ideas' that rely on a few basic assumptions about reality that may or may not have any factual basis. Through this system, ideas become coherent, repeated patterns through the subjective ongoing choices that people make. These ideas serve as the seed around which further thought grows. The belief in an ideology can range from passive acceptance up to fervent advocacy. According to most recent analysis, ideologies are neither necessarily right nor wrong[1]. Based on these ideologies, they form kinds of philosophy schools and ideas.



### (1) Import Data
We use pandas to import data and then we show the data to see whether we can find some questions (some relations) to seek.


```python
import numpy as np
import pandas as pd
import random

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
from wordcloud import WordCloud, STOPWORDS


######################################################################################################################
data = pd.read_csv('data.csv')

data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>author</th>
      <th>school</th>
      <th>sentence_spacy</th>
      <th>sentence_str</th>
      <th>original_publication_date</th>
      <th>corpus_edition_date</th>
      <th>sentence_length</th>
      <th>sentence_lowered</th>
      <th>tokenized_txt</th>
      <th>lemmatized_str</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>What's new, Socrates, to make you leave your ...</td>
      <td>What's new, Socrates, to make you leave your ...</td>
      <td>-350</td>
      <td>1997</td>
      <td>125</td>
      <td>what's new, socrates, to make you leave your ...</td>
      <td>['what', 'new', 'socrates', 'to', 'make', 'you...</td>
      <td>what be new , Socrates , to make -PRON- lea...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>Surely you are not prosecuting anyone before t...</td>
      <td>Surely you are not prosecuting anyone before t...</td>
      <td>-350</td>
      <td>1997</td>
      <td>69</td>
      <td>surely you are not prosecuting anyone before t...</td>
      <td>['surely', 'you', 'are', 'not', 'prosecuting',...</td>
      <td>surely -PRON- be not prosecute anyone before ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>The Athenians do not call this a prosecution b...</td>
      <td>The Athenians do not call this a prosecution b...</td>
      <td>-350</td>
      <td>1997</td>
      <td>74</td>
      <td>the athenians do not call this a prosecution b...</td>
      <td>['the', 'athenians', 'do', 'not', 'call', 'thi...</td>
      <td>the Athenians do not call this a prosecution ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>What is this you say?</td>
      <td>What is this you say?</td>
      <td>-350</td>
      <td>1997</td>
      <td>21</td>
      <td>what is this you say?</td>
      <td>['what', 'is', 'this', 'you', 'say']</td>
      <td>what be this -PRON- say ?</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>Someone must have indicted you, for you are no...</td>
      <td>Someone must have indicted you, for you are no...</td>
      <td>-350</td>
      <td>1997</td>
      <td>101</td>
      <td>someone must have indicted you, for you are no...</td>
      <td>['someone', 'must', 'have', 'indicted', 'you',...</td>
      <td>someone must have indict -PRON- , for -PRON- ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>360803</th>
      <td>Women, Race, And Class</td>
      <td>Davis</td>
      <td>feminism</td>
      <td>But the socialization of housework including m...</td>
      <td>But the socialization of housework including m...</td>
      <td>1981</td>
      <td>1981</td>
      <td>142</td>
      <td>but the socialization of housework including m...</td>
      <td>['but', 'the', 'socialization', 'of', 'housewo...</td>
      <td>but the socialization of housework include me...</td>
    </tr>
    <tr>
      <th>360804</th>
      <td>Women, Race, And Class</td>
      <td>Davis</td>
      <td>feminism</td>
      <td>The only significant steps toward endingdomest...</td>
      <td>The only significant steps toward endingdomest...</td>
      <td>1981</td>
      <td>1981</td>
      <td>117</td>
      <td>the only significant steps toward endingdomest...</td>
      <td>['the', 'only', 'significant', 'steps', 'towar...</td>
      <td>the only significant step toward endingdomest...</td>
    </tr>
    <tr>
      <th>360805</th>
      <td>Women, Race, And Class</td>
      <td>Davis</td>
      <td>feminism</td>
      <td>Working women, therefore, have a special and v...</td>
      <td>Working women, therefore, have a special and v...</td>
      <td>1981</td>
      <td>1981</td>
      <td>90</td>
      <td>working women, therefore, have a special and v...</td>
      <td>['working', 'women', 'therefore', 'have', 'spe...</td>
      <td>working woman , therefore , have a special an...</td>
    </tr>
    <tr>
      <th>360806</th>
      <td>Women, Race, And Class</td>
      <td>Davis</td>
      <td>feminism</td>
      <td>Moreover, under capitalism, campaigns for jobs...</td>
      <td>Moreover, under capitalism, campaigns for jobs...</td>
      <td>1981</td>
      <td>1981</td>
      <td>199</td>
      <td>moreover, under capitalism, campaigns for jobs...</td>
      <td>['moreover', 'under', 'capitalism', 'campaigns...</td>
      <td>moreover , under capitalism , campaign for jo...</td>
    </tr>
    <tr>
      <th>360807</th>
      <td>Women, Race, And Class</td>
      <td>Davis</td>
      <td>feminism</td>
      <td>This strategy calls into question the validity...</td>
      <td>This strategy calls into question the validity...</td>
      <td>1981</td>
      <td>1981</td>
      <td>126</td>
      <td>this strategy calls into question the validity...</td>
      <td>['this', 'strategy', 'calls', 'into', 'questio...</td>
      <td>this strategy call into question the validity...</td>
    </tr>
  </tbody>
</table>
<p>360808 rows × 11 columns</p>
</div>



Based on the obeservation, the data does not exist garbled.

### (2) Data Information


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 360808 entries, 0 to 360807
    Data columns (total 11 columns):
     #   Column                     Non-Null Count   Dtype 
    ---  ------                     --------------   ----- 
     0   title                      360808 non-null  object
     1   author                     360808 non-null  object
     2   school                     360808 non-null  object
     3   sentence_spacy             360808 non-null  object
     4   sentence_str               360808 non-null  object
     5   original_publication_date  360808 non-null  int64 
     6   corpus_edition_date        360808 non-null  int64 
     7   sentence_length            360808 non-null  int64 
     8   sentence_lowered           360808 non-null  object
     9   tokenized_txt              360808 non-null  object
     10  lemmatized_str             360808 non-null  object
    dtypes: int64(3), object(8)
    memory usage: 30.3+ MB


From the imported data, we can see that the data is formed by title, author, school, sentence length, tokenized text and lemmatized_str. And there is no null data, hence we can think that the data is reliable to use without further processing.


```python
pd.DataFrame(data.groupby(by=['school','author'])['title'].count())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>title</th>
    </tr>
    <tr>
      <th>school</th>
      <th>author</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="7" valign="top">analytic</th>
      <th>Kripke</th>
      <td>12479</td>
    </tr>
    <tr>
      <th>Lewis</th>
      <td>13120</td>
    </tr>
    <tr>
      <th>Moore</th>
      <td>3668</td>
    </tr>
    <tr>
      <th>Popper</th>
      <td>4678</td>
    </tr>
    <tr>
      <th>Quine</th>
      <td>7373</td>
    </tr>
    <tr>
      <th>Russell</th>
      <td>5073</td>
    </tr>
    <tr>
      <th>Wittgenstein</th>
      <td>9034</td>
    </tr>
    <tr>
      <th>aristotle</th>
      <th>Aristotle</th>
      <td>48779</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">capitalism</th>
      <th>Keynes</th>
      <td>3411</td>
    </tr>
    <tr>
      <th>Ricardo</th>
      <td>3090</td>
    </tr>
    <tr>
      <th>Smith</th>
      <td>11693</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">communism</th>
      <th>Lenin</th>
      <td>4469</td>
    </tr>
    <tr>
      <th>Marx</th>
      <td>13489</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">continental</th>
      <th>Deleuze</th>
      <td>12540</td>
    </tr>
    <tr>
      <th>Derrida</th>
      <td>5999</td>
    </tr>
    <tr>
      <th>Foucault</th>
      <td>15240</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">empiricism</th>
      <th>Berkeley</th>
      <td>2734</td>
    </tr>
    <tr>
      <th>Hume</th>
      <td>8312</td>
    </tr>
    <tr>
      <th>Locke</th>
      <td>8885</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">feminism</th>
      <th>Beauvoir</th>
      <td>13017</td>
    </tr>
    <tr>
      <th>Davis</th>
      <td>3059</td>
    </tr>
    <tr>
      <th>Wollstonecraft</th>
      <td>2559</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">german_idealism</th>
      <th>Fichte</th>
      <td>5308</td>
    </tr>
    <tr>
      <th>Hegel</th>
      <td>22700</td>
    </tr>
    <tr>
      <th>Kant</th>
      <td>14128</td>
    </tr>
    <tr>
      <th>nietzsche</th>
      <th>Nietzsche</th>
      <td>13548</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">phenomenology</th>
      <th>Heidegger</th>
      <td>15239</td>
    </tr>
    <tr>
      <th>Husserl</th>
      <td>5742</td>
    </tr>
    <tr>
      <th>Merleau-Ponty</th>
      <td>7592</td>
    </tr>
    <tr>
      <th>plato</th>
      <th>Plato</th>
      <td>38366</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">rationalism</th>
      <th>Descartes</th>
      <td>1132</td>
    </tr>
    <tr>
      <th>Leibniz</th>
      <td>5027</td>
    </tr>
    <tr>
      <th>Malebranche</th>
      <td>12997</td>
    </tr>
    <tr>
      <th>Spinoza</th>
      <td>3793</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">stoicism</th>
      <th>Epictetus</th>
      <td>323</td>
    </tr>
    <tr>
      <th>Marcus Aurelius</th>
      <td>2212</td>
    </tr>
  </tbody>
</table>
</div>



And then we can see the composition of all the school including the author of the respective school.

# Method & Analysis

Our core method to analyze the data is to use EDA. **Exploratory data analysis (EDA)** is used by data scientists to analyze and investigate data sets and summarize their main characteristics, often employing data visualization methods[2]. In this part, we will elicit some questions first and then use EDA to give a direct analyis about the question with some explanation.

## Questions and Answer

## (a) Question 1: What is the popularity of ideological schools in the dataset ？

**The reason why the question is posted** is because from the raw data, we can see the different groups of ideologies. Therefore, we want to know which school, author or title is used most, which can help us to reflect current ideological mainstream to some degrees. In this section, we will use *bar plot* to show the output to analyze the question, which can intuitively and clearly reflect the number of distributions.


```python
# Distributions

#Title

plt.figure(figsize=(15,5))
data['title'].value_counts().plot(kind='bar', color='green')
plt.title('The distribution of the number of times the title has been cited')
plt.legend(['The number of times the title has been cited'])
plt.grid()
plt.show()
plt.savefig('1.png')
print("The group with maximum number", data['author'].value_counts().max()) # Most cited
print("The number of the mean:", data['title'].value_counts().mean())   # Average level
```


    
![png](output_12_0.png)
    


    The group with maximum number 48779
    The number of the mean: 6115.389830508475



    <Figure size 432x288 with 0 Axes>



```python
# color plot with exact number

df = data.groupby('title').size().reset_index(name='counts')
n = df['title'].unique().__len__()+1
all_colors = list(plt.cm.colors.cnames.keys())
random.seed(100)
c = random.choices(all_colors, k=n)

# Plot Bars
plt.figure(figsize=(20,10), dpi= 80)
plt.bar(df['title'], df['counts'], color=c, width=.5)
for i, value in enumerate(df['counts'].values):
    plt.text(i, value, float(value), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})

# Decoration
plt.title("Frequency by Titles", fontsize=20)
plt.xlabel('Title', fontsize=15)
plt.ylabel('Frequency')
plt.show()
plt.savefig('2.png')
```


    
![png](output_13_0.png)
    



    <Figure size 432x288 with 0 Axes>



```python
# Author
plt.figure(figsize=(15,5))
data['author'].value_counts().plot(kind='bar', color='blue')
plt.title('The distribution of the number of times the author has been cited')
plt.legend(['The number of times the author has been cited'])
plt.grid()
plt.savefig('3.png')
plt.show()

print("The group with maximum number", data['author'].value_counts().max()) # Most cited
print("The number of the mean:", data['author'].value_counts().mean())  # Average level
```


    
![png](output_14_0.png)
    


    The group with maximum number 48779
    The number of the mean: 10022.444444444445



```python
# color plot with exact number

df = data.groupby('author').size().reset_index(name='counts')
n = df['author'].unique().__len__()+1
all_colors = list(plt.cm.colors.cnames.keys())
random.seed(100)
c = random.choices(all_colors, k=n)

# Plot Bars
plt.figure(figsize=(20,10), dpi= 80)
plt.bar(df['author'], df['counts'], color=c, width=.5)
for i, value in enumerate(df['counts'].values):
    plt.text(i, value, float(value), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})

# Decoration
plt.title("Frequency by Authors", fontsize=20)
plt.xlabel('Author', fontsize=15)
plt.ylabel('Frequency')
plt.savefig('4.png')
plt.show()
```


    
![png](output_15_0.png)
    



```python
# School

plt.figure(figsize=(15,5))
data['school'].value_counts().plot(kind='bar', color='grey')
plt.title('The distribution of the number of times school has been cited')
plt.legend(['The number of times the school has been cited'])
plt.grid()
plt.savefig('5.png')
plt.show()

print("The group with maximum number", data['school'].value_counts().max()) # Most cited
print("The number of the mean:", data['school'].value_counts().mean())  # Average level
```


    
![png](output_16_0.png)
    


    The group with maximum number 55425
    The number of the mean: 27754.46153846154



```python
# color plots with exact number:

# Prepare Data
df = data.groupby('school').size().reset_index(name='counts')
n = df['school'].unique().__len__()+1
all_colors = list(plt.cm.colors.cnames.keys())
random.seed(100)
c = random.choices(all_colors, k=n)

# Plot Bars
plt.figure(figsize=(15,5), dpi= 80)
plt.bar(df['school'], df['counts'], color=c, width=.5)
for i, value in enumerate(df['counts'].values):
    plt.text(i, value, float(value), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})

# Decoration
plt.title("Frequency by Schools", fontsize=20)
plt.xlabel('Schools', fontsize=15)
plt.ylabel('Frequency')
plt.savefig('6.png')
plt.show()
```


    
![png](output_17_0.png)
    


From the distribution plot, we can see that the most popular ideology is about **Analytic--Aristotle--Aristotle (completed work)** .  Hence we may see that Analytic school and Aristotle is the most mainstream in the dataset.

## (b) Question 2: Which philosopher school is the most quite and which school likes to speak more ?

In this section, we may want to investigate the speaking length of philosophers, that is, we may want to find which school like to speak more and which school prefers to speak a liitle.

We will focus on the length of sentence and the token texts by school to do analysis and then deduce their speaking hlength, because the length of speaking/tokens is a significant feature of a person's speaking preference, and it can also reflect a person's personality to a certain extent.


```python
data['n_tokens'] = list(map(len,map(eval,data.tokenized_txt)))
```


```python
print(data.sentence_length.describe())

plt.figure(figsize=(12,5))
data.sentence_length.plot(kind='hist', bins=200, color='orange')
plt.title('Sentence Length')
plt.grid()
plt.savefig('7.png')
plt.show()
```

    count    360808.000000
    mean        150.790964
    std         104.822072
    min          20.000000
    25%          75.000000
    50%         127.000000
    75%         199.000000
    max        2649.000000
    Name: sentence_length, dtype: float64



    
![png](output_21_1.png)
    


We can roughly see the shape of the histogram, however, the graph is too skewed, hence we can do log to the variable.


```python
plt.figure(figsize=(12,5))
np.log10(data.sentence_length).plot(kind='hist', bins=50, color='pink')
plt.title('log(Sentence Length)') # Do log 10 here
plt.grid()
plt.savefig('8.png')
plt.show()

print("The mean of sentence length:", data.sentence_length.mean())
```


    
![png](output_23_0.png)
    


    The mean of sentence length: 150.79096361499745


Based on the graph, we can see that the max length is 2649 and the min length is 20, moreover, the mean is 150, which can be regarded as a standard to judge the sentence length.


After showing the data information about sentence length, we then analyze the data information about 'tokens'.


```python
print(data.n_tokens.describe())

plt.figure(figsize=(12,5))
data.n_tokens.plot(kind='hist', bins = 250)
plt.title('Number of Tokens')
plt.grid()
plt.savefig('9.png')
plt.show()
```

    count    360808.000000
    mean         25.693216
    std          17.766261
    min           0.000000
    25%          13.000000
    50%          22.000000
    75%          34.000000
    max         398.000000
    Name: n_tokens, dtype: float64



    
![png](output_25_1.png)
    


We then can find that the 'Number of Tokens' mainly concentrated when number is around 25, which is the mean value. And the frequency on both sides of the peak shows a decreasing trend.

And then we will investigate the number of sentence length and Tokens by school with violin and box plot.


```python
# violin plot sentence length split by school, violinplot

plt.figure(figsize=(13,10), dpi= 80)
sns.violinplot(x='school', y='sentence_length', data=data, scale='width', inner='quartile')
plt.title('Violin Plot of Sentence Length - By School', fontsize=20)
plt.grid()
plt.savefig('10.png')
plt.show()
```


    
![png](output_27_0.png)
    



```python
# plot sentence length split by school, boxplot

    
    # Draw Plot
plt.figure(figsize=(13,10), dpi= 80)
sns.boxplot(x='school', y='sentence_length', data=data)
sns.stripplot(x='school', y='sentence_length', data=data, color='black', size=3, jitter=1)
    
for i in range(len(data['school'].unique())-1):
    plt.vlines(i+.5, 10, 45, linestyles='solid', colors='gray', alpha=0.2)
    
    # Decoration
plt.title('Box Plot of sentence length by school', fontsize=20)
plt.savefig('11.png')
plt.show()
```


    
![png](output_28_0.png)
    



```python
# plot number of tokens split by school, violinplot


plt.figure(figsize=(13,10), dpi= 80)
sns.violinplot(x='school', y='n_tokens', data=data, scale='width', inner='quartile')
plt.title('Violin Plot of Number of Tokens - By School', fontsize=20)
plt.grid()
plt.savefig('12.png')
plt.show()
```


    
![png](output_29_0.png)
    



```python
plt.figure(figsize=(13,10), dpi= 80)
sns.boxplot(x='school', y='n_tokens', data=data, notch=False)
    
    # Add N Obs inside boxplot (optional)
def add_n_obs(data,group_col,y):
    medians_dict = {grp[0]:grp[1][y].median() for grp in data.groupby(group_col)}
    xticklabels = [x.get_text() for x in plt.gca().get_xticklabels()]
    n_obs = data.groupby(group_col)[y].size().values
    for (x, xticklabel), n_ob in zip(enumerate(xticklabels), n_obs):
        plt.text(x, medians_dict[xticklabel]*1.01, "#obs : "+str(n_ob), horizontalalignment='center', fontdict={'size':14}, color='white')
    
    add_n_obs(data,group_col='school',y='n_tokens')    
    
    # Decoration
plt.title('Box Plot of n_tokens length by school', fontsize=20)
plt.ylim(0, 410)
plt.savefig('13.png')
plt.show()

```


    
![png](output_30_0.png)
    


From the violin and box graph, on the one hand, we can find that no matter numbers of tokens and sentence length, **continental** tends to be the most, which reflects that the school of *continental* prefers to speak or cite more. On the other hand, **plato school** is the most quite class among all schools, which may reflect that they would like to consider more but speak less.

## (c) Question 3: What are the speaking habits of philosophers from different schools ?  

In this section, we will investigate which words the philosophers like to speak most by school.


We firstly use wordcloud to see the words each school used. The bigger the words on the graph is, the more frequency the words are used.

### c-1. Words that different genres like to say.


```python
schools = data.school.unique().tolist()

stopwords = set(STOPWORDS)
for sc in schools:
    data_temp = data[data.school == sc]
    
    print('School = ', sc.upper(), ':')
    
    text = " ".join(txt for txt in data_temp.sentence_lowered)
    wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,
                          width = 600, height = 400,
                          background_color="white").generate(text)
    
    plt.figure(figsize=(12,8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('14.png')
    plt.show()

```

    School =  PLATO :



    
![png](output_34_1.png)
    


    School =  ARISTOTLE :



    
![png](output_34_3.png)
    


    School =  EMPIRICISM :



    
![png](output_34_5.png)
    


    School =  RATIONALISM :



    
![png](output_34_7.png)
    


    School =  ANALYTIC :



    
![png](output_34_9.png)
    


    School =  CONTINENTAL :



    
![png](output_34_11.png)
    


    School =  PHENOMENOLOGY :



    
![png](output_34_13.png)
    


    School =  GERMAN_IDEALISM :



    
![png](output_34_15.png)
    


    School =  COMMUNISM :



    
![png](output_34_17.png)
    


    School =  CAPITALISM :



    
![png](output_34_19.png)
    


    School =  STOICISM :



    
![png](output_34_21.png)
    


    School =  NIETZSCHE :



    
![png](output_34_23.png)
    


    School =  FEMINISM :



    
![png](output_34_25.png)
    


We can then find the words for each school likes to speak more:

* PLATO : will, thing, one, socrates...
* ARISTOTLE : thing, animal, case...
* EMPIRICISM : principle, relation, mind...
* RATIONLISM : reason, cause, sensation...
* ANALYTIC : propostion, statement, case...
* CONTINENTAL: difference, madness, representation...
* PHENOMENOLOGY : consciousness, perception, object...
* GERMAN_IDEALISM : consciousness, determination, concept...
* COMMUNISM : capital, commodity, labour, value...
* CAPITALISM : price, money, country, employment...
* STOICISM : thing, thou, doth...
* NIETZSCHE : even, man, zarathustra...
* FEMINISM : women, will, marriage...

From their often used words, we can see their characters to some extents. For example, *Capitalism* likes to mention "employment", "money" , however *communism* prefers to talk about labour, commodity.


In addition, the frequency of used words among all the schools is also an important point to figure out the habits of philosophers.

### c-2. Words that all genres like to say


```python
from numpy import *
import operator

dic = {}

for i in data['tokenized_txt']:
    dic1 = eval(i)  # Transfer format
    for j in dic1:
        if j not in dic:
            dic[j] = 1
        else: 
            dic[j] = dic[j] + 1

output = sorted(dic.items(),key=operator.itemgetter(1),reverse=True)          
part_output = sorted(dic.items(),key=operator.itemgetter(1),reverse=True)[:300]
print(part_output)
```

    [('the', 660444), ('of', 422626), ('and', 271548), ('to', 260476), ('is', 235136), ('in', 220195), ('that', 175098), ('it', 153145), ('as', 106479), ('be', 94540), ('for', 86780), ('not', 83436), ('which', 81708), ('this', 79114), ('are', 72531), ('or', 67067), ('but', 66023), ('we', 60538), ('by', 60455), ('with', 53765), ('one', 48216), ('from', 47132), ('they', 45628), ('have', 45531), ('all', 42813), ('if', 40284), ('an', 39859), ('its', 39817), ('he', 39492), ('on', 38719), ('what', 37370), ('at', 34045), ('their', 33295), ('so', 32381), ('has', 30998), ('other', 30839), ('only', 30238), ('them', 28125), ('there', 27979), ('will', 27960), ('no', 27320), ('can', 27124), ('his', 27010), ('being', 26782), ('would', 24171), ('itself', 23896), ('was', 23866), ('more', 23498), ('when', 23162), ('these', 22890), ('same', 22236), ('such', 22102), ('any', 22071), ('must', 21140), ('who', 20703), ('our', 20647), ('than', 20461), ('you', 19642), ('do', 18738), ('then', 18580), ('into', 18123), ('some', 17727), ('things', 17482), ('those', 16881), ('also', 16781), ('may', 16298), ('man', 16152), ('does', 15367), ('time', 15307), ('been', 15280), ('because', 15165), ('us', 15100), ('even', 14461), ('were', 14366), ('first', 13880), ('should', 13051), ('now', 13041), ('my', 12918), ('say', 12856), ('way', 12854), ('nature', 12796), ('two', 12474), ('about', 12240), ('world', 12088), ('therefore', 11722), ('reason', 11464), ('had', 11462), ('without', 11432), ('own', 11333), ('thing', 11275), ('him', 11243), ('something', 10841), ('thus', 10482), ('another', 10466), ('very', 10378), ('most', 10336), ('out', 10302), ('like', 10265), ('good', 10214), ('through', 10090), ('since', 10011), ('part', 9838), ('case', 9596), ('each', 9459), ('true', 9435), ('just', 9422), ('her', 9422), ('both', 9354), ('between', 9342), ('every', 9336), ('me', 9313), ('form', 9262), ('sense', 9255), ('upon', 9181), ('how', 9178), ('body', 9047), ('cannot', 8976), ('men', 8918), ('object', 8848), ('different', 8828), ('could', 8716), ('up', 8639), ('make', 8605), ('said', 8514), ('she', 8510), ('possible', 8432), ('well', 8354), ('always', 8334), ('nothing', 8279), ('knowledge', 8117), ('fact', 8051), ('certain', 7931), ('here', 7899), ('see', 7839), ('many', 7723), ('yet', 7700), ('general', 7682), ('know', 7602), ('truth', 7599), ('whole', 7581), ('means', 7551), ('however', 7519), ('life', 7496), ('think', 7466), ('thought', 7434), ('much', 7431), ('order', 7431), ('themselves', 7419), ('self', 7367), ('great', 7321), ('others', 7278), ('either', 7261), ('present', 7250), ('power', 7235), ('though', 7207), ('where', 7170), ('nor', 7117), ('whether', 7091), ('god', 7045), ('given', 7016), ('idea', 6932), ('state', 6921), ('still', 6920), ('mind', 6897), ('people', 6881), ('made', 6773), ('kind', 6651), ('might', 6611), ('existence', 6603), ('too', 6590), ('ideas', 6414), ('place', 6392), ('use', 6376), ('before', 6371), ('value', 6337), ('particular', 6318), ('never', 6315), ('relation', 6233), ('point', 6206), ('question', 6182), ('matter', 6152), ('himself', 6145), ('natural', 6136), ('far', 6116), ('over', 6004), ('necessary', 5985), ('subject', 5953), ('rather', 5919), ('after', 5907), ('consciousness', 5871), ('concept', 5848), ('experience', 5730), ('within', 5727), ('take', 5702), ('less', 5691), ('while', 5569), ('labour', 5524), ('under', 5500), ('right', 5471), ('called', 5441), ('again', 5425), ('why', 5387), ('cause', 5343), ('already', 5298), ('parts', 5201), ('hand', 5167), ('law', 5131), ('against', 5114), ('objects', 5114), ('soul', 5106), ('human', 5033), ('let', 4989), ('new', 4950), ('come', 4932), ('shall', 4929), ('work', 4885), ('become', 4872), ('according', 4830), ('give', 4754), ('number', 4724), ('am', 4687), ('merely', 4666), ('end', 4634), ('words', 4621), ('language', 4588), ('difference', 4576), ('makes', 4565), ('principle', 4434), ('pure', 4408), ('real', 4355), ('find', 4354), ('motion', 4347), ('seems', 4288), ('meaning', 4251), ('movement', 4198), ('did', 4185), ('anything', 4173), ('example', 4140), ('perhaps', 4128), ('universal', 4117), ('love', 4099), ('understanding', 4099), ('common', 4085), ('else', 4079), ('contrary', 4073), ('long', 4068), ('indeed', 4058), ('your', 4050), ('view', 4024), ('having', 4019), ('neither', 4005), ('greater', 4004), ('further', 4001), ('latter', 3986), ('women', 3981), ('everything', 3969), ('together', 3940), ('terms', 3876), ('person', 3869), ('woman', 3839), ('found', 3834), ('above', 3831), ('mean', 3817), ('clear', 3776), ('money', 3764), ('name', 3756), ('among', 3742), ('word', 3734), ('whose', 3729), ('little', 3724), ('laws', 3714), ('call', 3707), ('science', 3666), ('essence', 3663), ('taken', 3642), ('least', 3605), ('second', 3602), ('simple', 3587), ('animals', 3568), ('once', 3566), ('able', 3541), ('produce', 3537), ('theory', 3515), ('cases', 3500), ('longer', 3495), ('course', 3482), ('capital', 3472), ('sort', 3451), ('often', 3418), ('hence', 3418), ('unity', 3393), ('becomes', 3389), ('system', 3376), ('bodies', 3315), ('individual', 3294), ('free', 3288), ('pleasure', 3288), ('better', 3279), ('property', 3279), ('ever', 3274)]


And then we will use plot to show the part of output.


```python
plt.figure(figsize=(30,30))
data_frame = pd.DataFrame(data=output[0:19],columns=["words","Frequency"])
data_frame.plot(x = "words", y = "Frequency", kind='bar', color='red')
plt.savefig('15.png')
plt.show()
```


    <Figure size 2160x2160 with 0 Axes>



    
![png](output_39_1.png)
    


We can see that the most frequent words are all Pronouns, conjunctions and prepositions or Copula and Modal verbs, which is a lack of meaning. Hence by obeservation, we find when around 150th frequency, the words begin to have characters.



```python
data_frame = pd.DataFrame(data=output[150:180],columns=["words","Frequency"])
data_frame.plot(x = "words", y = "Frequency", kind='bar', color = "gray")
plt.figure(figsize=(30,30))
plt.savefig('16.png')
plt.show()
```


    
![png](output_41_0.png)
    



    <Figure size 2160x2160 with 0 Axes>



```python
data_frame = pd.DataFrame(data=output[0:20],columns=["words","Frequency"])
data_frame.plot(x="words", y = "Frequency", kind='pie')
plt.savefig('17.png')
```


    
![png](output_42_0.png)
    


We also use pie chart to show the Proportion about the top 20 used words.


From the graph and print output, we can see that except Pronouns, conjunctions and prepositions or Copula and Modal verbs, philosophers also like to say ***power***, ***god***, ***mind***, ***people***, and ***existent*** and so on.

# Conclusion

1. We find that **Analytic--Aristotle--Aristotle (completed work)**  may be the most popular school/author/title, because its frequence is ranked first among all the school.

2. We find that **continental** tends to speak the most but **plato** school is the most quite school. 

3. Every school has its unique speaking preference and prefered words, but they all may prefer to mention about people, god, mind, power and existent and so on.

# Development

1. In this section, we do not analyze the conclusion with its background, that is, we do not link the data output with the real world background.
2. There is no in-depth exploration of what factors can affect the length of speech and other issues.
3. For c-2 parts, it is possible that some schools dominate a word that may not be used by all schools.

# Reference

[1] [https://en.wikipedia.org/wiki/Ideology]

[2] [https://www.ibm.com/cloud/learn/exploratory-data-analysis]

