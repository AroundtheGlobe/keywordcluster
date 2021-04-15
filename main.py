import string
from collections import Counter
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
import plotly.express as px
from nltk import ngrams, collections
from collections import Counter
from itertools import chain
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import itertools

sw = stopwords.words('dutch')
sw.extend(['kun', 'nl', 'mooie', 'waar', 'ligt', 'welk', 'land', 'waar', 'staat', 'lang', 'duurt', 'wanneer', 'mag'])

# Import your data to a Pandas.DataFrame
df = pd.read_csv("202103.csv", dtype={'clicks': 'int', 'impressions': 'int', 'query': 'str'})
df = df.groupby(['query', 'match_queries', 'category']).agg({"clicks": "sum", "impressions": "sum"}).sort_values(["impressions"], ascending=False).reset_index(drop=False)
#df = df.drop('match_queries', 1)
df['category'] = df.category.astype('category')
df = df[df['impressions'] > 10]

#df = df[:100000]
# Instaniate our lookup hash table
group_lookup = {}

# Write a function for cleaning strings and returning an array of ngrams
def ngrams_analyzer(string):
    string = re.sub(r'[,-./]', r'', string)
    ngrams = zip(*[string[i:] for i in range(5)])  # N-Gram length is 5
    return [''.join(ngram) for ngram in ngrams]

def find_group(row, col):
    # If either the row or the col string have already been given
    # a group, return that group. Otherwise return none
    if row in group_lookup:
        return group_lookup[row]
    elif col in group_lookup:
        return group_lookup[col]
    else:
        return None

def add_vals_to_lookup(group, row, col):
    # Once we know the group name, set it as the value
    # for both strings in the group_lookup
    group_lookup[row] = group
    group_lookup[col] = group


def add_pair_to_lookup(row, col):
    # in this function we'll add both the row and the col to the lookup
    group = find_group(row, col)  # first, see if one has already been added
    if group is not None:
        # if we already know the group, make sure both row and col are in lookup
        add_vals_to_lookup(group, row, col)
    else:
        # if we get here, we need to add a new group.
        # The name is arbitrary, so just make it the row
        add_vals_to_lookup(row, row, col)

def token_counter(token):
    global lists
    no_of_lists_per_name = Counter(chain.from_iterable(map(set, lists)))

    for name, no_of_lists in no_of_lists_per_name.most_common():
        if no_of_lists == 2:
            break  # since it is ordered by count, once we get this low we are done
        if ''.join(token) == name:
            print(f"'{name}' is in {no_of_lists} lists")
            return pd.Series([name, no_of_lists], index=['cat','cat_count'])
    return pd.Series([], dtype=pd.StringDtype())

# Construct your vectorizer for building the TF-IDF matrix
vectorizer = TfidfVectorizer(analyzer=ngrams_analyzer, stop_words=sw)

# Grab the column you'd like to group, filter out duplicate values
# and make sure the values are Unicode
vals = df['query'].unique().astype('U')
# Build the matrix!!!
tf_idf_matrix = vectorizer.fit_transform(vals)

cosine_matrix = awesome_cossim_topn(tf_idf_matrix, tf_idf_matrix.transpose(), vals.size, 0.65)

# Build a coordinate matrix
coo_matrix = cosine_matrix.tocoo()

# for each row and column in coo_matrix
# if they're not the same string add them to the group lookup
for row, col in zip(coo_matrix.row, coo_matrix.col):
    if row != col:
        add_pair_to_lookup(vals[row], vals[col])

df['category_erik'] = df['query'].map(group_lookup).fillna(df['query'])
df['token'] = df['category_erik'].apply(word_tokenize)
df['token'] = df['token'].apply(lambda x: [item for item in x if item not in sw])

df['tokenstring'] = [' '.join(map(str, l)) for l in df['token']]

lists = df['token']
row_list = []
no_of_lists_per_name = Counter(chain.from_iterable(map(set, lists)))
for name, no_of_lists in no_of_lists_per_name.most_common():
    if no_of_lists == 1:
        break  # since it is ordered by count, once we get this low we are done
    row_list.append([name, no_of_lists])
df_cat_1 = pd.DataFrame(row_list, columns=['cat', 'cat_count'])


#print(Counter(list(ngrams(df['token'], 2))))
counts = collections.Counter()   # or nltk.FreqDist()
#for sent in df['token']:
#    counts.update(nltk.ngrams(sent, 2))

#print(counts)

for sent in df['token']:
    counts.update(" ".join(n) for n in nltk.ngrams(sent, 2))
df_cat_2 = pd.DataFrame.from_records(counts.most_common(), columns=['cat','cat_count'])
df_cat = df_cat_1.append(df_cat_2)
df = pd.merge(df, df_cat, left_on=["tokenstring"], right_on="cat")

df.to_csv('temp.csv', encoding='utf-8-sig', index=False)

df = df.groupby(['cat']).agg({"clicks": "sum", "impressions": "sum"}).sort_values(["impressions"], ascending=False).reset_index(drop=False)

fig = px.scatter(df[:50], x="impressions", y="clicks", size="impressions", color="cat", hover_name="cat", log_x=True, size_max=60)
fig.show()