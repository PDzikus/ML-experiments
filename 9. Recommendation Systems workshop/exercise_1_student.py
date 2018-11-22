
# coding: utf-8

# In[121]:


from urllib import request
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.version.version


# Accessing data:

# In[122]:


DATASET_URL = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
DATASET_ARCHIVE = 'ml-100k.zip'

request.urlretrieve(DATASET_URL, DATASET_ARCHIVE)
with zipfile.ZipFile(DATASET_ARCHIVE) as archive:
    archive.extractall()


# # Exploratory Analysis 

# Check readme file

# This data set consists of:
#    * 100,000 ratings (1-5) from 943 users on 1682 movies. 
#    * Each user has rated at least 20 movies. 

# In[123]:


users_num = 943
movies_num = 1682


# How our data looks?

# In[124]:


df = pd.read_csv('ml-100k/u.data', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], 
                 delim_whitespace=True)


# In[125]:


df.head()


# Check if every pair (user, item) appears only once

# In[126]:


df.groupby(['user_id','item_id']).count().reset_index()['timestamp'].max()


# In[127]:


df.duplicated(subset = ['user_id', 'item_id', 'rating']).count()


# Check for nan

# In[128]:


df['rating'].unique()
df.isnull().sum()


# Ratings distribution

# In[129]:


df['rating'].plot.hist(xticks = [1, 2, 3, 4, 5])


# In[130]:


# plt.hist(df['rating'], bins=5)


# Choose one movie from u.info file and check its ratings

# In[131]:


df[df.item_id==168]['rating'].plot.hist(xticks = [1, 2, 3, 4, 5])


# #### Sparsity
# calculate the number of movies each user rated

# In[132]:


grouped = df[['user_id', 'item_id']].groupby('user_id').count().rename(columns = {'item_id' : 'rating_count'}).sort_values('rating_count', ascending=False)
grouped.head()


# In[133]:


grouped['sparsity'] = 1.0 - grouped.rating_count / float(movies_num)
grouped.head()


# In[134]:


np.average(grouped['sparsity'])


# In[135]:


grouped['sparsity'].plot.hist(xticks = [i*0.05 for i in range(1,21)])


# In[136]:


grouped['sparsity'].plot.hist(xticks = [i*0.1 for i in range(10)])


# Reindex ids to numpy's like

# In[137]:


[(df[x].min(),df[x].max()) for  x in ['user_id', 'item_id']]


# In[138]:


for col in  ['user_id', 'item_id']:
    df[col] = df[col]-1


# In[139]:


[(df[x].min(),df[x].max()) for  x in ['user_id', 'item_id']]


# # User-User CF

# Firstly we want to have our data in form of matrix, where rows coresponds to users, columns to items and values to ratings

# In[140]:


ratings_matrix = np.matrix(df.pivot(index = 'user_id', columns = 'item_id', values = 'rating').fillna(0).values)
ratings_matrix


# Similarity functions:
# * cosine similarity
# * Pearson correlation

# In[141]:


def cosine_similarity_distance(M):
# insert your code here ~ 8-10 lines
    n,k = M.shape
    distance_matrix = np.matrix(np.zeros(n*n))
    distance_matrix.shape = (n, n)
    for a in range(n):
        for b in range(n):
            vec_sum = float(np.count_nonzero(M[a].A1+M[b].A1))
            if vec_sum == 0:
                distance_matrix[a,b] = 1
            else:
                distance_matrix[a,b] = 1 - np.count_nonzero(M[a].A1 * M[b].A1) / vec_sum
            
    return distance_matrix # n x n matrix, where M is n x k matrix


# In[142]:


distance_matrix = cosine_similarity_distance(ratings_matrix)


# In[143]:


distance_matrix


# In[144]:


from scipy.stats.stats import pearsonr

def scipy_pearson_similarity_distance(M):
# insert your code here ~ 8-10 lines
    n,k = M.shape
    distance_matrix = np.matrix(np.zeros(n*n))
    distance_matrix.shape = (n, n)
    for a in range(n):
        for b in range(n):
            vec_sum = float(np.count_nonzero(M[a].A1+M[b].A1))
            if vec_sum == 0:
                distance_matrix[a,b] = 1
            else:
                distance_matrix[a,b] = 1 - pearsonr(M[a].A1, M[b].A1)[0]
    return distance_matrix # n x n matrix, where M is n x k matrix


# In[145]:


distance_matrix_pearson = scipy_pearson_similarity_distance(ratings_matrix)
distance_matrix_pearson


# k- nearest neighboors

# In[146]:


def knn(ratings_matrix, k, similarity_function):
    #insert your code here ~ 3 lines
    dist_matrix = similarity_function(ratings_matrix) # obliczamy macierz odległości
    idx = np.argsort(dist_matrix)[:,1:k+1] # sortujemy każdy wiersz
    # na pierwszym miejscu zawsze jest user sam do siebie, bo jest najbliżej
    sorted_dist = np.take_along_axis(dist_matrix, idx, axis = 1)
    return idx, sorted_dist # two n x k matrix


# In[147]:


# let's test it
neighbors, neighbors_dist = knn(ratings_matrix, 3, cosine_similarity_distance)
neighbors


# In[148]:


neighbors_dist


# In[149]:


def calculate_recommendation(ratings_matrix, neighbors, distances):
     #insert your code here ~ 8-12 lines
    u,k = neighbors.shape
    p = ratings_matrix.shape[1]
    recommendation_matrix = np.matrix(np.zeros(u*p))
    recommendation_matrix.shape = (u,p)
    for i in range(u):
        for j in range(p):
            numerator = np.sum((1-distances[i, :].A1) * ratings_matrix[neighbors[i],j].A1)
            denominator = np.sum(1 - distances[i,:].A1) # similarity = 1 - distance
            if denominator == 0:
                recommendation_matrix[i,j] = 0
            else:
                recommendation_matrix[i,j] = numerator / float(denominator)
    return recommendation_matrix # same shape as ratings_matrix filled with our prediction


# In[150]:


recommendation_matrix = calculate_recommendation(ratings_matrix, neighbors, neighbors_dist)


# In[151]:


recommendation_matrix
# niektóre wartości są zerami, bo mamy bardzo mały zasięg sąsiedztwa (3) - niektórzy nasi sąsiedzi nie widzieli
# jeszcze filmów, więc nie możemy ich ocenić.


# We need indicator for existing ratings

# In[152]:


rating_ind = np.matrix(ratings_matrix != 0, dtype=float)
rating_ind


# In[153]:


def get_recommendation(recommendation_matrix, rating_ind, n):
    # insert your code here ~ 2 lines
    X = np.multiply(recommendation_matrix, 1-rating_ind) # mnozymy razy odwrotność macieszy rating_ind -chcemy tylko rekomendacje dla tych filmów, które nie mają jeszcze ratingu
    recommendation = np.argsort(-X)[:, :n] # sortujemy rekomendacje i wybieramy pierwsze n. sorty zawsze jest rosnący, dlatego -X
    ratings = np.take_along_axis(recommendation_matrix, recommendation, axis = 1)
    return recommendation, ratings # n recommendations for every user and estimated ratings


# In[154]:


recommendation, est_ratings = get_recommendation(recommendation_matrix, rating_ind, 5)
recommendation


# In[155]:


est_ratings


# # Item-Item CF

# neighbors i distance muszą być teraz dla Items a nie users. Trzeba zrobić transpozycję tablicy ratingów, albo zmienić pivot (zamienić index i columns)

# What we have to change?

# In[156]:


ratings_item_matrix = np.matrix(df.pivot(index = 'item_id', columns = 'user_id', values = 'rating').fillna(0).values)
ratings_item_matrix


# In[157]:


def calculate_recommendation_item_based(ratings_matrix, item_neighbors, item_distances):
    #insert your code here ~ 8-12 lines (most code can be pasted from previous calculate_recommendation)
    p,k = item_neighbors.shape
    u = ratings_matrix.shape[0]
    recommendation_matrix = np.matrix(np.zeros(u*p))
    recommendation_matrix.shape = (u,p)
    for i in range(u):
        for j in range(p):
            numerator = np.sum((1-item_distances[i, :].A1) * ratings_matrix[item_neighbors[i],j].A1)
            denominator = np.sum(1 - item_distances[i,:].A1) # similarity = 1 - distance
            if denominator == 0:
                recommendation_matrix[i,j] = 0
            else:
                recommendation_matrix[i,j] = numerator / float(denominator)
    
    return recommendation_matrix


# In[158]:


item_neighbors, item_distances = knn(ratings_item_matrix, 3, cosine_similarity_distance)
item_neighbors


# # Testing

# In[159]:


train_1 = pd.read_csv('ml-100k/u1.base', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], 
                 delim_whitespace=True)
test_1 = pd.read_csv('ml-100k/u1.test', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], 
                 delim_whitespace=True)


# In[160]:


for col in  ['user_id', 'item_id']:
    train_1[col] = train_1[col]-1
    test_1[col] = test_1[col]-1


# In[161]:


train_1_ratings_matrix = np.matrix(pd.crosstab(index=pd.Categorical(train_1['user_id'],categories = [i for i in range(943)]), 
                              columns=pd.Categorical(train_1['item_id'],categories = [i for i in range(1682)]),
                             values=train_1['rating'], aggfunc=np.sum, dropna= False).fillna(0).values)
test_1_ratings_matrix = np.matrix(pd.crosstab(index=pd.Categorical(test_1['user_id'],categories = [i for i in range(943)]), 
                              columns=pd.Categorical(test_1['item_id'],categories = [i for i in range(1682)]),
                             values=test_1['rating'], aggfunc=np.sum, dropna= False).fillna(0).values)


# In[162]:


train_1_if_filled = np.matrix(train_1_ratings_matrix != 0, dtype = float)
test_1_if_filled = np.matrix(test_1_ratings_matrix != 0, dtype = float)


# Check if train + test is equal to full data_set

# In[163]:


np.all(train_1_if_filled + test_1_if_filled == (ratings_matrix != 0))


# In[164]:


neighbors, distances = knn(train_1_ratings_matrix, 3, cosine_similarity_distance)


# In[165]:


recommendation_matrix = calculate_recommendation(train_1_ratings_matrix, neighbors, distances)


# In[171]:


recommendation_matrix


# ### RMSE

# In[166]:


# teraz obliczymy MSE, na ile się pomyliliśmy jeśli chodzi o rekomendacje
def rmse(ratings_matrix,recommendation_matrix, test_ind):
    #insert your code here ~ 2-4 lines
    X = np.multiply((ratings_matrix - recommendation_matrix), test_ind)
    Y = np.sum(np.multiply(X,X))
    Z = np.sum(test_ind)
    output = np.sqrt(Y/Z)
    return output
    


# In[170]:


X = rmse(ratings_matrix, recommendation_matrix, test_1_if_filled)
X


# # Validation

# In[168]:


def cross_val_testing(ratings_matrix):
    results =[]
    for i in range(1,6):
        if(i!=5):
            continue
        train = pd.read_csv('ml-100k/u{}.base'.format(i), header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], 
                             delim_whitespace=True)
        test = pd.read_csv('ml-100k/u{}.test'.format(i), header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], 
                             delim_whitespace=True)

        for col in  ['user_id', 'item_id']:
            train[col] = train[col]-1
            test[col] = test[col]-1
        
        train_ratings_matrix = np.matrix(pd.crosstab(index=pd.Categorical(train['user_id'],categories = [i for i in range(943)]), 
                                                    columns=pd.Categorical(train['item_id'],categories = [i for i in range(1682)]),
                                                    values=train['rating'], aggfunc=np.sum, dropna= False).fillna(0).values)
        test_ratings_matrix = np.matrix(pd.crosstab(index=pd.Categorical(test['user_id'],categories = [i for i in range(943)]), 
                                                    columns=pd.Categorical(test['item_id'],categories = [i for i in range(1682)]),
                                                    values=test['rating'], aggfunc=np.sum, dropna= False).fillna(0).values)
        
        train_ratings_item_matrix = np.matrix(pd.crosstab(index=pd.Categorical(train['item_id'],categories = [i for i in range(1682)]), 
                                                    columns=pd.Categorical(train['user_id'],categories = [i for i in range(943)]),
                                                    values=train['rating'], aggfunc=np.sum, dropna= False).fillna(0).values)       
        
        train_ind = np.matrix(train_ratings_matrix!=0, dtype=float)
        test_ind = np.matrix(test_ratings_matrix!=0, dtype=float)
        

        
        for function_name,dist_function in zip(['cosine','pearson'],[cosine_similarity_distance, scipy_pearson_similarity_distance]):
            for k in [3,7]:
                for cf_type,rec_fun in zip(['u-u','i-i'],[calculate_recommendation, calculate_recommendation_item_based]):                       
                    neighbors, distances = knn(train_ratings_matrix, k, dist_function) if cf_type =='u-u'                                            else knn(train_ratings_item_matrix, k, dist_function)
                    recommendation_matrix = rec_fun(train_ratings_matrix, neighbors, distances)
                    rmse_val = rmse(ratings_matrix, recommendation_matrix, test_ind)
                    results.append([i,cf_type,function_name,k, rmse_val])
                    print ([i,cf_type,function_name,k, rmse_val])
        
        
    return results


# In[169]:


result = cross_val_testing(ratings_matrix)


# In[ ]:


result

