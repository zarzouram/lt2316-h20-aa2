# %% [markdown]
# The functions in these files are extracted from https://github.com/henrywoo/MyML/blob/master/Copy_of_nlu_2.ipynb

# I used the following functions from the repository mentioned above:
# 1. `co_occurrence_matrix(token_ids, V, K=2)`
# 2. `def PPMI(C)` with some minor modification
# 4. `def SVD(X, d)` with some minor modification

# %% [markdown]

# ## Copyright 2018 Google LLC.

# # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from sklearn.decomposition import TruncatedSVD


# %%
def cooccurrence_matrix(tokens_ids, V, K):
    # get sparse co-occurrence matrix given a corpus.
    # token_id is 1-D array or list of the ids that represents words in the corpus.
    # vocabulary size = V
    # the context window is +-K

    # initiate co-currence matrix
    C = csc_matrix((V,V), dtype=np.float32)
    for token_ids in tokens_ids:
        for k in range(1, K+1):
            i = token_ids[:-k]  # ids from sent begining till k
            j = token_ids[k:]   # word from k till end
            # counts = 1 at (i,j) ==> (current word, word k ahead)
            data = (np.ones_like(i), (i,j))  # counts, indices
            # format a matrix in C shape with ones at (i,j), zeros else where
            C_upper = coo_matrix(data, shape=C.shape, dtype=np.float32)
            C_upper = csc_matrix(C_upper)
            # consider k words behind. Upper triangular matrix = lower 
            C_lower = C_upper.T  
            C += C_upper + C_lower # cummulative counts
    return C

def PPMI(C):
    # Total count.
    Z = float(C.sum())
    # Sum each row (along columns).
    Zr = np.array(C.sum(axis=1), dtype=np.float64).flatten()
    # Get indices of relevant elements.
    ii, jj = C.nonzero()  # row, column indices
    Cij = np.array(C[ii,jj], dtype=np.float64).flatten()
    # PMI equation.
    pmi = np.log(Cij * Z / (Zr[ii] * Zr[jj]))
    # Truncate to positive only.
    ppmi = np.maximum(0, pmi)  # take positive only
    
    # Re-format as sparse matrix.
    ret = csc_matrix((ppmi, (ii,jj)), shape=C.shape,
                                  dtype=np.float64)
    ret.eliminate_zeros()  # remove zeros
    return ret

def SVD(X, d):
    svd = TruncatedSVD(n_components=d, random_state=42)
    Wv = svd.fit_transform(X)
    # Normalize all vectors to unit length.
    Wv = Wv / np.linalg.norm(Wv, axis=1).reshape([-1,1])
    variance_explained = svd.explained_variance_ratio_
    return Wv, variance_explained

def ppmi_embedding(corpus_tokens_id, V, K, d, test=0):
    distrib_matrix = cooccurrence_matrix(corpus_tokens_id, V, K)
    ppmi = PPMI(distrib_matrix)
    embeddings, vars_exp = SVD(ppmi.toarray(), d)
    
    d_test = [100, 500, 1000, 3000, 7000, 8000, 9000]
    if test:
        vars_sum = []
        for d_ in d_test:
            _, var_ = SVD(ppmi.toarray(), d_)
            vars_sum.append(var_.sum())
        return 0, vars_sum

    return embeddings, vars_exp