import numpy as np
import pandas as pd
import heapq
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10,
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))
    return train, test


def rankPopularity(popularity):
    array = np.array(popularity)
    temp = array.argsort()
    ranks = np.empty(len(array), int)
    ranks[temp] = np.arange(len(array))
    return ranks

def normalize(popularity):
    minval = min(popularity)
    maxval = max(popularity)
    for i in range(len(popularity)):
        popularity[i] = (popularity[i] - minval)/(maxval - minval)
    return popularity
def PopularityMethod(ratings):
    popularity = []
    for i in range(len(ratings[0])):
        pop_count = 0
        for j in range(len(ratings)):
            if ratings[j][i]>0:
                pop_count += 1
        popularity.append(pop_count)

        ranks = rankPopularity(popularity)
    popularity = normalize(popularity)
    return popularity, ranks

def MatrixFactorization(R, P, Q, K, test, steps=13, alpha=0.002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
        print("Step ", step)
        MeanSquareError(R, np.dot(P, Q))
        MeanSquareError(test, np.dot(P,Q))
    return P, Q.T

def MeanSquareError(P, Q):
    loss = 0
    count = 0
    for i in range(len(P)):
        for j in range(len(P[i])):
            if P[i][j] > 0:
                loss += (P[i][j] - Q[i][j])* (P[i][j] - Q[i][j])
                count += 1
    print("Loss: ", loss/count)

def predictionMatrix(nR,R):
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] > 0:
                nR[i][j] = 0
    return nR

def TopNMoviesList(predictedR, user, N):
    return np.argsort(predictedR[user-1])[::-1][:N], heapq.nlargest(N, predictedR[user-1])


names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=names)
df.head()

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
ratings = np.zeros((n_users, n_items))
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
train, test = train_test_split(ratings)
popularity, rank = PopularityMethod(train)

def CompleteExecution(train, test, ratings, popularity, rank):
    N = n_users
    M = n_items
    K = 13
    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)
    nP, nQ = MatrixFactorization(train, P, Q, K, test)
    nR = np.dot(nP, nQ.T)
    user = 190
    predictedR = predictionMatrix(nR, ratings)
    TopN, TopNRatings = TopNMoviesList(predictedR, user, 5)
    print("Recommendations for the user: ", user)
    for i in range(len(TopN)):
        print(i + 1, ". Movie id: ", TopN[i] + 1, "| Popularity Score: ", popularity[TopN[i]], "| Popularity Rank: ", rank[TopN[i]])
    TopN, TopNRatings = TopNMoviesList(predictedR, user+105, 5)
    print("Recommendations for the user: ", user+105)
    for i in range(len(TopN)):
        print(i + 1, ". Movie id: ", TopN[i] + 1, "| Popularity Score: ", popularity[TopN[i]], "| Popularity Rank: ",
              rank[TopN[i]])
    TopN, TopNRatings = TopNMoviesList(predictedR, user - 55, 5)
    print("Recommendations for the user: ", user - 55)
    for i in range(len(TopN)):
        print(i + 1, ". Movie id: ", TopN[i] + 1, "| Popularity Score: ", popularity[TopN[i]], "| Popularity Rank: ",
              rank[TopN[i]])
    TopN, TopNRatings = TopNMoviesList(predictedR, 720, 5)
    print("Recommendations for the user: ", 720)
    for i in range(len(TopN)):
        print(i + 1, ". Movie id: ", TopN[i] + 1, "| Popularity Score: ", popularity[TopN[i]], "| Popularity Rank: ",
              rank[TopN[i]])


def diagFunction(a):
    a = np.array(a)
    return np.diag(a)


# def modifiedFunction(train, test, ratings, popularity):
#     diag_popularity = diagFunction(popularity)
#     newRatings = np.dot(ratings, diag_popularity)
#     newTrain = np.dot(train, diag_popularity)
#     newTest = np.dot(test, diag_popularity)
#     return newTrain, newTest, newRatings

def modifiedFunction(train, test, ratings, popularity):
    #diag_popularity = diagFunction(popularity)

    newRatings = changeMatrix(ratings, popularity)
    newTrain = changeMatrix(train,popularity)
    newTest = changeMatrix(test,popularity)
    return newTrain, newTest, newRatings

def changeMatrix(mat, popularity):
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if mat[i][j] > 0:
                mat[i][j] = 0.6*mat[i][j] + 0.2*popularity[j]
    return mat


CompleteExecution(train, test, ratings, popularity, rank)
mTrain, mTest, mRatings = modifiedFunction(train, test, ratings, popularity)

print("********************************")
CompleteExecution(mTrain, mTest, mRatings, popularity, rank)
pass