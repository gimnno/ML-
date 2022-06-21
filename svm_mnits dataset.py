

"" following program performs SVM algorithm of machine learning for the dataset Mnits  ""

#%% Loading the data
import tensorflow as tf
from sklearn.model_selection import train_test_split
# store dataset from keras
mnist = tf.keras.datasets.mnist
# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()

import numpy as np 
# Convert to float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1]
x_train, x_test = x_train / 255., x_test / 255.


# maping the labels and make an array to use Quantitative evaluation by ConfusionMatrixDisplay 
import numpy as np 
labels_map = { 0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}

# packing classes names together in a array  
target_names =  np.array(list(labels_map.items()))[:,1]

############ spliting the data in train test 
#%% spliting 

#   Ration initializing 
DATASET_SIZE = 70000
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3

#   Join a sequence of arrays along an existing axis
X = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test])

#   Split using train_test_split 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=(1-TRAIN_RATIO)) 

print(" x_train: ", np.shape(x_train),
      " y_train: ",np.shape(y_train), "\n"
      " x_test: ", np.shape(x_test),
      " y_test: ", np.shape(y_test) )

############ preProcessing avoiding overfitin and underfiting
#%% Compute a Principal component analysis (PCA)
from sklearn.decomposition import PCA
""" Reduce channel or pixel values: 
        reshape all information like color channel by getting all in just one matrix 
            -> model.fit or pca need 2D array but x_train and x_test are 3D. Convert they into 2D using np.concatenate
"""
# convert 3D in 2D without losing information
x_train = [np.concatenate(i) for i in x_train]
x_test = [np.concatenate(i) for i in x_test]

#  Compute a Principal component analysis PCA 
pca = PCA(n_components= 150, svd_solver="randomized", whiten=True).fit(x_train)
X_train_pca = pca.transform(x_train)
X_test_pca = pca.transform(x_test)


############ Model training and optimizing hyper parameter 
#%% Train a SVM classification model + Hyperparameter Tuning 

# Hyperparameter Tuning: Randomized search on hyper parameters
from sklearn.model_selection import RandomizedSearchCV
# parameter list as dic, builds different possible combinatorial parameter pairs 
param_grid = { 
    "C": [1, 2], 
    'kernel': ['linear'
               # ,'poly' ,'rbf', , 'sigmoid','precomputed'
               ],
    "gamma": [1, 10]
    # more parameter   
    } 
# Randomized search on hyper parameters + train a SVM classification model
from sklearn.svm import SVC
clf = RandomizedSearchCV( SVC(), param_grid, n_iter=8 )
#clf = RandomizedSearchCV( SVC( kernel="rbf", class_weight="balanced"), param_grid, n_iter=8 )
clf = clf.fit(X_train_pca, y_train)

#Predicting using preprocessed test data X_test_pca
y_pred = clf.predict(X_test_pca)

#%% Quantitative evaluation of the model quality on the test set

# classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=target_names))

# ConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(clf, X_test_pca, y_test, display_labels=target_names, xticks_rotation="vertical")

# plot
import matplotlib.pyplot as plt
plt.tight_layout()
plt.show()
