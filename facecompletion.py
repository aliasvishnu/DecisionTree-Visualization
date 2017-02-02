
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces


# In[2]:

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV


# In[44]:

data = fetch_olivetti_faces()


# In[11]:

plt.imshow(data['images'][0], cmap = plt.cm.gray)
plt.show()


# In[64]:

shuffle = np.random.choice(len(data.images), len(data.images))
images = (data.images.reshape((len(data.images), -1)))[shuffle]
targets = (data.target)[shuffle]


# In[74]:

npixels = np.product(images[0].shape)
trainX = images[targets < 30][:, :int(np.ceil(0.5*npixels))]
trainY = images[targets < 30][:, int(np.floor(0.5*npixels)):]

testX = images[targets >= 30][:, :int(np.ceil(0.5*npixels))]
testY = images[targets >= 30][:, int(np.floor(0.5*npixels)):]


# In[77]:

plt.imshow(trainX[0].reshape((32, 64)), cmap = plt.cm.gray)
plt.show()


# In[97]:

models = {
    "Extra trees": ExtraTreesRegressor(n_estimators = 10,
                                     max_features = 32, random_state=0),
    "K-nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(), 
#     "Ridge": RidgeCV(),
}


# In[98]:

prediction = {}
for name, model in models.items():
    print name
    model.fit(trainX, trainY)
    prediction[name] = model.predict(testX)


# In[99]:

image_shape = (64, 64)
n_cols = 1+len(models)
n_faces = 5
plt.figure(figsize=(2.0*n_cols, 2.26*n_faces))
plt.suptitle("Face completion")
for i in range(5):
    true_face = np.hstack((testX[i], testY[i]))
    
    if i:
        sub = plt.subplot(n_faces, n_cols, i*n_cols+1)
    else:
        sub = plt.subplot(n_faces, n_cols, i*n_cols+1, title = "True faces")
            
    sub.axis("off")
    sub.imshow(true_face.reshape(image_shape),cmap=plt.cm.gray,
              interpolation="nearest")

    for j, est in enumerate(sorted(models)):
        completed_face = np.hstack((testX[i], prediction[est][i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                              title=est)

        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

plt.show()


# In[ ]:



