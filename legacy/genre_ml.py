#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


df=pd.read_csv("music_features.csv")
df.shape
df.sample(5)


# In[3]:


df["label"].value_counts()


# In[4]:


df.isnull().sum()


# In[5]:


df.dtypes


# In[6]:


df["label"].value_counts().sort_index().plot(kind="bar",figsize=(8,4))
plt.title("Number of segments per genre")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.show()


# In[7]:


features = df.drop(columns=["song_id", "label", "length"])

features.hist(bins=30, figsize=(20, 24))
plt.tight_layout()
plt.show()


# In[8]:


features.plot(kind="kde", subplots=True, layout=(12, 5), figsize=(22, 38), sharex=False)
plt.tight_layout()
plt.show()


# In[9]:


features = df.drop(columns=["song_id", "label", "length"])

fig, axes = plt.subplots(12, 5, figsize=(22, 45))
for ax, col in zip(axes.ravel(), features.columns):
    sns.boxplot(data=df, x="label", y=col, ax=ax)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90)
plt.tight_layout()
plt.show()


# In[10]:


features = df.drop(columns=["song_id", "label", "length"])

plt.figure(figsize=(16, 14))
sns.heatmap(features.corr(), cmap="coolwarm", center=0)
plt.show()


# In[11]:


features = df.drop(columns=["song_id", "label", "length"])

Q1 = features.quantile(0.25)
Q3 = features.quantile(0.75)
IQR = Q3 - Q1

outliers = ((features < Q1 - 1.5 * IQR) | (features > Q3 + 1.5 * IQR)).sum()
outliers.sort_values(ascending=False).head(15)
(outliers / len(features) * 100).sort_values(ascending=False).head(15)


# In[12]:


y = df["label"]
groups = df["song_id"]
X = df.drop(columns=["label", "song_id", "length"])

print(X.shape)
print(y.shape)


# In[13]:


from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(X_train.shape, X_test.shape)
print("Songs in both sets:", len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])))
print(y_test.value_counts())


# In[14]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.mean().round(3), X_train_scaled.std().round(3))
print(X_test_scaled.mean().round(3), X_test_scaled.std().round(3))


# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

print("Train accuracy:", accuracy_score(y_train, train_pred))
print("Test accuracy: ", accuracy_score(y_test, test_pred))


# In[17]:


from sklearn.metrics import ConfusionMatrixDisplay, classification_report

ConfusionMatrixDisplay.from_predictions(y_test, test_pred, xticks_rotation=45, cmap="Blues")
plt.show()

print(classification_report(y_test, test_pred))


# In[18]:


from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer()
X_train_pt = pt.fit_transform(X_train)
X_test_pt = pt.transform(X_test)

model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train_pt, y_train)

print("Train accuracy:", accuracy_score(y_train, model2.predict(X_train_pt)))
print("Test accuracy: ", accuracy_score(y_test, model2.predict(X_test_pt)))


# In[20]:


X_train_pt_df = pd.DataFrame(X_train_pt, columns=X.columns)

X_train_pt_df.plot(kind="kde", subplots=True, layout=(12, 5), figsize=(22, 38), sharex=False)
plt.tight_layout()
plt.show()


# In[21]:


lo = X_train.quantile(0.01)
hi = X_train.quantile(0.99)

X_train_c = X_train.clip(lo, hi, axis=1)
X_test_c = X_test.clip(lo, hi, axis=1)

pt2 = PowerTransformer()
X_train_c_pt = pt2.fit_transform(X_train_c)
X_test_c_pt = pt2.transform(X_test_c)

model4 = LogisticRegression(max_iter=1000)
model4.fit(X_train_c_pt, y_train)
print("Test accuracy:", accuracy_score(y_test, model4.predict(X_test_c_pt)))


# In[23]:


from sklearn.preprocessing import PolynomialFeatures, StandardScaler

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_pt)
X_test_poly = poly.transform(X_test_pt)

sc = StandardScaler()
X_train_poly = sc.fit_transform(X_train_poly)
X_test_poly = sc.transform(X_test_poly)

model_poly = LogisticRegression(max_iter=2000)
model_poly.fit(X_train_poly, y_train)

print("Train accuracy:", accuracy_score(y_train, model_poly.predict(X_train_poly)))
print("Test accuracy: ", accuracy_score(y_test, model_poly.predict(X_test_poly)))


# In[24]:


for C in [0.0005, 0.002, 0.01]:
    m = LogisticRegression(C=C, max_iter=2000)
    m.fit(X_train_poly, y_train)
    print(f"C={C}  train: {accuracy_score(y_train, m.predict(X_train_poly)):.3f}  "
          f"test: {accuracy_score(y_test, m.predict(X_test_poly)):.3f}")


# In[13]:


best = LogisticRegression(C=0.01, max_iter=2000)
best.fit(X_train_poly, y_train)

proba = pd.DataFrame(best.predict_proba(X_test_poly), columns=best.classes_)
proba["song"] = groups.iloc[test_idx].values

song_proba = proba.groupby("song").mean()
song_pred = song_proba.idxmax(axis=1)
song_true = y_test.groupby(groups.iloc[test_idx].values).first().loc[song_pred.index]

print("Songs in test set:", len(song_pred))
print("Song-level accuracy:", accuracy_score(song_true, song_pred))


# In[10]:


from sklearn.svm import SVC

svm = SVC(kernel="rbf", C=10, gamma="scale")
svm.fit(X_train_pt, y_train)

print("Train accuracy:", accuracy_score(y_train, svm.predict(X_train_pt)))
print("Test accuracy (segment):", accuracy_score(y_test, svm.predict(X_test_pt)))

scores = pd.DataFrame(svm.decision_function(X_test_pt), columns=svm.classes_)
scores["song"] = groups.iloc[test_idx].values
song_pred = scores.groupby("song").mean().idxmax(axis=1)
song_true = y_test.groupby(groups.iloc[test_idx].values).first().loc[song_pred.index]

print("Test accuracy (song):", accuracy_score(song_true, song_pred))


# In[25]:


from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer

groups_train = groups.iloc[train_idx]

pipe = make_pipeline(PowerTransformer(), SVC(kernel="rbf"))
grid = {"svc__C": [1, 3, 10], "svc__gamma": [0.001, 0.003, "scale"]}

search = GridSearchCV(
    pipe, grid,
    cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
    scoring="accuracy", n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train, groups=groups_train)

print(search.best_params_)
print("CV accuracy:", round(search.best_score_, 3))
pd.DataFrame(search.cv_results_)[["param_svc__C", "param_svc__gamma", "mean_test_score"]] \
    .sort_values("mean_test_score", ascending=False)


# In[26]:


grid = {"svc__C": [3, 10], "svc__gamma": [0.0175, 0.03, 0.06, 0.1]}

search = GridSearchCV(
    pipe, grid,
    cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
    scoring="accuracy", n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train, groups=groups_train)

print(search.best_params_)
pd.DataFrame(search.cv_results_)[["param_svc__C", "param_svc__gamma", "mean_test_score"]] \
    .sort_values("mean_test_score", ascending=False)


# In[27]:


final = make_pipeline(PowerTransformer(), SVC(kernel="rbf", C=3, gamma=0.03))
final.fit(X_train, y_train)

print("Segment accuracy:", accuracy_score(y_test, final.predict(X_test)))

scores = pd.DataFrame(final.decision_function(X_test), columns=final.classes_)
scores["song"] = groups.iloc[test_idx].values
song_pred = scores.groupby("song").mean().idxmax(axis=1)
song_true = y_test.groupby(groups.iloc[test_idx].values).first().loc[song_pred.index]
print("Song accuracy:", accuracy_score(song_true, song_pred))


# In[28]:


from sklearn.metrics import ConfusionMatrixDisplay, classification_report

ConfusionMatrixDisplay.from_predictions(song_true, song_pred, xticks_rotation=45, cmap="Blues")
plt.show()
print(classification_report(song_true, song_pred))


# In[29]:


from sklearn.decomposition import PCA

pipe_pca = make_pipeline(PowerTransformer(), PCA(), SVC(kernel="rbf", C=3, gamma="scale"))
grid = {"pca__n_components": [10, 20, 30, 40, 57]}

search = GridSearchCV(
    pipe_pca, grid,
    cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
    scoring="accuracy", n_jobs=-1,
)
search.fit(X_train, y_train, groups=groups_train)

pd.DataFrame(search.cv_results_)[["param_pca__n_components", "mean_test_score"]]


# In[30]:


pipe_poly = make_pipeline(PowerTransformer(), SVC(kernel="poly", coef0=1, gamma="scale")) #polynomial kernel svm
grid = {"svc__degree": [2, 3], "svc__C": [1, 3]}

search = GridSearchCV(
    pipe_poly, grid,
    cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
    scoring="accuracy", n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train, groups=groups_train)

pd.DataFrame(search.cv_results_)[["param_svc__degree", "param_svc__C", "mean_test_score"]] \
    .sort_values("mean_test_score", ascending=False)


# In[24]:


cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
seg_scores, song_scores = [], []

for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
    m = make_pipeline(PowerTransformer(), SVC(kernel="rbf", C=3, gamma=0.03))
    m.fit(X.iloc[tr], y.iloc[tr])

    seg = accuracy_score(y.iloc[te], m.predict(X.iloc[te]))

    s = pd.DataFrame(m.decision_function(X.iloc[te]), columns=m.classes_)
    s["song"] = groups.iloc[te].values
    pred = s.groupby("song").mean().idxmax(axis=1)
    true = y.iloc[te].groupby(groups.iloc[te].values).first().loc[pred.index]
    song = accuracy_score(true, pred)

    seg_scores.append(seg)
    song_scores.append(song)
    print(f"Fold {fold}: segment={seg:.3f}  song={song:.3f}")

print(f"\nSegment-level: {np.mean(seg_scores):.3f} +/- {np.std(seg_scores):.3f}")
print(f"Song-level:    {np.mean(song_scores):.3f} +/- {np.std(song_scores):.3f}")


# In[ ]:




