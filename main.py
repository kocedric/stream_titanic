import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
# %matplotlib inline


#---------------------------------#
# Title
# image = Image.open('logo.png')
# st.image(image, width = 500)
st.title('TITANIC')

st.write(
    """
    # Bienvenue sur TITANIC - REVIEW
    Vous pouvez tester votre chance de survie à bord du Titanic.
    """
)

st.sidebar.header("Vos informations personneles")

def user_input():
    # pclass = st.sidebar.slider("Classe du passager", 1, 3, 3)
    pclass = st.sidebar.radio(
        "Classe du passager",
        ('1', '2', '3'))
    # sex = st.sidebar.slider("Genre du passager(0:Homme, 1:Femme)", 0, 1, 0)
    sex = st.sidebar.radio(
     "Genre du passager",
     ('Homme', 'Femme'))
    age = st.sidebar.slider("Votre âge", 1, 99, 28)

    if sex.lower() == 'homme':
        sex = 0
    else:
        sex = 1

    data = {
        'pclass': int(pclass),
        'sex': sex,
        'age': age
    }

    input_parametres = pd.DataFrame(data, index=[0])
    return input_parametres


df = user_input()
st.subheader("Prédiction de vos chanses de survie à bord du Titanic")
st.write(df)


## Code Machine Learnig

titanic = pd.read_excel('Dataset/titanic3.xls')

# Supprime toutes les colones du dataset sauf celles selectionnées
titanic = titanic[['survived', 'pclass', 'sex', 'age']]
titanic.dropna(axis=0, inplace=True)  # Supprime toutes les lignes ayant des valeurs manquantes

titanic['sex'].replace(['male', 'female'], [0, 1], inplace=True)
model = KNeighborsClassifier()

y = titanic['survived']
X = titanic.drop('survived', axis=1)

model.fit(X, y) # entrainement du modele
model.score(X, y) # évaluation

# Fonction de prediction de survie d'un individu
def survie(model, pclass=3, sex=0, age=26):
  x = np.array([pclass, sex, age]).reshape(1, 3)
  prediction = model.predict(x)
  taux = model.predict_proba(x)
#   print(prediction)
#   print(taux)
  # print(f"Vous avez {taux[0][1]*100}% de chances de suvivre au novrage du Titanic")
  # print(f"Contre {taux[0][0]*100}% de malchances d'y peri")

  st.subheader(f"Vous avez {taux[0][1]*100}% de chances de suvivre au novrage du Titanic")
  st.subheader(f"Contre {taux[0][0]*100}% de malchances d'y peri")
  if prediction == 1:
    # print(f"Nous vous classons donc dans la classe des survivants")
    st.write("Nous vous classons donc dans la classe des survivants")
  else:
    # print(f"Nous vous classons donc dans la classe des personnes qui ont peri")
    st.write("Nous vous classons donc dans la classe des personnes qui ont peri")

survie(model, pclass=df['pclass'], sex=df['sex'], age=df['age'])


#---------------------------------#
# Graphics

sns.pairplot(data=titanic, hue='sex')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# st.subheader()
fig,ax = plt.subplots(figsize=(8,6))
sns.countplot(data=titanic, x='survived',hue= 'sex',ax=ax)
st.pyplot(fig)


fig,ax = plt.subplots(figsize=(8,6))
sns.countplot(data=titanic, x='survived',hue= 'pclass',ax=ax)
st.pyplot(fig)

kp = px.sunburst(titanic, path=['sex', 'pclass'],values='survived')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

#---------------------------------#
kp = px.sunburst(titanic, path=['sex', 'age'],values='survived')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

#---------------------------------#
#ending
