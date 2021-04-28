import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


#---------------------------------#
# Title
# image = Image.open('logo.png')
# st.image(image, width = 500)
st.title('Bienvenue sur TITANIC - REVIEWS')

st.write(
    """
    
    ## Vous pouvez tester votre chance de survie à bord du Titanic.
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
y = titanic['survived']
X = titanic.drop('survived', axis=1)

# Diviser le Dataset en données d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Recherche des meilleurs parametres de prediction du model
param_grid = {'n_neighbors': np.arange(1, 20),
              'metric': ['euclidean', 'manhattan']}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)

grid.fit(X_train, y_train) # Entrainement sur les données d'entrainements

model = grid.best_estimator_


# Fonction de prediction de survie d'un individu
def survie(model, pclass=3, sex=0, age=26):
  x = np.array([pclass, sex, age]).reshape(1, 3)
  prediction = model.predict(x)
  taux = model.predict_proba(x)
  
  chance_de_survie = taux[0][1]*100
  malchance_de_peri = taux[0][0]*100

  st.write(f"## Vous avez {chance_de_survie:.2f}% de chances de suvivre au novrage du Titanic")
  st.write(f"## Contre {malchance_de_peri:.2f}% de malchances d'y peri")
  if prediction == 1:
    # print(f"Nous vous classons donc dans la classe des survivants")
    st.write("# Felicitation, vous êtes classé parmit les survivants du Titanic")
  else:
    # print(f"Nous vous classons donc dans la classe des personnes qui ont peri")
    st.write("# Désolé, vous êtes classé parmit les personnes qui ont peri")
    

survie(model, pclass=df['pclass'], sex=df['sex'], age=df['age'])

st.subheader('')
st.subheader('')


#---------------------------------#
# Graphics

sns.pairplot(data=titanic, hue='sex')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# st.subheader()
fig,ax = plt.subplots(figsize=(12,6))
sns.countplot(data=titanic, x='survived',hue= 'sex',ax=ax)
st.pyplot(fig)

fig1 ,ax1 = plt.subplots(figsize=(12,6))
sns.countplot(data=titanic, x='survived',hue='pclass', ax=ax1)
st.pyplot(fig1)

sns.catplot(x="survived", y="age",hue='sex' ,kind="box", data=titanic)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


#---------------------------------#
#kp = px.sunburst(titanic, path=['sex', 'age'],values='survived')
#st.set_option('deprecation.showPyplotGlobalUse', False)
#st.pyplot(kp)

#---------------------------------#
#ending
