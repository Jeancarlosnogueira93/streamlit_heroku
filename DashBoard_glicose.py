from xml.sax.handler import feature_external_ges
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Titulo
st.write(""" Prevendo Diabetes\n
        App que utiliza machine learning para prever possivel diabetes dos paciente.\n
        Fonte : PIMA - India(Kaggle)""")

#dataset
df = pd.read_csv("D:\Projeto\Python\Heroku + Steamlit\diabetes.csv")

#Cabeçalho
st.subheader('Informações dos dados')

#nome do usuario
user_input = st.sidebar.text_input("Digite seu Nome : ")
st.write('Paciente :' , user_input)

#Dados de entrada
x = df.drop(['Outcome'],1)
y = df['Outcome']

#Separa dados em treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.2,random_state=42)

#dados dos usuario com função 

def get_user_data():
    pregnancies = st.sidebar.slider('Gravidez',0,15,1)
    glucose = st.sidebar.slider('Glucose',0,200,110)
    blood_pressure = st.sidebar.slider('Pressão Sanguinea',0,122,72)
    skin_thicknes = st.sidebar.slider('Espessura da pele',0,99,20)
    insulin = st.sidebar.slider('Insulina',0,900,30)
    bmi = st.sidebar.slider('Indice de massa corporal',0.0,70.0,15.0)
    dpf = st.sidebar.slider('Historico familiar de biabetes',0.0,3.0,0.0)
    age = st.sidebar.slider('Idade',15,100,21)
    
    user_data = { 'gravidez' : pregnancies,
                  'Glicose': glucose,
                  'Pressão Sanguinea': blood_pressure,
                  'Espessura da pele' : skin_thicknes,
                  'Insulina' : insulin,
                  'Indice de massa corporal' : bmi,
                  'Historico familiar de biabetes':dpf,
                  'Idade' : age
                }
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input_variables = get_user_data()

#Graficos
graf = st.bar_chart(user_input_variables)

st.subheader('Dados do Usuario')
st.write(user_input_variables)

dtc = DecisionTreeClassifier(criterion='entropy',max_depth=3)
dtc.fit(x_train, y_train)

#Acuracia do modelo
st.subheader('Acuracia do modelo')
st.write(accuracy_score(y_test, dtc.predict(x_test))*100)

#Previsão
prediction = dtc.predict(user_input_variables)
st.subheader('Previsão : ')
st.write(prediction)