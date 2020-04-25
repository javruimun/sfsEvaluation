import pandas as pd
import numpy as np
import math
import sklearn

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

datos = pd.read_csv('titanic.csv', sep=',', index_col=0)
dataFrame = pd.DataFrame(datos)

relation = datos.groupby('Survived').size()
relationDiv = relation[0]/relation[1]

cv = KFold(n_splits=3)
accuracies = list()
numeroAtributos = len(list(dataFrame))
rangoProf = range(1, numeroAtributos + 1)

#Creamos el árbol de decisión para los distintos atributos

for profundidad in rangoProf:
    actual_accuracies = []
    tree = DecisionTreeClassifier(criterion='entropy',
                                             min_samples_split=20,
                                             min_samples_leaf=5,
                                             max_depth = profundidad,
                                             class_weight={1:relationDiv})
    
    for foldEntrenamiento, testFold in cv.split(dataFrame):
        conjuntoEntrenamiento = dataFrame.iloc[foldEntrenamiento]
        conjuntoTest = dataFrame.iloc[testFold] 
        
        modelo = tree.fit(X = conjuntoEntrenamiento.drop(['Survived'], axis=1), 
                               y = conjuntoEntrenamiento["Survived"])
        accuraciesValidas = modelo.score(X = conjuntoTest.drop(['Survived'], axis=1), 
                                y = conjuntoTest["Survived"]) # calculamos la precision con el segmento de validacio
        
        actual_accuracies.append(accuraciesValidas)
    
    avg = sum(actual_accuracies)/len(actual_accuracies)
    accuracies.append(avg)

# Mostramos los resultados obtenidos
df = pd.DataFrame({"Max Depth": rangoProf, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))
