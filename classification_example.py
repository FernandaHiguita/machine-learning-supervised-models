import pandas as pd

# Herramienta para dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Algoritmo de clasificación que utilizaremos
from sklearn.linear_model import LogisticRegression

# Métricas para evaluar el modelo
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# Cargar el dataset 

df = pd.read_csv("data/Titanic-Dataset.csv")

# Seleccionamos columnas necesarias 

df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]

#Convertir variables categorigas a númericas 

# Sex_male
# 1 = Hombre
# 0 = Mujer

df = pd.get_dummies(
    df,
    columns=["Sex"],
    drop_first=True
)

# Limpiar datos faltantes 

df = df.dropna()

# Separamos Feature del Target 

X = df.drop("Survived", axis=1)

y = df["Survived"]

# Dividir los datos 

# 80% para entrenamiento
# 20% para prueba

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Crear el modelo 

model = LogisticRegression(max_iter=1000)

# Entrenar modelo 

model.fit(X_train, y_train)

# Realizar predicciones 

# El modelo intenta predecir si un pasajero
# sobrevivió o no sobrevivió.

y_pred = model.predict(X_test)

# Mostramos las primeras predicciones
print("Primeras predicciones:")
print(y_pred[:10])

#Evaluar el modelo 

# Accuracy:
# Qué porcentaje de predicciones fueron correctas.

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(round(accuracy, 3))

# Reporte detallado 

# Muestra:
#
# Precision:
# De todos los pasajeros que predije como sobrevivientes,
# cuántos realmente sobrevivieron.
#
# Recall:
# De todos los sobrevivientes reales,
# cuántos encontró el modelo.
#
# F1:
# Balance entre Precision y Recall.

print("\nClassification Report:")
print(classification_report(y_test, y_pred))