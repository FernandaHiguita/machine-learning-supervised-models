# Predecir ¿Cuánto costó el boleto de un pasajero del Titanic?

import pandas as pd 

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, r2_score

# Cargar el dataset 

df = pd.read_csv("data/Titanic-Dataset.csv")

# Seleccionar columnas 

df = df[["Fare", "Pclass", "Sex", "Age"]]

# Convertir variable categórica a numérica 

df = pd.get_dummies(df, columns=["Sex"], drop_first=True)

# Eliminar valores nulos 

df = df.dropna()

# Separar Features y Target 

X = df.drop("Fare", axis=1)
Y = df["Fare"]

# Dividir entrenamiento y prueba 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2, 
    random_state=42
)

# Crear modelo 

model = LinearRegression() 

# Entrenar modelo 

model.fit(X_train, y_train)

#Realizar predicciones 

y_pred =  model.predict(X_test)

print(y_pred[:5])

# Evaluación 

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", round(mae, 2))
print("R²:", round(r2, 3))