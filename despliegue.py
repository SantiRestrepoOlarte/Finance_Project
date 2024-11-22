#Librerias

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# Cargar datos nuevos
data_new = pd.read_csv('data//datos_nuevos_creditos.csv')

# Cargar modelo
svc_opt = joblib.load('salidas/model.pkl')

# Eliminación de columna 'NewLoanApplication'
data_new.drop(['NewLoanApplication'], axis=1, inplace=True)

# Guardar la columna ID
ID = pd.DataFrame(data_new['ID'])

# Eliminación de variable 'ID'
data_new.drop(['ID'], axis=1, inplace=True)

# Dummies
data_new_dummies = pd.get_dummies(data_new)
data_new_dummies.head()

# Estandarización de datos
std = StandardScaler()
data_new_std = std.fit_transform(data_new_dummies)

# Convertir array en dataframes
data_new_final = pd.DataFrame(data_new_std,columns=data_new_dummies.columns)

# Predicciones
predicciones = svc_opt.predict(data_new_final)

predicciones = pd.DataFrame(predicciones, columns=['Risk_Level'])

# Concatenar ID
predicciones = pd.concat([ID, predicciones], axis=1)

# Agregar columna con la tasa de interés

# Definir las condiciones y los valores asociados
condiciones = [
    predicciones['Risk_Level'] == 0,
    predicciones['Risk_Level'] == 1,
    predicciones['Risk_Level'] == 2,
    predicciones['Risk_Level'] == 3
]

valores = [0.12, 0.18, 0.24, 0.3]  # Valores correspondientes a cada nivel

# Agregar la columna 'tasa'
predicciones['int_rc'] = np.select(condiciones, valores, default=None)

# Eliminar columna 'Risk_Level'
predicciones.drop(['Risk_Level'], axis=1, inplace=True)

# Guardar predicciones
predicciones.to_csv('salidas/predicciones.csv', index=False)