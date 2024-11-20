### ---------------------- TÍTULOS ------------------------
### -------------------------------------------------------

### -----> SUBTÍTULOS

## COM: Comentarios

### -------------------- LIBRERÍAS ------------------------
### -------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

### -----> COLORES
A1 = '#dc9526'
A2 = '#f3ac3d'
A3 = '#0c2054'
G1 = '#263238'
G2 = '#999999'
G3 = '#191919'

### ----------------- LECTURA DE DATOS --------------------
### -------------------------------------------------------

data_hist = pd.read_csv('data/datos_historicos.csv')
data_news = pd.read_csv('data/datos_nuevos_creditos.csv')

data_hist.info()
data_news.info()

# COM: Se tienen 10.000 datos históricos y 1.058 datos nuevos

data_hist.head()

# Creación de variable objetivo 'Risk_Level'
def classify_risk_level(df):
    # Define la columna 'Risk_Level' con los rangos de riesgo
    data_hist['Risk_Level'] = pd.cut(
        data_hist['NoPaidPerc'],
        bins=[-float('inf'), 0, 0.2, 0.25, float('inf')],  # Define los rangos de valores
        labels=['Low', 'Medium_Low', 'Medium_High', 'High']  # Etiquetas correspondientes a cada rango
    )
    return data_hist

data_hist = classify_risk_level(data_hist)

data_hist['Risk_Level'].value_counts()


# Crear el histograma
plt.figure(figsize=(10,6))
plt.hist(data_hist['Risk_Level'], bins=10, color='skyblue', edgecolor='black')

# Añadir título y etiquetas
plt.title('Histograma de Risk_Level')
plt.xlabel('Niveles de Riesgo')
plt.ylabel('Frecuencia')

# Mostrar el gráfico
plt.show()

# Agrupar por 'Risk_Level' y calcular el promedio de 'CreditScore'
promedio_credit_score = data_hist.groupby('Risk_Level')['CreditScore'].mean()

# Agrupar por 'Risk_Level' y calcular el promedio de 'DebtRatio'
promedio_debtratio = data_hist.groupby('Risk_Level')['DebtRatio'].mean()

# Agrupar por 'Risk_Level' y calcular el promedio de 'MonthlyIncome'
promedio_monthly_income = data_hist.groupby('Risk_Level')['MonthlyIncome'].mean()

# Agrupar por 'Risk_Level' y calcular el promedio de 'NumberOfTimesPastDue'
promedio_number_of_times_past_due = data_hist.groupby('Risk_Level')['NumberOfTimesPastDue'].mean()

# Agrupar por 'Risk_Level' y calcular el promedio de 'NumberOfOpenCreditLinesAndLoans'
promedio_number_of_open_credit_lines = data_hist.groupby('Risk_Level')['NumberOfOpenCreditLinesAndLoans'].mean()

#Matriz de correlación 

# Selección de columnas numéricas
columnas_numericas = [col for col in data_hist.columns if data_hist[col].dtype in ['int64', 'float64']]

# Filtro de DataFrame a las variables numéricas
df_numerico = data_hist[columnas_numericas]

# Calcular la matriz de correlación
corrmat = df_numerico.corr()

# Tamaño 
f, ax = plt.subplots(figsize=(10, 10)) 

# Mapa de calor
sns.heatmap(corrmat, vmax=.8, square=True, annot=True, annot_kws={"fontsize": 8}, fmt=".2f", cmap='Blues')

# Etiqueta de los ejes
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.title('Matriz de Correlación', fontsize=18)
plt.xlabel('Variables', fontsize=14)
plt.ylabel('Variables', fontsize=14)

plt.show()

### -----> Variables numércias
num_vars = [
    "CreditScore", "DebtRatio", "Assets", "Age", 
    "NumberOfDependents", "NumberOfOpenCreditLinesAndLoans", 
    "MonthlyIncome", "NumberOfTimesPastDue", 
    "EmploymentLength", "YearsAtCurrentAddress", "NoPaidPerc"
]

### -----> Variables categóricas
cat_vars = ["HomeOwnership", "Education", "MaritalStatus"]

# Crear variables dummy para las categóricas
data_dummies = pd.get_dummies(data_hist, columns=cat_vars, drop_first=True)

# Calcular la matriz de correlación
correlation_matrix = data_dummies.corr()

#------- SIN FILTRO

# Configuración de la figura y el tamaño
plt.figure(figsize=(15, 10))

# Crear el mapa de calor
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="PiYG", cbar_kws={'shrink': .8}, linewidths=0.5)

# Mostrar el gráfico
plt.title("Matriz de Correlación de Variables Numéricas y Categóricas")
plt.show()

#-------- CON FILTRO 

# Aplicar un filtro para mostrar solo valores por encima de 0.5 o por debajo de -0.5
filtered = correlation_matrix[(correlation_matrix > 0.5) | (correlation_matrix < -0.5)]

# Configuración de la figura y el tamaño
plt.figure(figsize=(15, 10))

# Crear el mapa de calor con el filtro aplicado
sns.heatmap(filtered, annot=True, fmt=".2f", cmap="PiYG", cbar_kws={'shrink': .8}, linewidths=0.5)

# Título y mostrar el gráfico
plt.title("Matriz de Correlación (Valores > 0.5 o < -0.5)")
plt.show()


# Eliminación de variables no significativas
#data_hist.drop(['ID'], axis=1, inplace=True)
#data_hist.drop(['NoPaidPerc'], axis=1, inplace=True)

plt.figure(figsize=(15, 13))

# Iterar sobre las variables numéricas y crear boxplots
for i in range(len(num_vars)):
    plt.subplot(2, 5, i + 1)
    sns.boxplot(y=data_hist[num_vars[i]], orient='v', width=0.5, color='#F3AC3D')
    plt.tight_layout()

plt.show()

data_dummies = pd.get_dummies(data_hist, columns=cat_vars, drop_first=True)

data_hist.head()

joblib.dump(data_hist, "salidas/data_hist.pkl")