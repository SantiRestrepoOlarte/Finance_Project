#Librerias

import numpy as np
import pandas as pd
import joblib
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pprint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

data_hist = joblib.load('salidas\\data_hist.pkl')

data_hist.head()

# Eliminación de variable 'NoPaidPerc'
data_hist.drop(['NoPaidPerc'], axis=1, inplace=True)

# Guardar la columna ID
ID = pd.DataFrame(data_hist['ID'])

# Eliminación de variable 'ID'
data_hist.drop(['ID'], axis=1, inplace=True)

# Variable objetivo Risk_Level a numerical
data_hist['Risk_Level'] = data_hist['Risk_Level'].map({'Low': 0, 'Medium_Low': 1, 'Medium_High': 2, 'High': 3})
data_hist['Risk_Level'] = data_hist['Risk_Level'].astype(int)

data_hist.columns

# Dummies
data_hist = pd.get_dummies(data_hist)
data_hist.head()

# Separación de caracteristicas y target (X , y)
y = data_hist['Risk_Level']
X = data_hist.drop(['Risk_Level'], axis=1)

# Separación en conjuntos de entrenamiento y validación con 80% de muestras para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Imprimir Tamaño de dataset
print("Tamaño del conjunto de entrenamiento:", X_train.shape)
print("Tamaño del conjunto de validación:",  X_test.shape)

#Nombre de caracteristicas númericas
numeric_columns=list(X.select_dtypes(exclude='object').columns)

#Estandarización de variables númericas
pipeline = ColumnTransformer([('num', StandardScaler() , numeric_columns)], remainder='passthrough')

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

# Convertir array en dataframes
X_train = pd.DataFrame(X_train,columns=X.columns)
X_test = pd.DataFrame(X_test,columns=X.columns)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
X_train.head()

# Modelo Regresión Logística
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Métricas de desempeño regresión logística
# ==============================================================================
print ("Train - Accuracy :", metrics.accuracy_score(y_train, lr.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, lr.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, lr.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, lr.predict(X_test)))

# Modelo Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Métricas de desempeño decision tree
# ==============================================================================
print ("Train - Accuracy :", metrics.accuracy_score(y_train, dt.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, dt.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, dt.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, dt.predict(X_test)))

# Modelo Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Métricas de desempeño random forest
# ==============================================================================
print ("Train - Accuracy :", metrics.accuracy_score(y_train, rf.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, rf.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, rf.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, rf.predict(X_test)))

# Modelo SVM
svc = SVC()
svc.fit(X_train, y_train)

# Métricas de desempeño SVM
# ==============================================================================
print ("Train - Accuracy :", metrics.accuracy_score(y_train, svc.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, svc.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, svc.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, svc.predict(X_test)))

# Modelo Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# Métricas de desempeño Gradient Boosting
# ==============================================================================
print ("Train - Accuracy :", metrics.accuracy_score(y_train, gb.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, gb.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, gb.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, gb.predict(X_test)))

###################################################### Optimizació de hiperparámetros Support Vector Machine ########################################################

# Identificación de parametros de optimización

print("+----------------------\nModelo - Support Vector Machine\n+----------------------:")
pprint.pprint(svc.get_params())

# Definición de parametros para optimización

param = {
    'C': [0.1, 1, 10, 100, 1000],  # Regularización
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Tipo de kernel
    'degree': [2, 3, 4, 5, 6],  # Solo aplica para kernel 'poly'
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Solo aplica para kernels 'rbf', 'poly' y 'sigmoid'
    'class_weight': [None, 'balanced'],  # Balanceo de clases
    'max_iter': [-1]  # Número máximo de iteraciones (-1 significa sin límite)
}

# Definición de cuadricula de búsqueda
svc_opt = RandomizedSearchCV(svc, param_distributions=param)

# Iniciar la búsqueda
svc_opt.fit(X_train, y_train)

print('Mejores Hiperparámetros: ', svc_opt.best_params_)


######## Modelo elegido: Support Vector Machine Clasifier con hiperparámetros optimizados ########

# Métricas de desempeño modelo con hiperparámetros optimizados
# ==============================================================================
svc_opt = svc_opt.best_estimator_
print ("Train - Accuracy :", metrics.accuracy_score(y_train, svc_opt.predict(X_train)))
print ("Train - classification report :", metrics.classification_report(y_train, svc_opt.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, svc_opt.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, svc_opt.predict(X_test)))

# Graficar curva ROC
# Predecir probabilidades
y_proba = svc_opt.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Línea diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()


# Graficar matriz de confusión
cm = confusion_matrix(y_test, svc_opt.predict(X_test))
# Visualización de la matriz de confusion
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.show()


############# Exportar modelo afinado #############
joblib.dump(svc, 'salidas\\model.pkl')