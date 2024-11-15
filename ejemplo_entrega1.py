import pandas as pd
import numpy as np


##### ejemplo formato primera entrega ######

url_nuevos='https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_nuevos_creditos.csv'
df_nuevos=pd.read_csv(url_nuevos)


df_nuevos['int_rc'] = 0.3
df_ejemplo=df_nuevos[['ID','int_rc']]
df_ejemplo.to_csv('data\\grupo_1.csv')


