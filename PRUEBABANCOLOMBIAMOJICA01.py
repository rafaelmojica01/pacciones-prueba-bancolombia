# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 01:38:47 2024

@author: rafae
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ruta_archivo_excel = 'https://github.com/rafaelmojica01/pacciones-prueba-bancolombia/blob/main/Precio_Acciones.xlsx?raw=true'

hojas = pd.read_excel(ruta_archivo_excel, sheet_name=None, engine='openpyxl')


df_Precio_acciones = hojas['Precio_acciones']

for nombre_hoja, df in hojas.items():
    print(f'DataFrame de la hoja: {df_Precio_acciones}')
    df.head()
    
df_Precio_acciones.head()

#Limpieza de datos

hojas = pd.read_excel(ruta_archivo_excel, sheet_name=None)

df_Precio_acciones = hojas['Precio_acciones']

df_Precio_acciones.dropna(inplace=True)

df_Precio_acciones.drop_duplicates(inplace=True)

for columna in df_Precio_acciones.columns:
    if columna != 'FECHA':  
        df_Precio_acciones[columna] = pd.to_numeric(df_Precio_acciones[columna], errors='coerce')

for columna in df_Precio_acciones.columns:
    if columna != 'FECHA':  
        sns.boxplot(x=df_Precio_acciones[columna])

        Q1 = df_Precio_acciones[columna].quantile(0.25)
        Q3 = df_Precio_acciones[columna].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_Precio_acciones = df_Precio_acciones[(df_Precio_acciones[columna] >= lower_bound) & (df_Precio_acciones[columna] <= upper_bound)]

# Comprobar los datos limpios
df_Precio_acciones.head()

#Medidas de tendencia Central
df_medidas = df_Precio_acciones.drop(columns=['FECHA']).describe().transpose()

print(df_medidas)

# Gráficos
for columna in df_Precio_acciones.select_dtypes(include=['number']).columns:
    if columna != 'FECHA':
        sns.histplot(x=df_Precio_acciones[columna])
        plt.title(f"Histograma de la columna {columna}")
        plt.show()

# Diagrama circular 
for columna in df_Precio_acciones.select_dtypes(include=['object']).columns:
    if columna != 'FECHA':
        df_Precio_acciones[columna].value_counts().plot(kind='pie')
        plt.title(f"Diagrama circular de la columna {columna}")
        plt.show()
        
acciones = ['CONCONC CB Equity', 'PFBCOLO CB Equity', 'PFGRUPSU CB Equity', 'CEMARGOS CB Equity', 'ECOPETL CB Equity']

# Rentabilidad diaria
rentabilidad_diaria = df_Precio_acciones[acciones].pct_change()

# Graficas
plt.figure(figsize=(12, 8))

for accion in acciones:
    sns.lineplot(data=rentabilidad_diaria, x=df_Precio_acciones['FECHA'], y=accion, label=accion)

plt.title("Rentabilidad diaria de las acciones especificadas")
plt.xlabel("Fecha")
plt.ylabel("Rentabilidad diaria")
plt.legend()
plt.show()

#bonus
import numpy as np
from scipy.stats import norm

def calcular_var_10_dias(portafolio, ponderaciones, nivel_confianza=0.95):
    
    rentabilidad_portafolio = np.dot(rentabilidad_diaria, ponderaciones)
    
    media = np.mean(rentabilidad_portafolio)
    desviacion = np.std(rentabilidad_portafolio)
    
    z = norm.ppf(nivel_confianza)
    var_10_dias = (media - z * desviacion)
    covarianza = rentabilidad_diaria.cov()
    var_por_componente = ponderaciones * covarianza.dot(ponderaciones) * norm.ppf(nivel_confianza)
    resultados = {
        'VaR_10_dias': var_10_dias,
        'Component_VaR': var_por_componente
    }

    return resultados

acciones = ['CONCONC CB Equity', 'PFBCOLO CB Equity', 'PFGRUPSU CB Equity', 'CEMARGOS CB Equity', 'ECOPETL CB Equity']

ponderaciones = [0.2, 0.2, 0.2, 0.2, 0.2]

portafolio = df_Precio_acciones[acciones].pct_change().dropna()

resultados = calcular_var_10_dias(portafolio, ponderaciones)

print("VaR a 10 días del portafolio:", resultados['VaR_10_dias'])
print("Component VaR del portafolio:", resultados['Component_VaR'])

plt.figure(figsize=(10, 6))
sns.barplot(x=acciones, y=resultados['Component_VaR'])
plt.title("Component VaR del portafolio a 10 días")
plt.ylabel("Component VaR")
plt.show()

rentabilidad_promedio = df_Precio_acciones[acciones].mean()
max_rentabilidades = rentabilidad_promedio.nlargest(3)

print(rentabilidad_diaria.mean())

rentabilidad_diaria_promedio = rentabilidad_diaria.mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=acciones, y=rentabilidad_diaria_promedio)
plt.title("Rentabilidad diaria promedio de las acciones especificadas")
plt.xlabel("Acciones")
plt.ylabel("Rentabilidad diaria promedio")
plt.show()

     



#punto 7

df_Valor_en_riesgo = pd.read_excel('https://github.com/rafaelmojica01/pacciones-prueba-bancolombia/blob/main/Precio_Acciones.xlsx?raw=true', sheet_name='Valor_en_riesgo')

plt.figure(figsize=(10, 6))

sns.lineplot(data=df_Valor_en_riesgo, x='Fecha', y='Valor Mercado', hue='Código')

plt.title('Evolución del valor de mercado de los portafolios')
plt.xlabel('Fecha')
plt.ylabel('Valor de mercado')

ax = plt.gca()

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))

plt.show()

df_var_absoluto = df_Valor_en_riesgo[df_Valor_en_riesgo['Fecha'] >= '2024-01-01']

plt.figure(figsize=(10, 6))

sns.lineplot(data=df_var_absoluto, x='Fecha', y='VaR Absoluto%', label='VaR Absoluto%')
sns.lineplot(data=df_var_absoluto, x='Fecha', y='Límite VaR Absoluto', label='Límite VaR Absoluto')

plt.title('VaR Absoluto % y su límite desde enero de 2024')
plt.xlabel('Fecha')
plt.ylabel('Valor')

ax = plt.gca()

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 100:.0f}%'))

plt.show()

df_var_activo = df_Valor_en_riesgo[df_Valor_en_riesgo['Fecha'] >= '2024-01-01']


plt.figure(figsize=(10, 6))
sns.lineplot(data=df_var_activo, x='Fecha', y='VaR Activo %', label='VaR Activo %')
sns.lineplot(data=df_var_activo, x='Fecha', y='Límite Riesgo Activo', label='Límite Riesgo Activo')


plt.title('VaR Activo % y su límite desde enero de 2024')
plt.xlabel('Fecha')
plt.ylabel('Valor')


ax = plt.gca()


ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 100:.0f}%'))


plt.show()

