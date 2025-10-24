# # Task: Eres un analista de datos y tu mánager quiere entender mejor el panorama de las keywords. Te ha pedido que prepares un pequeño informe respondiendo a las siguientes preguntas.

# 1) El crecimiento interanual (yoy_change) es útil, pero quiero una categoría simple: 'Crecimiento', 'Declino' o 'Estable'.
# 2) Ahora, ¿la competencia es diferente para las keywords que están en crecimiento vs. las que están en declino?
# 3) Dame un resumen. ¿Cuál es el promedio de avg_monthly_searches y la mediana de competition_index para cada grupo de tendencia?
# 4) Necesito una lista de 'joyas ocultas': keywords que tengan más de 1000 búsquedas mensuales pero que, por alguna razón, tengan una competencia 'Low'"

# %%
# 1. Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandasql import sqldf

# --- Configuraciones de buenas prácticas ---
# Configurar SQL
pysqldf = lambda q: sqldf(q, globals())
# Configura Seaborn para que los gráficos se vean más modernos
sns.set_theme(style="whitegrid")
# Evita que salgan números en notación científica
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("Librerías importadas y configuraciones aplicadas.")

# %%
# 2. Cargar el DataFrame LIMPIO
df_limpio = pd.read_csv("datos_google_ads_clean.csv")

print("Datos limpios cargados. ¡Listo para el análisis!")

# 3. Inspección rápida
print(df_limpio.info())
print(df_limpio.head())
print(df_limpio.tail())


# %%
# 4. Categorizar el crecimiento interanual
def categorizar_crecimiento(valor):
    if valor > 0.05:
        return 'Crecimiento'
    elif valor < -0.05:
        return 'Declino'
    elif -0.05 <= valor <= 0.05:
        return 'Estable'
    else:
        return 'Desconocido'

df_limpio['tendencia'] = df_limpio['yoy_change'].apply(categorizar_crecimiento)
print("Categorías de tendencia asignadas.")

# %%
# 5. Análisis de competencia por categoría de tendencia
consulta_competencia = """
SELECT competition, COUNT(*) as cantidad, tendencia
FROM df_limpio
GROUP BY competition, tendencia
ORDER BY competition, cantidad DESC
"""
df_competencia = pysqldf(consulta_competencia)
print("Análisis de competencia por categoría de tendencia realizado.")
print(df_competencia)

# %%
# 6. Resumen estadístico por grupo de tendencia
# sqlite no tiene la función MEDIAN, por lo que calculamos la mediana usando pandas directamente.
df_resumen = (
    df_limpio
    .groupby('tendencia')
    .agg(
        promedio_busquedas_mensuales=('avg_monthly_searches', 'mean'),
        mediana_indice_competencia=('competition_index', 'median')
    )
    .reset_index()
)

# Aseguramos un orden lógico de las categorías y reordenamos el resultado
ordered = ['Crecimiento', 'Declino', 'Estable', 'Desconocido']
df_resumen['tendencia'] = pd.Categorical(df_resumen['tendencia'], categories=ordered, ordered=True)
df_resumen = df_resumen.sort_values('tendencia').reset_index(drop=True)

print("Resumen estadístico por grupo de tendencia realizado.")
print(df_resumen)

# %%
# 7. Identificación de 'joyas ocultas'
joyas_ocultas = df_limpio[(df_limpio['avg_monthly_searches'] > 1000) & (df_limpio['Competition'] == 'Low')]
print("Identificación de 'joyas ocultas' realizada.")
print(joyas_ocultas)

# %%
# 8. Resultados finales
print("\n--- Resumen Final ---")
print("Análisis de competencia por categoría de tendencia:")
print(df_competencia)

print("\nResumen estadístico por grupo de tendencia:")
print(df_resumen)

print("\nJoyas ocultas:")
print(joyas_ocultas)

# %%
# 9. Visualización (opcional)
plt.figure(figsize=(10, 6))
sns.countplot(data=df_limpio, x='tendencia', hue='competition_index')
plt.title('Distribución de Competencia por Categoría de Tendencia')
plt.xlabel('Categoría de Tendencia')
plt.ylabel('Cantidad de Keywords')
plt.legend(title='Índice de Competencia')
plt.show()



# %%
# 10. Guardar resultados en archivos CSV
df_competencia.to_csv("analisis_competencia_por_tendencia.csv", index=False)
df_resumen.to_csv("resumen_estadistico_por_tendencia.csv", index=False)
joyas_ocultas.to_csv("joyas_ocultas.csv", index=False)
print("Resultados guardados en archivos CSV.")


