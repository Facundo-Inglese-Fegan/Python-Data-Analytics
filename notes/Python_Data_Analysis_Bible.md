# 📖 La Biblia del Analista de Datos con Python

## El Credo del Analista

Un Analista de Datos es un **traductor**. Habla el lenguaje de los datos (números, tablas, estadísticas) y lo traduce al lenguaje de las decisiones (estrategia, crecimiento, oportunidad).

### ¿Por Qué Python? 🐍

Python es el idioma universal del análisis. Es un **bisturí suizo** 🇨🇭:

* Es **legible** y fácil de escribir, por lo que tu código se documenta solo.
* Es **poderoso** y se integra con todo (bases de datos, web, IA).
* Tiene un **ecosistema** de librerías (`Pandas`, `NumPy`, `Seaborn`) que hacen el 90% del trabajo pesado por ti.

### El Objetivo Final

El objetivo no es crear código complejo. El objetivo es encontrar la **verdad** oculta en los datos y **comunicarla** de una forma tan clara y convincente que impulse a la acción.

### El Juramento: Las Buenas Prácticas

1. **Siempre conocerás tus datos:** Nunca inicies un análisis sin una exploración previa. Desconfía de los datos hasta que hayas probado que son limpios.
2. **Tu código es para humanos:** Escribirás código que otros (y tu "yo" del futuro) puedan entender. Usarás nombres de variables claros y dejarás comentarios.
3. **Tu análisis será reproducible:** Lo que haces en un *notebook* (`.ipynb`) debe poder ser ejecutado por alguien más y dar el mismo resultado.
4. **No sacarás conclusiones apresuradas:** Siempre visualizarás tus hallazgos. Un gráfico puede desmentir una estadística (como la media).

---

## El Ritual: El Flujo de Trabajo del Analista

Este es el proceso paso a paso, desde la pantalla en blanco hasta el *insight* final.

### Paso 0: La Invocación (Importar Librerías)

Todo análisis comienza con la misma invocación. Abres tu Notebook (`.ipynb`) y llamas a tus herramientas.

```python
# --- El "Santo Grial" de las importaciones ---

# 1. Pandas: La herramienta para manipular tablas (DataFrames)
import pandas as pd

# 2. NumPy: El motor matemático (lo usa Pandas por debajo)
import numpy as np

# 3. Matplotlib: La base para TODOS los gráficos
import matplotlib.pyplot as plt

# 4. Seaborn: La librería para gráficos estadísticos bonitos
import seaborn as sns

# 5. SQL: Habilitamos el uso de SQL dentros de Pandas
from pandasql import sqldf

# --- Configuraciones de buenas prácticas ---
# Configurar SQL
pysqldf = lambda q: sqldf(q, globals())
# Configura Seaborn para que los gráficos se vean más modernos
sns.set_theme(style="whitegrid")
# (Opcional) Evita que salgan números en notación científica
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("Librerías listas. Taller abierto.")
```

### Paso 1: La Adquisición (Cargar los Datos)

Debes traer el "material en bruto" (tus archivos) a tu taller (el DataFrame de Pandas).

```python
# --- Cargar un archivo CSV (El más común) ---
# (Asegúrate de que el CSV esté en la misma carpeta)
df = pd.read_csv('nombre_del_archivo.csv')

# --- Cargar un archivo Excel ---
# (Puede ser más lento que CSV. A veces hay que especificar la hoja)
df = pd.read_excel('nombre_del_archivo.xlsx', sheet_name='Hoja1')
```

### Paso 2: El Vistazo (La Primera Inspección)

Acabas de recibir una caja. Antes de usar su contenido, la abres y miras qué hay dentro.

```python
# --- 1. Ver las primeras 5 filas ---
# ¿Son los datos lo que esperabas? ¿Los nombres de columna son correctos?
print(df.head())

# --- 2. Ver las últimas 5 filas ---
# ¿Hay datos basura o sumarios al final del archivo?
print(df.tail())

# --- 3. Obtener la forma (Shape) ---
# ¿Cuántas filas y columnas tengo?
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

# --- 4. La Ficha Técnica (LA MÁS IMPORTANTE) ---
# .info() es tu radiografía. Te dice:
# - El nombre de CADA columna.
# - Cuántos valores NO NULOS tiene cada una (¡para detectar datos faltantes!).
# - El tipo de dato (Dtype) de cada columna (object=texto, float64=número).
print(df.info())
```

### Paso 3: La Radiografía (Resumen Estadístico)

Ahora que sabes qué hay, veamos cómo se comportan los datos.

```python
# --- 1. Resumen Estadístico de Columnas NUMÉRICAS ---
# .describe() te da la media, mediana (50%), desviación estándar,
# mínimo y máximo. Es VITAL para detectar outliers.
print(df.describe())

# --- 2. Resumen de Columnas CATEGÓRICAS (Texto) ---
# ¿Cuántos valores únicos hay?
print(df['Competition'].nunique()) # Ej: 3 (Low, Medium, High)

# ¿Cómo se distribuyen? (Contar cuántas veces aparece cada valor)
print(df['Competition'].value_counts())
```

### Paso 4: La Limpieza (El 80% del Trabajo)

Rara vez los datos vienen listos para usar. Debes ser un cirujano y arreglarlos.

```python
# --- 4.1. Manejo de Columnas (Renombrar) ---
# Buena práctica: Nombres en minúscula, sin espacios, todo en inglés.
df_limpio = df.rename(columns={
    'Avg. monthly searches': 'avg_monthly_searches',
    'YoY change': 'yoy_change'
})
# TRUCO PRO: Renombra TODAS las columnas a la vez
df_limpio.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '') for col in df_limpio.columns]


# --- 4.2. Manejo de Nulos (Datos Faltantes) ---
# ¿Cuántos nulos tengo por columna?
print(df_limpio.isnull().sum())

# Opción A: Eliminar filas que tengan nulos en una columna clave
df_limpio = df_limpio.dropna(subset=['avg_monthly_searches'])

# Opción B: Rellenar nulos (Ej: rellenar con la media o un valor específico)
media_bid = df_limpio['bid_low'].mean()
df_limpio['bid_low'] = df_limpio['bid_low'].fillna(media_bid)


# --- 4.3. Corrección de Tipos de Datos (Dtype) ---
# (Como hicimos con 'yoy_change' que era texto por el '%')
# 1. Limpiar el string
df_limpio['yoy_change'] = df_limpio['yoy_change'].str.replace('%', '')
# 2. Convertir a número (errors='coerce' es tu salvavidas)
df_limpio['yoy_change'] = pd.to_numeric(df_limpio['yoy_change'], errors='coerce')


# --- 4.4. Manejo de Duplicados ---
# ¿Cuántas filas están exactamente duplicadas?
print(f"Filas duplicadas: {df_limpio.duplicated().sum()}")
# Eliminarlas
df_limpio = df_limpio.drop_duplicates()
```

### Paso 5: La Interrogación (Análisis y NumPy)

Ahora que los datos están limpios, les hacemos preguntas. Aquí es donde Pandas (el coche) usa el motor de NumPy (los cálculos).

```python
# --- 5.1. Selección y Filtrado (El "WHERE" de SQL) ---
# Seleccionar una columna (devuelve una "Serie" de Pandas)
nombres_keywords = df_limpio['keyword']

# Seleccionar múltiples columnas (devuelve un DataFrame)
df_keywords_y_busquedas = df_limpio[['keyword', 'avg_monthly_searches']]

# Filtrar filas (condiciones booleanas)
df_competencia_baja = df_limpio[df_limpio['competition'] == 'Low']

# Filtro multi-condicional (& = "y", | = "o")
df_baja_y_alto_crecimiento = df_limpio[
    (df_limpio['competition'] == 'Low') & 
    (df_limpio['yoy_change'] > 100)
]


# --- 5.2. Creación de Nuevas Columnas (Feature Engineering) ---
# (Aquí Pandas usa NumPy por debajo para hacer los cálculos)
df_limpio['busquedas_por_competencia'] = df_limpio['avg_monthly_searches'] / df_limpio['competition_index']

# Usando una función de NumPy directamente (Ej: logaritmo para normalizar)
df_limpio['log_busquedas'] = np.log(df_limpio['avg_monthly_searches'] + 1) # +1 para evitar log(0)


# --- 5.3. Agregación (El "GROUP BY" de SQL) ---
# La herramienta MÁS PODEROSA de Pandas.
# Pregunta: ¿Cuál es la media de búsquedas y crecimiento por nivel de competencia?
df_agrupado = df_limpio.groupby('competition').agg(
    media_busquedas=('avg_monthly_searches', 'mean'),
    mediana_crecimiento=('yoy_change', 'median'),
    cantidad_keywords=('keyword', 'count')
)
# Ordenar el resultado
df_agrupado = df_agrupado.sort_values(by='media_busquedas', ascending=False)
print(df_agrupado)
```

### Paso 6: La Revelación (Visualización con Matplotlib/Seaborn)

Los números son abstractos. Las imágenes cuentan historias.
El Ritual de Matplotlib: Casi todos los gráficos de Seaborn terminan con comandos de Matplotlib (plt) para afinarlos.

```python
# --- El Ritual para CADA GRÁFICO ---
plt.figure(figsize=(10, 6)) # 1. Define el tamaño del lienzo
# (Aquí va el código de Seaborn)
plt.title('Título del Gráfico')
plt.xlabel('Etiqueta del Eje X')
plt.ylabel('Etiqueta del Eje Y')
plt.show() # 4. Muestra el gráfico
```

#### Pregunta 1: ¿Cómo se distribuyen mis datos? (Univariado)

Herramienta: Histograma o KDE (Kernel Density Estimate)

```python
# Ver la distribución de las búsquedas (¡la que vimos que estaba sesgada!)
plt.figure(figsize=(10, 6))
sns.histplot(data=df_limpio, x='avg_monthly_searches', kde=True, bins=50)
plt.title('Distribución de Búsquedas Mensuales')
plt.xlabel('Búsquedas Mensuales')
plt.ylabel('Frecuencia (Keywords)')
# (Opcional) Limitar el eje X para ver el detalle
plt.xlim(0, 1000)
plt.show()
```

#### Pregunta 2: ¿Hay relación entre dos variables? (Bivariado)

Herramienta: Gráfico de Dispersión (Scatterplot)

```python
# ¿Relación entre competencia y crecimiento? (El que hicimos)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_limpio, x='competition_index', y='yoy_change')
plt.title('Relación entre Competencia y Crecimiento (YoY)')
plt.xlabel('Índice de Competencia')
plt.ylabel('Crecimiento Interanual (%)')
plt.show()
```

#### Pregunta 3: ¿Cómo se comparan estas categorías?

Herramienta: Gráfico de Barras (Barplot)

```python
# Usamos el DataFrame agrupado que creamos antes
plt.figure(figsize=(10, 6))
# Nota: Para data=df_agrupado, 'x' debe ser una columna, así que reseteamos el índice
df_agrupado_para_grafico = df_agrupado.reset_index()
sns.barplot(data=df_agrupado_para_grafico, x='competition', y='media_busquedas')
plt.title('Media de Búsquedas por Nivel de Competencia')
plt.xlabel('Nivel de Competencia')
plt.ylabel('Media de Búsquedas')
plt.show()
```

#### Pregunta 4: ¿Cómo se distribuyen las categorías?

Herramienta: Gráfico de Conteo (Countplot)

```python
# Ver cuántas keywords hay de cada nivel de competencia
plt.figure(figsize=(8, 5))
sns.countplot(data=df_limpio, x='competition')
plt.title('Conteo de Keywords por Nivel de Competencia')
plt.xlabel('Nivel de Competencia')
plt.ylabel('Cantidad de Keywords')
plt.show()
```

#### Pregunta 5: ¿Cómo es la distribución por categoría?

Herramienta: Gráfico de Cajas (Boxplot) (Este es genial para comparar medianas y detectar outliers por grupo)

```python
# Ver la distribución de búsquedas PARA CADA nivel de competencia
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_limpio, x='competition', y='avg_monthly_searches')
plt.title('Distribución de Búsquedas por Competencia')
plt.xlabel('Nivel de Competencia')
plt.ylabel('Búsquedas Mensuales')
# (Opcional) Limitar el eje Y para que los outliers no aplasten el gráfico
plt.ylim(0, 2000)
plt.show()
```

### Paso 7: El Arte de Unir (Combinando Múltiples Archivos)

Rara vez todos tus datos estarán en un solo CSV.

#### La herramienta clave es `pd.merge()`

```python
# --- Paso Adicional: Uniendo Múltiples Fuentes ---

# Asumimos que tienes dos DataFrames:
# df_pedidos (con columnas 'id_cliente', 'id_producto', 'fecha_compra')
# df_clientes (con columnas 'id_cliente', 'nombre_cliente', 'ciudad')

# Queremos añadir la 'ciudad' del cliente a nuestra tabla de 'pedidos'.
# Usamos 'id_cliente' como la llave (key) que tienen en común.

df_completo = pd.merge(
    df_pedidos,
    df_clientes[['id_cliente', 'ciudad']], # Solo traemos las columnas que necesitamos
    on='id_cliente',   # La columna que usan para conectarse
    how='left'         # 'how=left' es el tipo de unión (como un LEFT JOIN en SQL)
                       # Mantiene todos los pedidos y añade info del cliente si la encuentra.
 )

print(df_completo.head())
```

#### `pd.concat()` (Concatenar)

* **¿Qué es?** Es la herramienta para **"pegar" o "apilar"** DataFrames, ya sea uno encima del otro (verticalmente) o uno al lado del otro (horizontalmente).
* **Analogía:** Piensa en `pd.concat()` como **pegar hojas de papel**.
  * **Unión Vertical (`axis=0`)**: Tienes dos hojas de asistencia (una de enero, otra de febrero) con las **mismas columnas** (Nombre, DNI, Asistencia). `pd.concat` las pega una debajo of the otra para crear una sola lista más larga (enero + febrero).
  * **Unión Horizontal (`axis=1`)**: Tienes una hoja con los *nombres* de los estudiantes y otra hoja con sus *notas*. Ambas tienen el **mismo índice** (el mismo orden). `pd.concat` las pega una al lado de la otra para crear una tabla más ancha.
* **Uso Principal:** Apilar datos que tienen la misma estructura (ej. `ventas_2024.csv` y `ventas_2025.csv`).

```python
# Ejemplo de Concatenación Vertical (la más común)

# Datos de ventas del primer trimestre
df_q1 = pd.DataFrame({
    'id_cliente': ['A', 'B', 'C'],
    'venta': [100, 200, 150]
})

# Datos de ventas del segundo trimestre
df_q2 = pd.DataFrame({
    'id_cliente': ['A', 'D', 'E'],
    'venta': [50, 300, 100]
})

# Concatenamos los dos trimestres para tener un historial completo
# ignore_index=True es importante para crear un nuevo índice (0, 1, 2, 3, 4, 5)
df_total_año = pd.concat([df_q1, df_q2], ignore_index=True)

print(df_total_año)
#  id_cliente  venta
#0          A    100
#1          B    200
#2          C    150
#3          A     50
#4          D    300
#5          E    100 
```

#### `df.join()` (Unir por Índice)

* **¿Qué es?** Es un atajo de `pd.merge()` que se especializa en unir DataFrames basándose en sus **índices (index)** en lugar de columnas.
* **Analogía:** Es un `VLOOKUP` (BUSCARV) de Excel donde la clave de búsqueda no es una columna, sino el **número de fila (o la etiqueta del índice)**.
* **Uso Principal:** Cuando tienes dos tablas donde las filas ya están alineadas por un índice común (ej. `id_cliente`) y quieres añadir columnas de una a la otra.
* **Diferencia clave con `merge`:** `df1.join(df2)` une el índice de `df1` con el índice de `df2` por defecto. `pd.merge()` une columnas de `df1` con columnas de `df2`.

```python
# Ejemplo de Join por Índice

# Datos de clientes, indexados por 'id_cliente'
df_clientes = pd.DataFrame({
    'nombre': ['Ana', 'Juan', 'Elena'],
    'email': ['ana@mail.com', 'juan@mail.com', 'elena@mail.com']
}, index=['c1', 'c2', 'c3'])
df_clientes.index.name = 'id_cliente'

# Datos demográficos, también indexados por 'id_cliente'
df_demograficos = pd.DataFrame({
    'edad': [28, 34, 45],
    'ciudad': ['Madrid', 'Lima', 'Bogotá']
}, index=['c1', 'c2', 'c3'])
df_demograficos.index.name = 'id_cliente'

# Usamos .join() para "pegar" las columnas de df_demograficos a df_clientes
# Como ambos tienen el mismo índice, es automático.
df_perfil_completo = df_clientes.join(df_demograficos)

print(df_perfil_completo)

#            nombre            email  edad  ciudad
#id_cliente
#c1            Ana     ana@mail.com    28  Madrid
#c2           Juan    juan@mail.com    34    Lima
#c3          Elena   elena@mail.com    45  Bogotá
```

**`merge` vs. `join` vs. `concat` (El Resumen)**

* pd.concat(): "Apilar" o "Pegar" (como apilar bloques de LEGO).
* pd.merge(): "Fusionar" (como un JOIN de SQL). Es la más potente y flexible. Se basa en columnas con valores comunes (ej. 'id_cliente' en ambas tablas).
* df.join(): "Unir" (como un VLOOKUP). Es un atajo para merge cuando la unión se basa en los índices (las etiquetas de las filas).

### Paso 8: El Dominio del Tiempo (Manejo de Fechas y Horas)

Los datos de series temporales deben ser tratados de forma especial. La herramienta clave es pd.to_datetime().

```python
# --- Paso Adicional: Manejo de Series de Tiempo ---

# Asumimos que df_completo tiene una columna 'fecha_compra' que es texto
print(df_completo.info()) # Mostraría 'fecha_compra' como 'object'

# 1. Convertir la columna de texto a formato datetime
df_completo['fecha_compra'] = pd.to_datetime(df_completo['fecha_compra'])

# 2. Ahora que es datetime, ¡podemos extraer partes de ella!
df_completo['mes_compra'] = df_completo['fecha_compra'].dt.month
df_completo['año_compra'] = df_completo['fecha_compra'].dt.year
df_completo['dia_semana'] = df_completo['fecha_compra'].dt.day_name()

print(df_completo[['fecha_compra', 'mes_compra', 'dia_semana']].head())

# 3. La magia: Establecer la fecha como el índice para análisis de tiempo
df_completo = df_completo.set_index('fecha_compra')

# Ahora puedes hacer cosas como "agrupar por mes" y sumar las ventas
df_ventas_mensuales = df_completo.resample('M')['total_venta'].sum()
print(df_ventas_mensuales)
```

### Paso 9: La Conclusión (Guardar y Comunicar)

Has encontrado la verdad. Ahora, guárdala y prepara tu informe.

```python
# --- Guardar tu DataFrame limpio para el futuro ---
# (index=False es VITAL para no guardar el índice como una columna)
df_limpio.to_csv('datos_google_ads_LIMPIO.csv', index=False)

# --- Guardar tu DataFrame agrupado (tu insight) ---
df_agrupado.to_csv('resumen_por_competencia.csv')

# --- Guardar tus Gráficos ---
# (Ejecuta esto en la celda de tu gráfico, justo ANTES de plt.show())
# plt.savefig('mi_grafico_de_barras.png', dpi=300) # dpi=300 para alta resolución

print("Análisis completado. Archivos limpios y de resultados guardados.")
```

### Paso 10: El Informe (La Comunicación)

El análisis no está completo hasta que se comunica. En tu notebook, añade una celda de Markdown al final y escribe tus conclusiones clave.

#### Reporte Ejecutivo: Análisis de Keywords de Google Ads

A continuación, se presentan los hallazgos clave del análisis:

* **Oportunidad Identificada:** Se detectó una tendencia emergente en la keyword "acceso financiero", con un crecimiento interanual (YoY) del **2000%** y una competencia clasificada como "Baja". Se recomienda la creación inmediata de contenido para capturar este interés.

* **Distribución de Búsquedas:** La gran mayoría de las keywords (mediana = 10 búsquedas) tienen un volumen bajo. La media (113 búsquedas) está fuertemente sesgada por unas pocas keywords de alto rendimiento.

* **Competencia vs. Crecimiento:** No se encontró una correlación clara entre el nivel de competencia de una keyword y su crecimiento interanual. Esto sugiere que las keywords de alta competencia no crecen necesariamente menos que las de baja competencia.

* **Calidad de Datos:** Se limpiaron y estandarizaron `X` columnas, incluyendo la conversión de `yoy_change` a formato numérico y el renombrado de 6 columnas para facilitar el análisis SQL.
