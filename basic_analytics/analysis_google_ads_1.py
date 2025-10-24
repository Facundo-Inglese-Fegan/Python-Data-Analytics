# 1. Importar la librería Pandas
# Le damos el apodo 'pd', que es el estándar en todo el mundo.
import pandas as pd

print("Iniciando análisis de Google Ads...")

# 2. Definir el nombre de nuestro archivo
# Como el script y el CSV están en la misma carpeta, solo necesitamos el nombre.
archivo_csv = "data_google_ads.csv"

# 3. Cargar los datos en un DataFrame
# Usamos la función read_csv() de Pandas para leer el archivo.
# El resultado lo guardamos en una variable que llamaremos 'df' (la abreviatura de DataFrame).
try:
    df = pd.read_csv(archivo_csv)

    # 4. ¡La primera exploración! (El "Hola, Mundo" del Análisis de Datos)
    # Siempre, siempre, SIEMPRE, lo primero que hacemos es mirar los datos.
    
    print("\n--- 1. Vistazo a las primeras 5 filas (head): ---")
    # .head() nos muestra las primeras 5 filas para darnos una idea de la estructura.
    print(df.head())

    print("\n--- 2. Ficha técnica de los datos (info): ---")
    # .info() es VITAL. Nos dice:
    # - Cuántas filas y columnas hay.
    # - Los nombres de todas las columnas.
    # - Cuántos valores NO nulos hay (para detectar datos faltantes).
    # - El tipo de dato (dtype) de cada columna (si son números, texto, etc.).
    print(df.info())

    print("\n--- 3. Resumen estadístico (describe): ---")
    # .describe() nos da estadísticas rápidas de TODAS las columnas NUMÉRICAS.
    # (Conteo, media, desviación estándar, mínimo, máximo, etc.)
    print(df.describe())

    # --- 4. HACIENDO PREGUNTAS A LOS DATOS ---
    print("\n--- 4.1. ¿Cuáles son las 10 keywords con MÁS búsquedas? ---")
    # Usamos .sort_values() para ordenar el DataFrame por una columna.
    # ascending=False significa que queremos el orden de mayor a menor.
    df_top_busquedas = df.sort_values(by="Avg. monthly searches", ascending=False)
    # Imprimimos las primeras 10 filas de ese NUEVO DataFrame ordenado.
    print(df_top_busquedas[['Keyword', 'Avg. monthly searches']].head(10))

    print("\n--- 4.2. ¿Cuáles son las keywords que NO tienen búsquedas? ---")
    # Usamos un filtro.
    df_cero_busquedas = df[df["Avg. monthly searches"] == 0]
    print(df_cero_busquedas[['Keyword', 'Avg. monthly searches']])

    print("\n--- 4.3. ¿Cuántos datos nulos (vacíos) tiene cada columna? ---")
    # .isnull() crea una tabla de True/False, y .sum() suma los 'True'.
    print(df.isnull().sum())


    print("\n¡Datos cargados y explorados con éxito!")

except FileNotFoundError:
    print(f"¡ERROR! No se pudo encontrar el archivo: {archivo_csv}")
    print("Asegúrate de que el archivo .csv esté en la misma carpeta que tu script .py")