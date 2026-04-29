import pandas as pd

df = pd.read_csv('data_science_job_posts_2025.csv')

print(f"Número de muestras: {df.shape[0]}")
print(f"Número de variables: {df.shape[1]}")
print("Columnas detectadas:", df.columns.tolist())
