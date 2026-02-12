import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), "src", "app"))
from utils.data_loader import cargar_datos_completos

print("Cargando datos desde SQLite...")
df = cargar_datos_completos()
print(f"Datos cargados: {df.shape}")

output_path = "data/olist_processed.csv"
print(f"Guardando en {output_path}...")
df.to_csv(output_path, index=False)
print("¡Éxito!")
