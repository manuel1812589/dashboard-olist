"""
Utilidades para carga y preprocesamiento de datos
"""

import numpy as np
import pandas as pd
import sqlite3
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

DB_PATH = "data/olist.sqlite"


def cargar_datos_completos(db_path=DB_PATH, limit=3000):
    """Carga datos completos, priorizando CSV optimizado para web"""
    # 1. Intentar cargar desde CSV (optimizado para despliegue)
    csv_path = "data/olist_processed.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Datos cargados desde CSV: {df.shape[0]} filas")
            return df
        except Exception as e:
            print(f"⚠️ Error leyendo CSV: {e}")

    # 2. Intentar cargar desde SQLite (entorno local)
    try:
        conn = sqlite3.connect(db_path)
        query = """
        SELECT 
            oi.price AS precio_producto,
            oi.freight_value AS costo_envio,
            p.product_category_name AS categoria_producto,
            p.product_weight_g AS peso_producto,
            p.product_photos_qty AS fotos_producto,
            s.seller_city AS ciudad_vendedor,
            s.seller_state AS estado_vendedor,
            c.customer_state AS estado_cliente,
            r.review_score AS puntuacion_satisfaccion,
            r.review_comment_message AS comentario,
            (julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)) AS dias_entrega,
            (julianday(o.order_estimated_delivery_date) - julianday(o.order_delivered_customer_date)) AS diferencia_estimado_real,
            CAST(strftime('%m', o.order_purchase_timestamp) AS INTEGER) AS mes_compra,
            CAST(strftime('%Y', o.order_purchase_timestamp) AS INTEGER) AS anio_compra,
            CAST(strftime('%w', o.order_purchase_timestamp) AS INTEGER) AS dia_semana_compra,
            CAST(strftime('%H', o.order_purchase_timestamp) AS INTEGER) AS hora_compra,
            o.order_status AS estado_orden
        FROM orders o
        JOIN order_reviews r ON o.order_id = r.order_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        JOIN sellers s ON oi.seller_id = s.seller_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_delivered_customer_date IS NOT NULL 
          AND r.review_score IS NOT NULL
          AND oi.price IS NOT NULL
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        print(f"✅ Datos cargados desde DB: {df.shape[0]} filas")
    except Exception as e:
        print(f"⚠️ Error cargando BD: {e}. Usando datos sintéticos.")
        df = generar_datos_sinteticos(limit)
    return df


def generar_datos_sinteticos(n=3000):
    """Genera datos sintéticos para demostración"""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "precio_producto": np.random.exponential(100, n),
            "costo_envio": np.random.uniform(10, 50, n),
            "categoria_producto": np.random.choice(
                [
                    "cama_mesa_banho",
                    "beleza_saude",
                    "esporte_lazer",
                    "informatica_acessorios",
                    "moveis_decoracao",
                ],
                n,
            ),
            "peso_producto": np.random.uniform(200, 2000, n),
            "fotos_producto": np.random.randint(1, 8, n),
            "ciudad_vendedor": np.random.choice(
                ["sao paulo", "rio de janeiro", "belo horizonte"], n
            ),
            "estado_vendedor": np.random.choice(["SP", "RJ", "MG"], n),
            "estado_cliente": np.random.choice(["SP", "RJ", "MG", "RS", "PR"], n),
            "puntuacion_satisfaccion": np.clip(
                4.5
                - 0.01 * np.random.exponential(100, n)
                - 0.1 * np.random.gamma(3, 2, n)
                + np.random.normal(0, 0.5, n),
                1,
                5,
            ),
            "comentario": np.random.choice(
                [None, "bom", "ótimo", "ruim"], n, p=[0.3, 0.3, 0.3, 0.1]
            ),
            "dias_entrega": np.random.gamma(3, 2, n),
            "diferencia_estimado_real": np.random.normal(0, 2, n),
            "mes_compra": np.random.randint(1, 13, n),
            "anio_compra": np.random.choice([2017, 2018], n),
            "dia_semana_compra": np.random.randint(0, 7, n),
            "hora_compra": np.random.randint(0, 24, n),
            "estado_orden": "delivered",
        }
    )
    return df


def preprocesar_datos(df):
    """Aplica preprocesamiento estándar a los datos"""
    df = df.copy()

    # Tratamiento de valores nulos
    df["dias_entrega"] = df["dias_entrega"].fillna(df["dias_entrega"].median())
    df["diferencia_estimado_real"] = df["diferencia_estimado_real"].fillna(0)
    df["categoria_producto"] = df["categoria_producto"].fillna("desconocida")
    df["ciudad_vendedor"] = df["ciudad_vendedor"].fillna("desconocido")
    df["estado_vendedor"] = df["estado_vendedor"].fillna("desconocido")
    df["estado_cliente"] = df["estado_cliente"].fillna("desconocido")
    df["peso_producto"] = df["peso_producto"].fillna(df["peso_producto"].median())
    df["fotos_producto"] = df["fotos_producto"].fillna(df["fotos_producto"].median())
    df["comentario"] = df["comentario"].fillna("")

    # Feature engineering
    df["tiene_comentario"] = (df["comentario"] != "").astype(int)
    df["valor_total_pedido"] = df["precio_producto"] + df["costo_envio"]
    df["ratio_precio_envio"] = df["precio_producto"] / (df["costo_envio"] + 0.01)
    df["entrega_tardia"] = (df["diferencia_estimado_real"] < 0).astype(int)
    df["mismo_estado"] = (df["estado_vendedor"] == df["estado_cliente"]).astype(int)
    df["cliente_satisfecho"] = (df["puntuacion_satisfaccion"] >= 4).astype(int)

    # Segmentación de precio
    df["segmento_precio"] = pd.cut(
        df["precio_producto"],
        bins=[0, 50, 150, 500, float("inf")],
        labels=["Muy Bajo", "Bajo", "Medio", "Alto"],
    )

    # Transformaciones logarítmicas
    df["precio_producto_log"] = np.log1p(df["precio_producto"])
    df["costo_envio_log"] = np.log1p(df["costo_envio"])
    df["dias_entrega_log"] = np.log1p(df["dias_entrega"])
    df["peso_producto_log"] = np.log1p(df["peso_producto"])

    return df


def obtener_datos_escalados(df, features):
    """Escala features específicos usando StandardScaler"""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled, scaler


def codificar_categoricas(df, columnas):
    """Codifica variables categóricas usando LabelEncoder"""
    df_encoded = df.copy()
    encoders = {}
    for col in columnas:
        if col in df.columns:
            le = LabelEncoder()
            df_encoded[f"{col}_encoded"] = le.fit_transform(df[col])
            encoders[col] = le
    return df_encoded, encoders
