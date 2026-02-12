"""
Dashboard Completo de Análisis Olist
Aplicación modular con todas las secciones del análisis
"""

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import sys
import os

# Agregar el directorio al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar componentes
from components import (
    seccion_problema,
    seccion_limpieza,
    seccion_segmentacion,
    seccion_clasificacion,
    seccion_regresion,
    seccion_conclusiones,
    seccion_plan_accion,
)

# Inicializar la aplicación
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://use.fontawesome.com/releases/v6.4.0/css/all.css",  # Font Awesome
    ],
    suppress_callback_exceptions=True,
    title="Dashboard Olist - Análisis Completo",
)
server = app.server

# Layout principal
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H1(
                                    [
                                        html.I(className="fas fa-chart-line me-3"),
                                        "Dashboard de Análisis Olist",
                                    ],
                                    className="text-center text-primary mb-2 mt-4",
                                ),
                                html.P(
                                    "Análisis completo de datos: Diagnóstico, Limpieza, Segmentación, Clasificación y Regresión",
                                    className="text-center text-muted lead mb-4",
                                ),
                            ]
                        )
                    ]
                )
            ]
        ),
        html.Hr(),
        # Navegación con Tabs
        # Navegación con Tabs
        dbc.Tabs(
            [
                dbc.Tab(
                    seccion_problema.get_layout(),
                    label="Problema",
                    tab_id="tab-problema",
                    activeTabClassName="fw-bold",
                    className="text-start",
                ),
                dbc.Tab(
                    seccion_limpieza.get_layout(),
                    label="Limpieza",
                    tab_id="tab-limpieza",
                    activeTabClassName="fw-bold",
                    className="text-start",
                ),
                dbc.Tab(
                    seccion_segmentacion.get_layout(),
                    label="Segmentación",
                    tab_id="tab-segmentacion",
                    activeTabClassName="fw-bold",
                    className="text-start",
                ),
                dbc.Tab(
                    seccion_clasificacion.get_layout(),
                    label="Clasificación",
                    tab_id="tab-clasificacion",
                    activeTabClassName="fw-bold",
                    className="text-start",
                ),
                dbc.Tab(
                    seccion_regresion.get_layout(),
                    label="Regresión",
                    tab_id="tab-regresion",
                    activeTabClassName="fw-bold",
                    className="text-start",
                ),
                dbc.Tab(
                    seccion_conclusiones.get_layout(),
                    label="Conclusiones",
                    tab_id="tab-conclusiones",
                    activeTabClassName="fw-bold",
                    className="text-start",
                ),
                dbc.Tab(
                    seccion_plan_accion.get_layout(),
                    label="Plan de Acción",
                    tab_id="tab-plan",
                    activeTabClassName="fw-bold",
                    className="text-start",
                ),
            ],
            id="tabs",
            active_tab="tab-problema",
            className="mb-4",
        ),
        # Footer
        html.Footer(
            [
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.P(
                                    [
                                        "Dashboard desarrollado para análisis de datos Olist | ",
                                        html.Strong("Modelado y Presentación de Datos"),
                                        " | 2026",
                                    ],
                                    className="text-center text-muted small mb-4",
                                )
                            ]
                        )
                    ]
                ),
            ]
        ),
    ],
    fluid=True,
    style={"backgroundColor": "#1e1e1e", "minHeight": "100vh"},
)

# Registrar callbacks de los componentes
seccion_clasificacion.register_callbacks(app)
seccion_regresion.register_callbacks(app)

# Ejecutar la aplicación
if __name__ == "__main__":
    print("=" * 60)
    print("Iniciando Dashboard Olist...")
    print("=" * 60)
    print("Secciones disponibles:")
    print("   1. Problema (Diagnóstico)")
    print("   2. Limpieza de Datos")
    print("   3. Segmentación (PCA + Clustering)")
    print("   4. Clasificación (CART)")
    print("   5. Regresión (Ensamble)")
    print("   6. Conclusiones")
    print("   7. Plan de Acción")
    print("=" * 60)
    print("Abriendo en: http://127.0.0.1:8050")
    print("=" * 60)

    app.run(debug=True, host="127.0.0.1", port=8050)
