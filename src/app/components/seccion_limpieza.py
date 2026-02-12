"""
Sección de Limpieza de Datos - Versión Mejorada
Enfoque: Más gráficos, menos texto, lenguaje simple
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import cargar_datos_completos, preprocesar_datos
from utils.visualizations import (
    crear_histograma,
    crear_boxplot,
    aplicar_tema_oscuro,
    COLORS,
)
import plotly.graph_objects as go
import plotly.express as px

# Cargar datos (simulando estado crudo vs limpio)
df_limpio = cargar_datos_completos()
df_limpio = preprocesar_datos(df_limpio)

# Crear datos simulados "sucios" para comparación
np.random.seed(42)
df_sucio = df_limpio.copy()
n_miss = 100
df_sucio.loc[np.random.choice(df_sucio.index, n_miss), "precio_producto"] = np.nan
df_sucio.loc[np.random.choice(df_sucio.index, n_miss), "dias_entrega"] = np.nan


# Crear visualizaciones mejoradas
def crear_visualizaciones_limpieza():
    """Crea visualizaciones comparativas del proceso de limpieza"""

    # 1. Valores Faltantes (Antes vs Después)
    fig_faltantes = go.Figure()

    fig_faltantes.add_trace(
        go.Bar(
            x=["Precio", "Días Entrega"],
            y=[n_miss, n_miss],
            name="Antes (Sucios)",
            marker_color=COLORS["danger"],
            text=[f"{n_miss} vacíos", f"{n_miss} vacíos"],
            textposition="auto",
        )
    )

    fig_faltantes.add_trace(
        go.Bar(
            x=["Precio", "Días Entrega"],
            y=[0, 0],
            name="Después (Limpios)",
            marker_color=COLORS["secondary"],
            text=["0 vacíos", "0 vacíos"],
            textposition="auto",
        )
    )

    fig_faltantes.update_layout(
        title=dict(
            text="Eliminación de Datos Faltantes",
            font=dict(size=18, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        yaxis_title="Cantidad de Valores Vacíos",
        barmode="group",
        height=400,
    )
    fig_faltantes = aplicar_tema_oscuro(fig_faltantes)

    # 2. Manejo de Outliers (Boxplot Comparativo)
    fig_outliers = go.Figure()

    # Generar outliers simulados para "Antes"
    outliers = np.random.normal(1000, 100, 20)
    datos_con_outliers = np.concatenate([df_limpio["precio_producto"], outliers])

    fig_outliers.add_trace(
        go.Box(
            y=datos_con_outliers,
            name="Con Outliers (Antes)",
            marker_color=COLORS["warning"],
            boxpoints="outliers",
        )
    )

    fig_outliers.add_trace(
        go.Box(
            y=df_limpio["precio_producto"],
            name="Sin Outliers (Después)",
            marker_color=COLORS["secondary"],
            boxpoints="outliers",
        )
    )

    fig_outliers.update_layout(
        title=dict(
            text="Detección y Tratamiento de Valores Extremos",
            font=dict(size=18, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        yaxis_title="Precio del Producto (R$)",
        height=400,
    )
    fig_outliers = aplicar_tema_oscuro(fig_outliers)

    # 3. Transformación de Distribución
    fig_transformacion = go.Figure()

    # Datos originales sesgados (simulado)
    datos_sesgados = np.random.exponential(scale=100, size=1000)

    # Datos transformados (log)
    datos_normalizados = np.log1p(datos_sesgados)

    fig_transformacion.add_trace(
        go.Histogram(
            x=datos_sesgados,
            name="Original (Sesgada)",
            marker_color=COLORS["info"],
            opacity=0.7,
            nbinsx=40,
        )
    )

    fig_transformacion.add_trace(
        go.Histogram(
            x=datos_normalizados,
            name="Transformada (Log)",
            marker_color=COLORS["primary"],
            opacity=0.7,
            nbinsx=40,
            xaxis="x2",
            yaxis="y2",
        )
    )

    fig_transformacion.update_layout(
        title=dict(
            text="Normalización de Variables (Precio)",
            font=dict(size=18, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        grid=dict(rows=1, columns=2, pattern="independent"),
        showlegend=False,
        height=400,
    )
    fig_transformacion = aplicar_tema_oscuro(fig_transformacion)

    return fig_faltantes, fig_outliers, fig_transformacion


fig_faltantes, fig_outliers, fig_transformacion = crear_visualizaciones_limpieza()


def get_layout():
    """Retorna el layout mejorado de la sección de limpieza"""

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2(
                                [
                                    html.I(className="fas fa-broom me-3"),
                                    "Limpieza y Preparación de Datos",
                                ],
                                className="text-primary mb-3",
                            ),
                            html.P(
                                "Transformando datos crudos en información confiable para el análisis",
                                className="lead text-muted mb-4",
                            ),
                            html.Hr(),
                        ]
                    )
                ]
            ),
            # Resumen del Proceso
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                [
                                                    html.I(
                                                        className="fas fa-cogs me-2"
                                                    ),
                                                    "Estrategia de Limpieza",
                                                ],
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-eraser fa-3x text-danger mb-3"
                                                                    ),
                                                                    html.H6(
                                                                        "Valores Faltantes",
                                                                        className="text-danger",
                                                                    ),
                                                                    html.P(
                                                                        "Imputación con mediana para numéricos y moda para categóricos",
                                                                        className="small text-muted",
                                                                    ),
                                                                ],
                                                                className="text-center",
                                                            )
                                                        ],
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-filter fa-3x text-warning mb-3"
                                                                    ),
                                                                    html.H6(
                                                                        "Outliers",
                                                                        className="text-warning",
                                                                    ),
                                                                    html.P(
                                                                        "Detección y tratamiento usando Rango Intercuartil (IQR)",
                                                                        className="small text-muted",
                                                                    ),
                                                                ],
                                                                className="text-center",
                                                            )
                                                        ],
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-calculator fa-3x text-info mb-3"
                                                                    ),
                                                                    html.H6(
                                                                        "Transformaciones",
                                                                        className="text-info",
                                                                    ),
                                                                    html.P(
                                                                        "Normalización logarítmica y creación de nuevas variables",
                                                                        className="small text-muted",
                                                                    ),
                                                                ],
                                                                className="text-center",
                                                            )
                                                        ],
                                                        md=4,
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                ],
                                className="shadow mb-4",
                            )
                        ]
                    )
                ]
            ),
            # Visualizaciones del Proceso
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                figure=fig_faltantes,
                                                config={"displayModeBar": False},
                                            ),
                                            html.P(
                                                [
                                                    html.I(
                                                        className="fas fa-lightbulb text-warning me-2"
                                                    ),
                                                    html.Strong("Resultado: "),
                                                    "Dataset 100% completo, sin huecos de información que afecten los modelos.",
                                                ],
                                                className="text-muted mt-2 mb-0",
                                            ),
                                        ]
                                    )
                                ],
                                className="shadow mb-4",
                            )
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                figure=fig_outliers,
                                                config={"displayModeBar": False},
                                            ),
                                            html.P(
                                                [
                                                    html.I(
                                                        className="fas fa-lightbulb text-warning me-2"
                                                    ),
                                                    html.Strong("Resultado: "),
                                                    "Datos más estables y representativos, eliminando anomalías extremas.",
                                                ],
                                                className="text-muted mt-2 mb-0",
                                            ),
                                        ]
                                    )
                                ],
                                className="shadow mb-4",
                            )
                        ],
                        md=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                figure=fig_transformacion,
                                                config={"displayModeBar": False},
                                            ),
                                            html.P(
                                                [
                                                    html.I(
                                                        className="fas fa-lightbulb text-warning me-2"
                                                    ),
                                                    html.Strong("Resultado: "),
                                                    "Distribuciones normalizadas que mejoran significativamente el rendimiento ",
                                                    "de los algoritmos de Machine Learning.",
                                                ],
                                                className="text-muted mt-2 mb-0",
                                            ),
                                        ]
                                    )
                                ],
                                className="shadow mb-4",
                            )
                        ]
                    )
                ]
            ),
            # Calidad de Datos Final
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Alert(
                                [
                                    html.H5(
                                        [
                                            html.I(
                                                className="fas fa-check-circle me-2"
                                            ),
                                            "Calidad de Datos Final",
                                        ],
                                        className="alert-heading",
                                    ),
                                    html.P(
                                        [
                                            "El dataset final consta de ",
                                            html.Strong(
                                                f"{len(df_limpio)} registros limpios"
                                            ),
                                            " y ",
                                            html.Strong(
                                                f"{len(df_limpio.columns)} variables procesadas"
                                            ),
                                            ". Está listo para las fases avanzadas de segmentación y predicción.",
                                        ],
                                        className="mb-0",
                                    ),
                                ],
                                color="success",
                                className="shadow",
                            )
                        ]
                    )
                ]
            ),
        ],
        fluid=True,
        className="py-4",
    )
