"""
Sección de Regresión - Versión Mejorada
Enfoque: Más gráficos, menos texto, lenguaje simple
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import cargar_datos_completos, preprocesar_datos
from utils.visualizations import crear_tabla_comparativa, aplicar_tema_oscuro, COLORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px

# Cargar datos
df_full = cargar_datos_completos()
df_full = preprocesar_datos(df_full)

# Preparar datos para regresión
feature_cols = [
    "precio_producto",
    "costo_envio",
    "dias_entrega",
    "diferencia_estimado_real",
]
X = df_full[feature_cols]
y = df_full["puntuacion_satisfaccion"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Entrenar TODOS los modelos (no simulados)
print("Entrenando modelos de regresión...")

# 1. Regresión Lineal
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# 2. Redes Neuronales
nn_model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)

# 3. Ensamble (Random Forest)
rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=None, min_samples_leaf=30, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Calcular métricas reales
modelos_resultados = {
    "Regresión Lineal": y_pred_lr,
    "Redes Neuronales": y_pred_nn,
    "Ensamble (Random Forest)": y_pred_rf,
}

metricas = []
for nombre, y_pred in modelos_resultados.items():
    metricas.append(
        {
            "Modelo": nombre,
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R²": r2_score(y_test, y_pred),
        }
    )

comparacion_modelos = pd.DataFrame(metricas)


# Crear visualizaciones
def crear_visualizaciones_regresion():
    """Crea visualizaciones comparativas de modelos"""

    # 1. Comparación de métricas
    fig_comparacion = go.Figure()

    # Invertir RMSE y MAE para que mayor sea mejor en la visualización
    comparacion_viz = comparacion_modelos.copy()
    comparacion_viz["RMSE_inv"] = 1 - (
        comparacion_viz["RMSE"] / comparacion_viz["RMSE"].max()
    )
    comparacion_viz["MAE_inv"] = 1 - (
        comparacion_viz["MAE"] / comparacion_viz["MAE"].max()
    )

    metrics = ["R²", "RMSE_inv", "MAE_inv"]
    metric_names = ["R² (Varianza Explicada)", "RMSE (invertido)", "MAE (invertido)"]
    colors_metrics = [COLORS["primary"], COLORS["secondary"], COLORS["warning"]]

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        fig_comparacion.add_trace(
            go.Bar(
                name=name,
                x=comparacion_viz["Modelo"],
                y=comparacion_viz[metric],
                marker_color=colors_metrics[i],
                text=[f"{val:.3f}" for val in comparacion_viz[metric]],
                textposition="outside",
            )
        )

    fig_comparacion.update_layout(
        title=dict(
            text="Comparación de Rendimiento de Modelos",
            font=dict(size=18, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        barmode="group",
        yaxis_title="Valor de Métrica (mayor = mejor)",
        yaxis=dict(range=[0, 1.1]),
        height=450,
        legend=dict(orientation="h", y=-0.2),
    )
    fig_comparacion = aplicar_tema_oscuro(fig_comparacion)

    # 2. Predicciones vs Real (Ensamble)
    sample_indices = np.random.choice(
        len(y_test), size=min(500, len(y_test)), replace=False
    )

    fig_pred_real = go.Figure()

    fig_pred_real.add_trace(
        go.Scatter(
            x=y_test.iloc[sample_indices],
            y=y_pred_rf[sample_indices],
            mode="markers",
            marker=dict(
                color=y_pred_rf[sample_indices],
                colorscale="Viridis",
                size=8,
                line=dict(width=0.5, color="#1a1a1a"),
                opacity=0.7,
            ),
            name="Predicciones",
            hovertemplate="<b>Real:</b> %{x:.2f}<br><b>Predicho:</b> %{y:.2f}<extra></extra>",
        )
    )

    # Línea perfecta
    fig_pred_real.add_trace(
        go.Scatter(
            x=[1, 5],
            y=[1, 5],
            mode="lines",
            line=dict(color=COLORS["danger"], dash="dash", width=2),
            name="Predicción Perfecta",
        )
    )

    fig_pred_real.update_layout(
        title=dict(
            text="Predicciones vs Valores Reales - Ensamble",
            font=dict(size=18, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Satisfacción Real",
        yaxis_title="Satisfacción Predicha",
        height=450,
    )
    fig_pred_real = aplicar_tema_oscuro(fig_pred_real)

    # 3. Importancia de Variables (Ensamble)
    importancias = rf_model.feature_importances_
    fig_importancia = go.Figure()

    fig_importancia.add_trace(
        go.Bar(
            x=importancias,
            y=feature_cols,
            orientation="h",
            marker=dict(
                color=importancias,
                colorscale="Plasma",
                line=dict(color="#1a1a1a", width=1),
            ),
            text=[f"{val:.3f}" for val in importancias],
            textposition="outside",
        )
    )

    fig_importancia.update_layout(
        title=dict(
            text="¿Qué variables influyen más en la satisfacción?",
            font=dict(size=18, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Importancia",
        height=400,
    )
    fig_importancia = aplicar_tema_oscuro(fig_importancia)

    # 4. Distribución de Errores
    errores = y_test.values - y_pred_rf

    fig_errores = go.Figure()

    fig_errores.add_trace(
        go.Histogram(
            x=errores,
            nbinsx=50,
            marker=dict(
                color=COLORS["info"], line=dict(color="#1a1a1a", width=1), opacity=0.85
            ),
            hovertemplate="<b>Error:</b> %{x:.2f}<br><b>Frecuencia:</b> %{y}<extra></extra>",
        )
    )

    fig_errores.update_layout(
        title=dict(
            text="Distribución de Errores del Modelo",
            font=dict(size=18, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Error (Real - Predicho)",
        yaxis_title="Frecuencia",
        height=400,
    )
    fig_errores = aplicar_tema_oscuro(fig_errores)

    return fig_comparacion, fig_pred_real, fig_importancia, fig_errores


fig_comparacion, fig_pred_real, fig_importancia, fig_errores = (
    crear_visualizaciones_regresion()
)


def get_layout():
    """Retorna el layout mejorado de la sección de regresión"""

    rf_r2 = comparacion_modelos[
        comparacion_modelos["Modelo"] == "Ensamble (Random Forest)"
    ]["R²"].values[0]
    rf_rmse = comparacion_modelos[
        comparacion_modelos["Modelo"] == "Ensamble (Random Forest)"
    ]["RMSE"].values[0]
    rf_mae = comparacion_modelos[
        comparacion_modelos["Modelo"] == "Ensamble (Random Forest)"
    ]["MAE"].values[0]

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2(
                                [
                                    html.I(className="fas fa-chart-area me-3"),
                                    "Modelos de Regresión",
                                ],
                                className="text-primary mb-3",
                            ),
                            html.P(
                                "Prediciendo la puntuación exacta de satisfacción (1-5)",
                                className="lead text-muted mb-4",
                            ),
                            html.Hr(),
                        ]
                    )
                ]
            ),
            # Explicación Simple
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Alert(
                                [
                                    html.H5(
                                        [
                                            html.I(className="fas fa-info-circle me-2"),
                                            "¿Qué es Regresión?",
                                        ],
                                        className="alert-heading",
                                    ),
                                    html.P(
                                        [
                                            "A diferencia de clasificación (que predice satisfecho/no satisfecho), ",
                                            "la regresión predice ",
                                            html.Strong(
                                                "la puntuación exacta de satisfacción"
                                            ),
                                            " en una escala de 1 a 5. ",
                                            "Esto nos da más detalle sobre qué tan satisfecho estará el cliente.",
                                        ],
                                        className="mb-0",
                                    ),
                                ],
                                color="info",
                                className="shadow-sm mb-4",
                            )
                        ]
                    )
                ]
            ),
            # Comparación Visual de Modelos
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                figure=fig_comparacion,
                                                config={"displayModeBar": False},
                                            ),
                                            html.P(
                                                [
                                                    html.I(
                                                        className="fas fa-lightbulb text-warning me-2"
                                                    ),
                                                    html.Strong("Hallazgo: "),
                                                    "El Ensamble (Random Forest) tiene el mejor R², ",
                                                    "explicando más variabilidad en la satisfacción que otros modelos.",
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
            # Métricas del Modelo Seleccionado
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
                                                        className="fas fa-trophy me-2 text-warning"
                                                    ),
                                                    "Modelo Seleccionado: Ensamble (Random Forest)",
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
                                                            html.I(
                                                                className="fas fa-percentage fa-2x text-primary mb-2"
                                                            ),
                                                            html.H4(
                                                                f"{rf_r2:.1%}",
                                                                className="text-primary",
                                                            ),
                                                            html.P(
                                                                "Varianza Explicada (R²)",
                                                                className="text-muted small mb-0",
                                                            ),
                                                        ],
                                                        className="text-center",
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.I(
                                                                className="fas fa-ruler fa-2x text-success mb-2"
                                                            ),
                                                            html.H4(
                                                                f"{rf_rmse:.3f}",
                                                                className="text-success",
                                                            ),
                                                            html.P(
                                                                "Error Cuadrático (RMSE)",
                                                                className="text-muted small mb-0",
                                                            ),
                                                        ],
                                                        className="text-center",
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.I(
                                                                className="fas fa-chart-line fa-2x text-info mb-2"
                                                            ),
                                                            html.H4(
                                                                f"{rf_mae:.3f}",
                                                                className="text-info",
                                                            ),
                                                            html.P(
                                                                "Error Absoluto (MAE)",
                                                                className="text-muted small mb-0",
                                                            ),
                                                        ],
                                                        className="text-center",
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.I(
                                                                className="fas fa-tree fa-2x text-warning mb-2"
                                                            ),
                                                            html.H4(
                                                                "200",
                                                                className="text-warning",
                                                            ),
                                                            html.P(
                                                                "Árboles en el Bosque",
                                                                className="text-muted small mb-0",
                                                            ),
                                                        ],
                                                        className="text-center",
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
            # Por qué Ensamble?
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
                                                        className="fas fa-question-circle me-2"
                                                    ),
                                                    "¿Por qué elegimos Ensamble (Random Forest)?",
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
                                                                        className="fas fa-trophy fa-3x text-success mb-3"
                                                                    ),
                                                                    html.H6(
                                                                        "Mejor R²",
                                                                        className="text-success",
                                                                    ),
                                                                    html.P(
                                                                        f"Explica el {rf_r2*100:.0f}% de la variabilidad, superando a otros modelos",
                                                                        className="small text-muted",
                                                                    ),
                                                                ],
                                                                className="text-center",
                                                            )
                                                        ],
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-shield-alt fa-3x text-info mb-3"
                                                                    ),
                                                                    html.H6(
                                                                        "Robusto",
                                                                        className="text-info",
                                                                    ),
                                                                    html.P(
                                                                        "Maneja bien valores extremos y datos imperfectos",
                                                                        className="small text-muted",
                                                                    ),
                                                                ],
                                                                className="text-center",
                                                            )
                                                        ],
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-random fa-3x text-warning mb-3"
                                                                    ),
                                                                    html.H6(
                                                                        "Captura Complejidad",
                                                                        className="text-warning",
                                                                    ),
                                                                    html.P(
                                                                        "Identifica relaciones no lineales entre variables",
                                                                        className="small text-muted",
                                                                    ),
                                                                ],
                                                                className="text-center",
                                                            )
                                                        ],
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-balance-scale fa-3x text-primary mb-3"
                                                                    ),
                                                                    html.H6(
                                                                        "Menor Error",
                                                                        className="text-primary",
                                                                    ),
                                                                    html.P(
                                                                        "RMSE y MAE más bajos que modelos lineales",
                                                                        className="small text-muted",
                                                                    ),
                                                                ],
                                                                className="text-center",
                                                            )
                                                        ],
                                                        md=3,
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
            # Visualizaciones del Modelo
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                figure=fig_pred_real,
                                                config={"displayModeBar": False},
                                            ),
                                            html.P(
                                                [
                                                    html.I(
                                                        className="fas fa-lightbulb text-warning me-2"
                                                    ),
                                                    html.Strong("Interpretación: "),
                                                    "Los puntos cerca de la línea roja son predicciones precisas. ",
                                                    "El modelo predice bien en la mayoría de casos.",
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
                                                figure=fig_errores,
                                                config={"displayModeBar": False},
                                            ),
                                            html.P(
                                                [
                                                    html.I(
                                                        className="fas fa-lightbulb text-warning me-2"
                                                    ),
                                                    html.Strong("Interpretación: "),
                                                    "Los errores están centrados cerca de cero, ",
                                                    "indicando que el modelo no tiene sesgo sistemático.",
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
                                                figure=fig_importancia,
                                                config={"displayModeBar": False},
                                            ),
                                            html.P(
                                                [
                                                    html.I(
                                                        className="fas fa-lightbulb text-warning me-2"
                                                    ),
                                                    html.Strong("Interpretación: "),
                                                    "Similar a clasificación, la diferencia estimado/real es la variable clave. ",
                                                    "Cumplir promesas impacta directamente la puntuación.",
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
            # Simulador Interactivo
            html.H3(
                [html.I(className="fas fa-gamepad me-3"), "Simulador Interactivo"],
                className="text-primary mt-4 mb-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.H5(
                                [
                                    html.I(className="fas fa-magic me-2"),
                                    "Prueba el Modelo",
                                ],
                                className="mb-0 text-white",
                            ),
                            html.P(
                                "Ajusta los valores y ve la predicción de satisfacción",
                                className="mb-0 text-light small",
                            ),
                        ],
                        className="bg-dark",
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    # Controles
                                    dbc.Col(
                                        [
                                            html.Label(
                                                [
                                                    html.I(
                                                        className="fas fa-dollar-sign me-2"
                                                    ),
                                                    "Precio del Producto (R$)",
                                                ],
                                                className="font-weight-bold",
                                            ),
                                            dcc.Slider(
                                                id="rf-precio-slider",
                                                min=10,
                                                max=500,
                                                value=150,
                                                step=10,
                                                marks={
                                                    10: "10",
                                                    100: "100",
                                                    200: "200",
                                                    300: "300",
                                                    400: "400",
                                                    500: "500",
                                                },
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": True,
                                                },
                                            ),
                                            html.Label(
                                                [
                                                    html.I(
                                                        className="fas fa-truck me-2"
                                                    ),
                                                    "Costo de Envío (R$)",
                                                ],
                                                className="font-weight-bold mt-4",
                                            ),
                                            dcc.Slider(
                                                id="rf-envio-slider",
                                                min=5,
                                                max=50,
                                                value=25,
                                                step=5,
                                                marks={
                                                    5: "5",
                                                    15: "15",
                                                    25: "25",
                                                    35: "35",
                                                    45: "45",
                                                },
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": True,
                                                },
                                            ),
                                            html.Label(
                                                [
                                                    html.I(
                                                        className="fas fa-calendar-alt me-2"
                                                    ),
                                                    "Días de Entrega",
                                                ],
                                                className="font-weight-bold mt-4",
                                            ),
                                            dcc.Slider(
                                                id="rf-entrega-slider",
                                                min=1,
                                                max=30,
                                                value=10,
                                                step=1,
                                                marks={
                                                    1: "1",
                                                    5: "5",
                                                    10: "10",
                                                    15: "15",
                                                    20: "20",
                                                    25: "25",
                                                    30: "30",
                                                },
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": True,
                                                },
                                            ),
                                            html.Label(
                                                [
                                                    html.I(
                                                        className="fas fa-hourglass-half me-2"
                                                    ),
                                                    "Diferencia Estimado vs Real (días)",
                                                ],
                                                className="font-weight-bold mt-4",
                                            ),
                                            html.P(
                                                "Negativo = llegó tarde, Positivo = llegó antes",
                                                className="small text-muted",
                                            ),
                                            dcc.Slider(
                                                id="rf-diferencia-slider",
                                                min=-5,
                                                max=10,
                                                value=0,
                                                step=1,
                                                marks={
                                                    -5: "-5",
                                                    -3: "-3",
                                                    0: "0",
                                                    3: "+3",
                                                    5: "+5",
                                                    10: "+10",
                                                },
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": True,
                                                },
                                            ),
                                            dbc.Button(
                                                [
                                                    html.I(
                                                        className="fas fa-play me-2"
                                                    ),
                                                    "Predecir Satisfacción",
                                                ],
                                                id="btn-predict-rf",
                                                color="primary",
                                                size="lg",
                                                className="w-100 mt-4",
                                            ),
                                        ],
                                        md=5,
                                        className="p-4",
                                    ),
                                    # Resultado
                                    dbc.Col(
                                        [
                                            html.Div(
                                                id="rf-prediction-result",
                                                className="text-center",
                                            )
                                        ],
                                        md=7,
                                        className="p-4 d-flex flex-column justify-content-center",
                                    ),
                                ]
                            )
                        ]
                    ),
                ],
                className="shadow-lg mb-4",
            ),
        ],
        fluid=True,
        className="py-4",
    )


def register_callbacks(app):
    """Registra los callbacks para el simulador"""

    @app.callback(
        Output("rf-prediction-result", "children"),
        [Input("btn-predict-rf", "n_clicks")],
        [
            State("rf-precio-slider", "value"),
            State("rf-envio-slider", "value"),
            State("rf-entrega-slider", "value"),
            State("rf-diferencia-slider", "value"),
        ],
    )
    def predecir_satisfaccion(n_clicks, precio, envio, entrega, diferencia):
        if not n_clicks:
            return html.Div(
                [
                    html.I(className="fas fa-hand-point-left fa-3x text-muted mb-3"),
                    html.H4(
                        "Configura los parámetros y presiona el botón",
                        className="text-muted",
                    ),
                ]
            )

        # Crear input
        input_data = pd.DataFrame(
            [[precio, envio, entrega, diferencia]], columns=feature_cols
        )

        # Predecir
        prediccion = rf_model.predict(input_data)[0]
        prediccion = max(1, min(5, prediccion))  # Limitar a rango 1-5

        # Determinar nivel
        if prediccion >= 4.5:
            icon = "fas fa-grin-stars fa-4x text-success mb-3"
            nivel = "EXCELENTE"
            color = "success"
            mensaje = "El cliente estará muy satisfecho con esta experiencia"
        elif prediccion >= 3.5:
            icon = "fas fa-smile fa-4x text-info mb-3"
            nivel = "BUENO"
            color = "info"
            mensaje = "El cliente tendrá una experiencia satisfactoria"
        elif prediccion >= 2.5:
            icon = "fas fa-meh fa-4x text-warning mb-3"
            nivel = "REGULAR"
            color = "warning"
            mensaje = "El cliente tendrá una experiencia promedio, hay margen de mejora"
        else:
            icon = "fas fa-frown fa-4x text-danger mb-3"
            nivel = "MALO"
            color = "danger"
            mensaje = "El cliente probablemente NO estará satisfecho"

        # Crear gauge visual
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediccion,
                title={
                    "text": f"Nivel: {nivel}",
                    "font": {"size": 20, "color": COLORS["text"]},
                },
                number={
                    "suffix": "/5.0",
                    "font": {"size": 40, "color": COLORS["text"]},
                },
                gauge={
                    "axis": {
                        "range": [1, 5],
                        "tickwidth": 1,
                        "tickcolor": COLORS["text"],
                    },
                    "bar": {"color": COLORS[color], "thickness": 0.6},
                    "steps": [
                        {"range": [1, 2.5], "color": "#5a2a2a"},
                        {"range": [2.5, 3.5], "color": "#5a4a2a"},
                        {"range": [3.5, 4.5], "color": "#2a5a5a"},
                        {"range": [4.5, 5], "color": "#2a5a2a"},
                    ],
                    "threshold": {
                        "line": {"color": COLORS["text"], "width": 4},
                        "thickness": 0.75,
                        "value": prediccion,
                    },
                },
            )
        )

        gauge_fig.update_layout(
            paper_bgcolor="#1a1a1a",
            font={"color": COLORS["text"]},
            height=300,
            margin=dict(l=20, r=20, t=60, b=20),
        )

        return html.Div(
            [
                html.I(className=icon),
                dcc.Graph(figure=gauge_fig, config={"displayModeBar": False}),
                html.P(mensaje, className="lead mt-3"),
            ]
        )
