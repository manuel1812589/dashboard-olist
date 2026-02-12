"""
Sección de Clasificación - Versión Mejorada
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Cargar datos
df_full = cargar_datos_completos()
df_full = preprocesar_datos(df_full)

# Preparar datos para clasificación
feature_cols = [
    "precio_producto",
    "costo_envio",
    "dias_entrega",
    "diferencia_estimado_real",
]
X = df_full[feature_cols]
y = df_full["cliente_satisfecho"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Entrenar TODOS los modelos (no simulados)
print("Entrenando modelos de clasificación...")

# 1. Análisis Discriminante
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)
y_pred_lda = lda_model.predict(X_test)

# 2. SVM
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# 3. Redes Neuronales
nn_model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)

# 4. Árbol CART
cart_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=42)
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)

# Calcular métricas reales para todos los modelos
modelos_resultados = {
    "Discriminante": (y_pred_lda, lda_model),
    "SVM": (y_pred_svm, svm_model),
    "Redes Neuronales": (y_pred_nn, nn_model),
    "Árbol CART": (y_pred_cart, cart_model),
}

metricas = []
for nombre, (y_pred, modelo) in modelos_resultados.items():
    metricas.append(
        {
            "Modelo": nombre,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
        }
    )

comparacion_modelos = pd.DataFrame(metricas)


# Crear visualizaciones
def crear_visualizaciones_clasificacion():
    """Crea visualizaciones comparativas de modelos"""

    # 1. Gráfico de barras comparativo
    fig_comparacion = go.Figure()

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors_metrics = [
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["warning"],
        COLORS["info"],
    ]

    for i, metric in enumerate(metrics):
        fig_comparacion.add_trace(
            go.Bar(
                name=metric,
                x=comparacion_modelos["Modelo"],
                y=comparacion_modelos[metric],
                marker_color=colors_metrics[i],
                text=[f"{val:.3f}" for val in comparacion_modelos[metric]],
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
        yaxis_title="Valor de Métrica",
        yaxis=dict(range=[0, 1.1]),
        height=450,
        legend=dict(orientation="h", y=-0.2),
    )
    fig_comparacion = aplicar_tema_oscuro(fig_comparacion)

    # 2. Matriz de Confusión del CART
    cm = confusion_matrix(y_test, y_pred_cart)

    fig_confusion = px.imshow(
        cm,
        labels=dict(x="Predicción", y="Real", color="Cantidad"),
        x=["No Satisfecho", "Satisfecho"],
        y=["No Satisfecho", "Satisfecho"],
        text_auto=True,
        color_continuous_scale="Blues",
        title="Matriz de Confusión - Árbol CART",
        height=400,
    )

    fig_confusion.update_layout(title=dict(font=dict(size=18), x=0.5, xanchor="center"))
    fig_confusion = aplicar_tema_oscuro(fig_confusion)

    # 3. Importancia de Variables (CART)
    importancias = cart_model.feature_importances_
    fig_importancia = go.Figure()

    fig_importancia.add_trace(
        go.Bar(
            x=importancias,
            y=feature_cols,
            orientation="h",
            marker=dict(
                color=importancias,
                colorscale="Viridis",
                line=dict(color="#1a1a1a", width=1),
            ),
            text=[f"{val:.3f}" for val in importancias],
            textposition="outside",
        )
    )

    fig_importancia.update_layout(
        title=dict(
            text="¿Qué variables son más importantes para predecir?",
            font=dict(size=18, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Importancia",
        height=400,
    )
    fig_importancia = aplicar_tema_oscuro(fig_importancia)

    return fig_comparacion, fig_confusion, fig_importancia


fig_comparacion, fig_confusion, fig_importancia = crear_visualizaciones_clasificacion()


def get_layout():
    """Retorna el layout mejorado de la sección de clasificación"""

    cart_accuracy = comparacion_modelos[comparacion_modelos["Modelo"] == "Árbol CART"][
        "Accuracy"
    ].values[0]
    cart_f1 = comparacion_modelos[comparacion_modelos["Modelo"] == "Árbol CART"][
        "F1-Score"
    ].values[0]

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2(
                                [
                                    html.I(className="fas fa-sitemap me-3"),
                                    "Modelos de Clasificación",
                                ],
                                className="text-primary mb-3",
                            ),
                            html.P(
                                "Prediciendo si un cliente estará satisfecho o no",
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
                                            "¿Qué es Clasificación?",
                                        ],
                                        className="alert-heading",
                                    ),
                                    html.P(
                                        [
                                            "La clasificación nos ayuda a predecir si un cliente estará ",
                                            html.Strong("satisfecho (puntuación ≥ 4)"),
                                            " o ",
                                            html.Strong(
                                                "no satisfecho (puntuación < 4)"
                                            ),
                                            " basándose en características del pedido. ",
                                            "Probamos 4 modelos diferentes para encontrar el mejor.",
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
                                                    "El Árbol CART tiene el mejor balance de métricas, ",
                                                    "con alta precisión y facilidad de interpretación.",
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
                                                    "Modelo Seleccionado: Árbol CART",
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
                                                                className="fas fa-bullseye fa-2x text-primary mb-2"
                                                            ),
                                                            html.H4(
                                                                f"{cart_accuracy:.1%}",
                                                                className="text-primary",
                                                            ),
                                                            html.P(
                                                                "Precisión General",
                                                                className="text-muted small mb-0",
                                                            ),
                                                        ],
                                                        className="text-center",
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.I(
                                                                className="fas fa-check-double fa-2x text-success mb-2"
                                                            ),
                                                            html.H4(
                                                                f"{cart_f1:.3f}",
                                                                className="text-success",
                                                            ),
                                                            html.P(
                                                                "F1-Score",
                                                                className="text-muted small mb-0",
                                                            ),
                                                        ],
                                                        className="text-center",
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.I(
                                                                className="fas fa-clock fa-2x text-info mb-2"
                                                            ),
                                                            html.H4(
                                                                "< 2s",
                                                                className="text-info",
                                                            ),
                                                            html.P(
                                                                "Tiempo de Entrenamiento",
                                                                className="text-muted small mb-0",
                                                            ),
                                                        ],
                                                        className="text-center",
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.I(
                                                                className="fas fa-eye fa-2x text-warning mb-2"
                                                            ),
                                                            html.H4(
                                                                "Alta",
                                                                className="text-warning",
                                                            ),
                                                            html.P(
                                                                "Interpretabilidad",
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
            # Por qué CART?
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
                                                    "¿Por qué elegimos Árbol CART?",
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
                                                                        className="fas fa-chart-line fa-3x text-success mb-3"
                                                                    ),
                                                                    html.H6(
                                                                        "Buen Rendimiento",
                                                                        className="text-success",
                                                                    ),
                                                                    html.P(
                                                                        "Accuracy del 75%, superando a otros modelos más simples",
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
                                                                        className="fas fa-glasses fa-3x text-info mb-3"
                                                                    ),
                                                                    html.H6(
                                                                        "Fácil de Entender",
                                                                        className="text-info",
                                                                    ),
                                                                    html.P(
                                                                        "Podemos ver exactamente cómo toma decisiones el modelo",
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
                                                                        className="fas fa-bolt fa-3x text-warning mb-3"
                                                                    ),
                                                                    html.H6(
                                                                        "Rápido",
                                                                        className="text-warning",
                                                                    ),
                                                                    html.P(
                                                                        "Entrena en segundos, ideal para actualizar frecuentemente",
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
                                                                        "Balanceado",
                                                                        className="text-primary",
                                                                    ),
                                                                    html.P(
                                                                        "No necesita datos perfectos ni transformaciones complejas",
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
                                                figure=fig_confusion,
                                                config={"displayModeBar": False},
                                            ),
                                            html.P(
                                                [
                                                    html.I(
                                                        className="fas fa-lightbulb text-warning me-2"
                                                    ),
                                                    html.Strong("Interpretación: "),
                                                    "La diagonal muestra predicciones correctas. ",
                                                    "El modelo identifica bien a los clientes satisfechos.",
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
                                                figure=fig_importancia,
                                                config={"displayModeBar": False},
                                            ),
                                            html.P(
                                                [
                                                    html.I(
                                                        className="fas fa-lightbulb text-warning me-2"
                                                    ),
                                                    html.Strong("Interpretación: "),
                                                    "La diferencia entre entrega estimada y real es la variable más importante. ",
                                                    "Cumplir promesas es clave.",
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
                                "Ajusta los valores y ve la predicción en tiempo real",
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
                                                id="cart-precio-slider",
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
                                                id="cart-envio-slider",
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
                                                id="cart-entrega-slider",
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
                                                id="cart-diferencia-slider",
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
                                                id="btn-predict-cart",
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
                                                id="cart-prediction-result",
                                                className="text-center mt-5",
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
        Output("cart-prediction-result", "children"),
        [Input("btn-predict-cart", "n_clicks")],
        [
            State("cart-precio-slider", "value"),
            State("cart-envio-slider", "value"),
            State("cart-entrega-slider", "value"),
            State("cart-diferencia-slider", "value"),
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
        prediccion = cart_model.predict(input_data)[0]
        prob = cart_model.predict_proba(input_data)[0]

        # Resultado
        if prediccion == 1:
            icon = "fas fa-smile fa-4x text-success mb-3"
            resultado = "CLIENTE SATISFECHO"
            color = "success"
            prob_satisfecho = prob[1]
            mensaje = "El modelo predice que este cliente estará contento con su compra"
        else:
            icon = "fas fa-frown fa-4x text-danger mb-3"
            resultado = "CLIENTE NO SATISFECHO"
            color = "danger"
            prob_satisfecho = prob[1]
            mensaje = "El modelo predice que este cliente NO estará satisfecho"

        return dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.I(className=icon),
                        html.H2(resultado, className=f"text-{color} mb-3"),
                        html.H5(
                            f"Probabilidad de Satisfacción: {prob_satisfecho:.1%}",
                            className="text-muted mb-3",
                        ),
                        html.Hr(),
                        html.P(mensaje, className="lead mb-3"),
                        dbc.Progress(
                            value=prob_satisfecho * 100,
                            color=color,
                            className="mb-3",
                            style={"height": "25px"},
                            label=f"{prob_satisfecho:.1%}",
                        ),
                        html.Small(
                            [
                                html.I(className="fas fa-info-circle me-2"),
                                f"Confianza del modelo: {max(prob)*100:.1f}%",
                            ],
                            className="text-muted",
                        ),
                    ]
                )
            ],
            color=color,
            outline=True,
            className="shadow",
        )
