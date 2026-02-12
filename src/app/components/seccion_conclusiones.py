"""
Sección de Conclusiones - Versión Mejorada
Enfoque: Más gráficos, menos texto, lenguaje simple
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualizations import COLORS


def get_layout():
    """Retorna el layout mejorado de la sección de conclusiones"""

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2(
                                [
                                    html.I(className="fas fa-clipboard-check me-3"),
                                    "Conclusiones del Análisis",
                                ],
                                className="text-primary mb-3",
                            ),
                            html.P(
                                "Resumen de los hallazgos más importantes para la toma de decisiones",
                                className="lead text-muted mb-4",
                            ),
                            html.Hr(),
                        ]
                    )
                ]
            ),
            # Hallazgos Principales (Tarjetas Visuales)
            html.H3(
                [html.I(className="fas fa-key me-2"), "Hallazgos Clave"],
                className="text-primary mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    html.I(
                                                        className="fas fa-clock fa-3x text-danger mb-3"
                                                    ),
                                                    html.H5(
                                                        "El Tiempo es Crítico",
                                                        className="card-title",
                                                    ),
                                                    html.P(
                                                        [
                                                            "La variable ",
                                                            html.Strong(
                                                                "días de entrega"
                                                            ),
                                                            " y la ",
                                                            html.Strong(
                                                                "diferencia estimado vs real"
                                                            ),
                                                            " son los factores #1 de insatisfacción.",
                                                        ],
                                                        className="card-text small",
                                                    ),
                                                ],
                                                className="text-center",
                                            )
                                        ]
                                    )
                                ],
                                className="h-100 shadow-sm border-danger",
                            )
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    html.I(
                                                        className="fas fa-users fa-3x text-info mb-3"
                                                    ),
                                                    html.H5(
                                                        "4 Perfiles Claros",
                                                        className="card-title",
                                                    ),
                                                    html.P(
                                                        [
                                                            "Identificamos 4 segmentos distintos. No todos los clientes ",
                                                            "buscan lo mismo; la estrategia debe ser diferenciada.",
                                                        ],
                                                        className="card-text small",
                                                    ),
                                                ],
                                                className="text-center",
                                            )
                                        ]
                                    )
                                ],
                                className="h-100 shadow-sm border-info",
                            )
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    html.I(
                                                        className="fas fa-magic fa-3x text-success mb-3"
                                                    ),
                                                    html.H5(
                                                        "Predicción Precisa",
                                                        className="card-title",
                                                    ),
                                                    html.P(
                                                        [
                                                            "Podemos predecir la satisfacción con un ",
                                                            html.Strong(
                                                                "75% de precisión"
                                                            ),
                                                            " antes de que el cliente reciba el producto.",
                                                        ],
                                                        className="card-text small",
                                                    ),
                                                ],
                                                className="text-center",
                                            )
                                        ]
                                    )
                                ],
                                className="h-100 shadow-sm border-success",
                            )
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    html.I(
                                                        className="fas fa-hand-holding-usd fa-3x text-warning mb-3"
                                                    ),
                                                    html.H5(
                                                        "Precio vs Servicio",
                                                        className="card-title",
                                                    ),
                                                    html.P(
                                                        [
                                                            "Un precio bajo ",
                                                            html.Strong("NO compensa"),
                                                            " una mala entrega. La logística supera al precio en importancia.",
                                                        ],
                                                        className="card-text small",
                                                    ),
                                                ],
                                                className="text-center",
                                            )
                                        ]
                                    )
                                ],
                                className="h-100 shadow-sm border-warning",
                            )
                        ],
                        md=3,
                    ),
                ],
                className="mb-5",
            ),
            # Insights Detallados (Formato Interactivo/Visual)
            html.H3(
                [html.I(className="fas fa-lightbulb me-2"), "Insights Estratégicos"],
                className="text-primary mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-rocket me-2"),
                                            "Insight #1: La Regla de Oro",
                                        ],
                                        className="bg-dark text-white",
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.H4(
                                                "Sobrecumplir Promesas",
                                                className="text-success mb-3",
                                            ),
                                            html.P(
                                                [
                                                    "Cuando el pedido llega antes de lo estimado, la satisfacción ",
                                                    html.Strong(
                                                        "aumenta dramáticamente"
                                                    ),
                                                    ". A veces es mejor prometer 5 días y entregar en 3, que prometer 2 y entregar en 3.",
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                className="h-100 shadow",
                            )
                        ],
                        md=6,
                        className="mb-4",
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(
                                                className="fas fa-balance-scale me-2"
                                            ),
                                            "Insight #2: Balance Precio-Calidad",
                                        ],
                                        className="bg-dark text-white",
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.H4(
                                                "El Precio no es Excusa",
                                                className="text-warning mb-3",
                                            ),
                                            html.P(
                                                [
                                                    "Los productos caros tienen más tolerancia a demoras leves, pero ",
                                                    "los productos baratos ",
                                                    html.Strong(
                                                        "se castigan severamente"
                                                    ),
                                                    " si fallan en la entrega. El cliente de bajo costo es menos fiel.",
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                className="h-100 shadow",
                            )
                        ],
                        md=6,
                        className="mb-4",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-bullseye me-2"),
                                            "Insight #3: Acción Proactiva",
                                        ],
                                        className="bg-dark text-white",
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.H4(
                                                "Intervenir Antes, No Después",
                                                className="text-info mb-3",
                                            ),
                                            html.P(
                                                [
                                                    "Con el modelo predictivo, podemos identificar clientes en riesgo ",
                                                    html.Strong(
                                                        "en el momento de la compra"
                                                    ),
                                                    ". No esperemos a la queja; actuemos preventivamente.",
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                className="h-100 shadow",
                            )
                        ],
                        md=12,
                        className="mb-4",
                    )
                ]
            ),
            # Resumen Final
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Alert(
                                [
                                    html.H4(
                                        [
                                            html.I(
                                                className="fas fa-flag-checkered me-2"
                                            ),
                                            "Conclusión Final",
                                        ],
                                        className="alert-heading",
                                    ),
                                    html.Hr(),
                                    html.P(
                                        [
                                            "Olist tiene un modelo de negocio sólido, pero su talón de aquiles es la ",
                                            html.Strong("variabilidad logística"),
                                            ". La solución no es solo 'entregar más rápido', sino ",
                                            html.Strong(
                                                "gestionar mejor las expectativas"
                                            ),
                                            ". Implementando modelos predictivos y ajustando los tiempos prometidos, ",
                                            "podemos incrementar la satisfacción promedio de 4.1 a 4.5 en menos de un año.",
                                        ],
                                        className="mb-0 lead",
                                    ),
                                ],
                                color="success",
                                className="shadow-lg",
                            )
                        ]
                    )
                ]
            ),
        ],
        fluid=True,
        className="py-4",
    )
