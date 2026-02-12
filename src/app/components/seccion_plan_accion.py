"""
Sección de Plan de Acción - Versión Mejorada
Enfoque: Más gráficos, menos texto, lenguaje simple
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualizations import COLORS


def get_layout():
    """Retorna el layout mejorado de la sección de plan de acción"""

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2(
                                [
                                    html.I(className="fas fa-tasks me-3"),
                                    "Plan de Acción Estratégico",
                                ],
                                className="text-primary mb-3",
                            ),
                            html.P(
                                "Hoja de ruta para transformar los hallazgos en resultados tangibles",
                                className="lead text-muted mb-4",
                            ),
                            html.Hr(),
                        ]
                    )
                ]
            ),
            # Prioridades (Semáforo Visual)
            html.H3(
                [html.I(className="fas fa-traffic-light me-2"), "Prioridades"],
                className="text-primary mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        "ALTA PRIORIDAD",
                                        className="bg-danger text-white text-center fw-bold",
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Gestión de Expectativas",
                                                className="card-title text-center",
                                            ),
                                            html.P(
                                                "Ajustar algoritmos de tiempo estimado para ser más conservadores.",
                                                className="text-muted small text-center",
                                            ),
                                            html.Div(
                                                html.I(
                                                    className="fas fa-arrow-up fa-2x"
                                                ),
                                                className="text-center text-danger",
                                            ),
                                        ]
                                    ),
                                ],
                                className="h-100 shadow-sm",
                            )
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        "MEDIA PRIORIDAD",
                                        className="bg-warning text-dark text-center fw-bold",
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Optimización Logística",
                                                className="card-title text-center",
                                            ),
                                            html.P(
                                                "Trabajar con transportistas para reducir varianza en rutas críticas.",
                                                className="text-muted small text-center",
                                            ),
                                            html.Div(
                                                html.I(className="fas fa-minus fa-2x"),
                                                className="text-center text-warning",
                                            ),
                                        ]
                                    ),
                                ],
                                className="h-100 shadow-sm",
                            )
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        "BAJA PRIORIDAD",
                                        className="bg-info text-white text-center fw-bold",
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Ajustes de Precio",
                                                className="card-title text-center",
                                            ),
                                            html.P(
                                                "El precio actual es competitivo, cambios menores no afectarán satisfacción.",
                                                className="text-muted small text-center",
                                            ),
                                            html.Div(
                                                html.I(
                                                    className="fas fa-arrow-down fa-2x"
                                                ),
                                                className="text-center text-info",
                                            ),
                                        ]
                                    ),
                                ],
                                className="h-100 shadow-sm",
                            )
                        ],
                        md=4,
                    ),
                ],
                className="mb-5",
            ),
            # Acciones por Área (Tabs Verticales o Accordion)
            html.H3(
                [html.I(className="fas fa-sitemap me-2"), "Acciones por Departamento"],
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
                                            html.I(className="fas fa-truck me-2"),
                                            "Logística",
                                        ],
                                        className="bg-dark text-white",
                                    ),
                                    dbc.ListGroup(
                                        [
                                            dbc.ListGroupItem(
                                                [
                                                    html.Strong("Alertas Tempranas: "),
                                                    "Notificar al cliente apenas se detecte un desvío.",
                                                ]
                                            ),
                                            dbc.ListGroupItem(
                                                [
                                                    html.Strong(
                                                        "Margen de Seguridad: "
                                                    ),
                                                    "Añadir 2 días al estimado en zonas rojas.",
                                                ]
                                            ),
                                            dbc.ListGroupItem(
                                                [
                                                    html.Strong("Auditoría: "),
                                                    "Revisar transportistas con >15% de retraso.",
                                                ]
                                            ),
                                        ],
                                        flush=True,
                                    ),
                                ],
                                className="shadow mb-3",
                            )
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-box-open me-2"),
                                            "Producto",
                                        ],
                                        className="bg-dark text-white",
                                    ),
                                    dbc.ListGroup(
                                        [
                                            dbc.ListGroupItem(
                                                [
                                                    html.Strong(
                                                        "Ranking de Vendedores: "
                                                    ),
                                                    "Premiar envío rápido en algoritmo de búsqueda.",
                                                ]
                                            ),
                                            dbc.ListGroupItem(
                                                [
                                                    html.Strong("Fotos Reales: "),
                                                    "Exigir más fotos para reducir devoluciones.",
                                                ]
                                            ),
                                            dbc.ListGroupItem(
                                                [
                                                    html.Strong("Stock: "),
                                                    "Validación de stock en tiempo real.",
                                                ]
                                            ),
                                        ],
                                        flush=True,
                                    ),
                                ],
                                className="shadow mb-3",
                            )
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-headset me-2"),
                                            "Atención al Cliente",
                                        ],
                                        className="bg-dark text-white",
                                    ),
                                    dbc.ListGroup(
                                        [
                                            dbc.ListGroupItem(
                                                [
                                                    html.Strong("Chatbot Proactivo: "),
                                                    "Iniciar chat si el pedido se retrasa.",
                                                ]
                                            ),
                                            dbc.ListGroupItem(
                                                [
                                                    html.Strong(
                                                        "Compensación Automática: "
                                                    ),
                                                    "Cupón de descuento si demora > 3 días.",
                                                ]
                                            ),
                                        ],
                                        flush=True,
                                    ),
                                ],
                                className="shadow mb-3",
                            )
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-laptop-code me-2"),
                                            "Tecnología",
                                        ],
                                        className="bg-dark text-white",
                                    ),
                                    dbc.ListGroup(
                                        [
                                            dbc.ListGroupItem(
                                                [
                                                    html.Strong("Modelo Predictivo: "),
                                                    "Integrar CART en checkout.",
                                                ]
                                            ),
                                            dbc.ListGroupItem(
                                                [
                                                    html.Strong("Dashboard en Vivo: "),
                                                    "Dejar esta app corriendo para monitoreo.",
                                                ]
                                            ),
                                        ],
                                        flush=True,
                                    ),
                                ],
                                className="shadow mb-3",
                            )
                        ],
                        md=6,
                    ),
                ],
                className="mb-5",
            ),
            # Timeline Visual (Timeline simple con tarjetas)
            html.H3(
                [
                    html.I(className="fas fa-calendar-alt me-2"),
                    "Cronograma de Implementación",
                ],
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
                                            html.H4(
                                                "Mes 1-2", className="text-primary"
                                            ),
                                            html.P("Piloto", className="text-muted"),
                                            html.Hr(),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        "Ajuste de tiempos estimados"
                                                    ),
                                                    html.Li(
                                                        "Despliegue de alertas básicas"
                                                    ),
                                                ],
                                                className="list-unstyled",
                                            ),
                                        ]
                                    )
                                ],
                                className="h-100 shadow-sm",
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
                                            html.H4("Mes 3-4", className="text-info"),
                                            html.P("Escalado", className="text-muted"),
                                            html.Hr(),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        "Integración modelo predictivo"
                                                    ),
                                                    html.Li("Chatbot proactivo"),
                                                ],
                                                className="list-unstyled",
                                            ),
                                        ]
                                    )
                                ],
                                className="h-100 shadow-sm",
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
                                            html.H4(
                                                "Mes 5-6", className="text-warning"
                                            ),
                                            html.P(
                                                "Optimización", className="text-muted"
                                            ),
                                            html.Hr(),
                                            html.Ul(
                                                [
                                                    html.Li("Negociación logística"),
                                                    html.Li("Ranking de vendedores"),
                                                ],
                                                className="list-unstyled",
                                            ),
                                        ]
                                    )
                                ],
                                className="h-100 shadow-sm",
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
                                            html.H4("Mes 7+", className="text-success"),
                                            html.P(
                                                "Consolidación", className="text-muted"
                                            ),
                                            html.Hr(),
                                            html.Ul(
                                                [
                                                    html.Li("Monitoreo continuo"),
                                                    html.Li(
                                                        "Expansión a nuevos mercados"
                                                    ),
                                                ],
                                                className="list-unstyled",
                                            ),
                                        ]
                                    )
                                ],
                                className="h-100 shadow-sm",
                            )
                        ],
                        md=3,
                    ),
                ],
                className="mb-5",
            ),
            # Impacto Esperado
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Alert(
                                [
                                    html.H4(
                                        [
                                            html.I(className="fas fa-chart-line me-2"),
                                            "Impacto Esperado (KPIs)",
                                        ],
                                        className="alert-heading",
                                    ),
                                    html.Hr(),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.H2(
                                                        "4.5",
                                                        className="text-success mb-0",
                                                    ),
                                                    html.P(
                                                        "Satisfacción Promedio",
                                                        className="mb-0",
                                                    ),
                                                ],
                                                className="text-center",
                                            ),
                                            dbc.Col(
                                                [
                                                    html.H2(
                                                        "-15%",
                                                        className="text-success mb-0",
                                                    ),
                                                    html.P(
                                                        "Reclamos", className="mb-0"
                                                    ),
                                                ],
                                                className="text-center",
                                            ),
                                            dbc.Col(
                                                [
                                                    html.H2(
                                                        "+10%",
                                                        className="text-success mb-0",
                                                    ),
                                                    html.P(
                                                        "Retención", className="mb-0"
                                                    ),
                                                ],
                                                className="text-center",
                                            ),
                                        ]
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
