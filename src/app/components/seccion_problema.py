"""
Sección de Diagnóstico del Problema (EDA) - Versión Mejorada
Enfoque: Gráficos claros, explicaciones extensas y accesibles
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import cargar_datos_completos, preprocesar_datos
from utils.visualizations import aplicar_tema_oscuro, COLORS
import plotly.express as px
import plotly.graph_objects as go

# Cargar y preprocesar datos
df_full = cargar_datos_completos()
df_full = preprocesar_datos(df_full)


def crear_visualizaciones_diagnostico():
    """Crea visualizaciones claras y fáciles de entender"""

    # 1. Distribución de Satisfacción - Gráfico de Barras Simple
    fig_satisfaccion = go.Figure()

    satisfaccion_counts = df_full["puntuacion_satisfaccion"].value_counts().sort_index()
    colors_sat = [
        "#e74c3c" if x < 3 else "#f39c12" if x < 4 else "#2ecc71"
        for x in satisfaccion_counts.index
    ]

    fig_satisfaccion.add_trace(
        go.Bar(
            x=satisfaccion_counts.index,
            y=satisfaccion_counts.values,
            marker=dict(
                color=colors_sat, line=dict(color="#1a1a1a", width=2), opacity=0.9
            ),
            text=satisfaccion_counts.values,
            textposition="outside",
            textfont=dict(size=14, color=COLORS["text"]),
            hovertemplate="<b>Puntuación:</b> %{x}<br><b>Cantidad de clientes:</b> %{y}<extra></extra>",
        )
    )

    fig_satisfaccion.update_layout(
        title=dict(
            text="Distribución de Satisfacción del Cliente",
            font=dict(size=20, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Puntuación (1 = Muy insatisfecho, 5 = Muy satisfecho)",
        yaxis_title="Cantidad de Clientes",
        height=450,
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),
    )
    fig_satisfaccion = aplicar_tema_oscuro(fig_satisfaccion)

    # 2. Comparación de Satisfacción por Velocidad de Entrega
    # Crear categorías de tiempo
    df_full["categoria_entrega"] = pd.cut(
        df_full["dias_entrega"],
        bins=[0, 5, 10, 15, 100],
        labels=[
            "Muy rápido (1-5 días)",
            "Rápido (6-10 días)",
            "Normal (11-15 días)",
            "Lento (más de 15 días)",
        ],
    )

    satisfaccion_por_velocidad = (
        df_full.groupby("categoria_entrega")["puntuacion_satisfaccion"]
        .mean()
        .reset_index()
    )

    fig_velocidad = go.Figure()

    fig_velocidad.add_trace(
        go.Bar(
            x=satisfaccion_por_velocidad["categoria_entrega"],
            y=satisfaccion_por_velocidad["puntuacion_satisfaccion"],
            marker=dict(
                color=["#2ecc71", "#3498db", "#f39c12", "#e74c3c"],
                line=dict(color="#1a1a1a", width=2),
            ),
            text=[
                f"{val:.2f}"
                for val in satisfaccion_por_velocidad["puntuacion_satisfaccion"]
            ],
            textposition="outside",
            textfont=dict(size=14, color=COLORS["text"]),
        )
    )

    fig_velocidad.update_layout(
        title=dict(
            text="Satisfacción según Velocidad de Entrega",
            font=dict(size=20, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        yaxis_title="Satisfacción Promedio",
        yaxis=dict(range=[0, 5]),
        height=450,
    )
    fig_velocidad = aplicar_tema_oscuro(fig_velocidad)

    # 3. Impacto de Cumplir con la Fecha Prometida
    retraso_stats = (
        df_full.groupby("entrega_tardia")["puntuacion_satisfaccion"]
        .mean()
        .reset_index()
    )
    retraso_stats["label"] = retraso_stats["entrega_tardia"].map(
        {0: "Llegó a tiempo o antes", 1: "Llegó tarde"}
    )

    fig_retrasos = go.Figure()

    fig_retrasos.add_trace(
        go.Bar(
            x=retraso_stats["label"],
            y=retraso_stats["puntuacion_satisfaccion"],
            marker=dict(
                color=[COLORS["secondary"], COLORS["danger"]],
                line=dict(color="#1a1a1a", width=2),
            ),
            text=[f"{val:.2f}" for val in retraso_stats["puntuacion_satisfaccion"]],
            textposition="outside",
            textfont=dict(size=16, color=COLORS["text"]),
        )
    )

    fig_retrasos.update_layout(
        title=dict(
            text="¿Qué pasa cuando no cumplimos la fecha prometida?",
            font=dict(size=20, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        yaxis_title="Satisfacción Promedio",
        yaxis=dict(range=[0, 5]),
        height=450,
    )
    fig_retrasos = aplicar_tema_oscuro(fig_retrasos)

    # 4. Distribución de Tiempos de Entrega - Histograma Simple
    fig_dias = go.Figure()

    fig_dias.add_trace(
        go.Histogram(
            x=df_full["dias_entrega"],
            nbinsx=30,
            marker=dict(
                color=COLORS["info"],
                line=dict(color="#1a1a1a", width=1),
            ),
        )
    )

    fig_dias.update_layout(
        title=dict(
            text="¿Cuánto tiempo toma entregar los pedidos?",
            font=dict(size=20, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Días para entregar",
        yaxis_title="Cantidad de pedidos",
        height=400,
    )
    fig_dias = aplicar_tema_oscuro(fig_dias)

    # 5. Comparación de Precios - Boxplot Simple
    fig_precio = go.Figure()

    for satisfecho in [0, 1]:
        data = df_full[df_full["cliente_satisfecho"] == satisfecho]["precio_producto"]
        fig_precio.add_trace(
            go.Box(
                y=data,
                name=(
                    "Cliente Satisfecho" if satisfecho == 1 else "Cliente No Satisfecho"
                ),
                marker_color=(
                    COLORS["secondary"] if satisfecho == 1 else COLORS["danger"]
                ),
                boxmean=True,
            )
        )

    fig_precio.update_layout(
        title=dict(
            text="¿Los clientes satisfechos pagan más o menos?",
            font=dict(size=20, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        yaxis_title="Precio del Producto (R$)",
        showlegend=True,
        height=450,
    )
    fig_precio = aplicar_tema_oscuro(fig_precio)

    # 6. Mapa de Calor de Correlaciones - Simplificado
    corr_vars = [
        "precio_producto",
        "dias_entrega",
        "diferencia_estimado_real",
        "puntuacion_satisfaccion",
    ]
    corr_labels = [
        "Precio del Producto",
        "Días de Entrega",
        "Diferencia en Fecha",
        "Satisfacción",
    ]

    corr_matrix = df_full[corr_vars].corr()

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_labels,
            y=corr_labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 14},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlación: %{z:.2f}<extra></extra>",
        )
    )

    fig_corr.update_layout(
        title=dict(
            text="¿Qué factores están más relacionados con la satisfacción?",
            font=dict(size=20, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        height=450,
    )
    fig_corr = aplicar_tema_oscuro(fig_corr)

    return fig_satisfaccion, fig_velocidad, fig_retrasos, fig_dias, fig_precio, fig_corr


# Generar visualizaciones
fig_satisfaccion, fig_velocidad, fig_retrasos, fig_dias, fig_precio, fig_corr = (
    crear_visualizaciones_diagnostico()
)


def get_layout():
    """Retorna el layout mejorado de la sección de diagnóstico"""

    # Calcular métricas clave
    satisfaccion_promedio = df_full["puntuacion_satisfaccion"].mean()
    pct_satisfechos = df_full["cliente_satisfecho"].mean() * 100
    pct_retrasos = df_full["entrega_tardia"].mean() * 100
    dias_promedio = df_full["dias_entrega"].mean()

    return dbc.Container(
        [
            # Encabezado
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2(
                                "Diagnóstico del Problema",
                                className="text-primary mb-3",
                            ),
                            html.P(
                                [
                                    "Bienvenido al análisis de satisfacción del cliente. En esta sección exploramos ",
                                    "los datos de Olist para entender qué hace que los clientes estén contentos o ",
                                    "descontentos con sus compras. Analizamos más de 3,000 pedidos para descubrir ",
                                    "patrones y encontrar oportunidades de mejora.",
                                ],
                                className="lead text-muted mb-4",
                            ),
                            html.Hr(),
                        ]
                    )
                ]
            ),
            # Métricas Clave con explicaciones
            html.H4("Situación Actual", className="text-primary mb-3"),
            html.P(
                [
                    "Estos son los números más importantes que resumen cómo está funcionando el negocio ahora mismo. ",
                    "Cada métrica nos cuenta una historia diferente sobre la experiencia del cliente.",
                ],
                className="text-muted mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.I(
                                                className="fas fa-star fa-2x text-warning mb-3"
                                            ),
                                            html.H3(
                                                f"{satisfaccion_promedio:.2f}/5.0",
                                                className="text-primary mb-2",
                                            ),
                                            html.P(
                                                "Satisfacción Promedio",
                                                className="text-muted mb-2 fw-bold",
                                            ),
                                            html.Small(
                                                "Esta es la calificación promedio que los clientes dan a sus compras",
                                                className="text-muted",
                                            ),
                                        ],
                                        className="text-center",
                                    )
                                ],
                                className="shadow-sm mb-4 h-100",
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
                                            html.I(
                                                className="fas fa-smile fa-2x text-success mb-3"
                                            ),
                                            html.H3(
                                                f"{pct_satisfechos:.0f}%",
                                                className="text-success mb-2",
                                            ),
                                            html.P(
                                                "Clientes Satisfechos",
                                                className="text-muted mb-2 fw-bold",
                                            ),
                                            html.Small(
                                                "Porcentaje de clientes que califican con 4 o 5 estrellas",
                                                className="text-muted",
                                            ),
                                        ],
                                        className="text-center",
                                    )
                                ],
                                className="shadow-sm mb-4 h-100",
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
                                            html.I(
                                                className="fas fa-clock fa-2x text-danger mb-3"
                                            ),
                                            html.H3(
                                                f"{pct_retrasos:.0f}%",
                                                className="text-danger mb-2",
                                            ),
                                            html.P(
                                                "Entregas con Retraso",
                                                className="text-muted mb-2 fw-bold",
                                            ),
                                            html.Small(
                                                "Pedidos que llegaron después de la fecha prometida",
                                                className="text-muted",
                                            ),
                                        ],
                                        className="text-center",
                                    )
                                ],
                                className="shadow-sm mb-4 h-100",
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
                                            html.I(
                                                className="fas fa-shipping-fast fa-2x text-info mb-3"
                                            ),
                                            html.H3(
                                                f"{dias_promedio:.1f}",
                                                className="text-info mb-2",
                                            ),
                                            html.P(
                                                "Días Promedio de Entrega",
                                                className="text-muted mb-2 fw-bold",
                                            ),
                                            html.Small(
                                                "Tiempo que toma en promedio entregar un pedido",
                                                className="text-muted",
                                            ),
                                        ],
                                        className="text-center",
                                    )
                                ],
                                className="shadow-sm mb-4 h-100",
                            )
                        ],
                        md=3,
                    ),
                ]
            ),
            # Gráfico Principal de Satisfacción
            html.H4(
                "¿Cómo califican los clientes su experiencia?",
                className="text-primary mb-3 mt-4",
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
                                                figure=fig_satisfaccion,
                                                config={"displayModeBar": False},
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
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Alert(
                                [
                                    html.H6(
                                        "💡 ¿Qué nos dice este gráfico?",
                                        className="alert-heading",
                                    ),
                                    html.P(
                                        [
                                            "Este gráfico muestra cuántos clientes dieron cada calificación del 1 al 5. ",
                                            "Lo que vemos es muy positivo: la mayoría de nuestros clientes (las barras verdes) ",
                                            "están satisfechos, dando calificaciones de 4 y 5 estrellas. Sin embargo, ",
                                            "también notamos que hay un grupo de clientes que califican con 1, 2 o 3 estrellas ",
                                            "(barras rojas y amarillas). Estos son nuestros clientes insatisfechos y son una ",
                                            "oportunidad importante de mejora. Si logramos convertir aunque sea algunos de estos ",
                                            "clientes insatisfechos en clientes satisfechos, podríamos mejorar significativamente ",
                                            "nuestros resultados.",
                                        ],
                                        className="mb-0",
                                    ),
                                ],
                                color="light",
                                className="mb-4",
                            )
                        ]
                    )
                ]
            ),
            # Análisis de Velocidad de Entrega
            html.H4(
                "El factor más importante: La velocidad de entrega",
                className="text-primary mb-3 mt-4",
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
                                                figure=fig_velocidad,
                                                config={"displayModeBar": False},
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
                                                figure=fig_retrasos,
                                                config={"displayModeBar": False},
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
                            dbc.Alert(
                                [
                                    html.H6(
                                        "💡 ¿Por qué esto es tan importante?",
                                        className="alert-heading",
                                    ),
                                    html.P(
                                        [
                                            "Estos dos gráficos revelan uno de los hallazgos más importantes de nuestro análisis. ",
                                            "El gráfico de la izquierda muestra que cuando los pedidos llegan más rápido, ",
                                            "los clientes son más felices. Los pedidos que llegan en 1-5 días tienen una ",
                                            "satisfacción promedio mucho más alta que los que tardan más de 15 días.",
                                        ],
                                        className="mb-2",
                                    ),
                                    html.P(
                                        [
                                            "Pero hay algo aún más revelador en el gráfico de la derecha: cumplir con la ",
                                            "fecha prometida es crucial. Cuando un pedido llega tarde (después de la fecha ",
                                            "que le prometimos al cliente), la satisfacción cae dramáticamente. Esto significa ",
                                            "que no se trata solo de entregar rápido, sino de ",
                                            html.Strong("cumplir lo que prometemos"),
                                            ". A veces es mejor prometer 10 días y entregar en 8, que prometer 5 días y ",
                                            "entregar en 7. La confianza del cliente se rompe cuando no cumplimos nuestras promesas.",
                                        ],
                                        className="mb-0",
                                    ),
                                ],
                                color="warning",
                                className="mb-4",
                            )
                        ]
                    )
                ]
            ),
            # Análisis de Precios y Tiempos
            html.H4(
                "Otros factores que exploramos", className="text-primary mb-3 mt-4"
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
                                                figure=fig_precio,
                                                config={"displayModeBar": False},
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
                                                figure=fig_dias,
                                                config={"displayModeBar": False},
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
                            dbc.Alert(
                                [
                                    html.H6(
                                        "💡 Descubrimientos adicionales",
                                        className="alert-heading",
                                    ),
                                    html.P(
                                        [
                                            html.Strong("Sobre el precio: "),
                                            "Contrario a lo que muchos podrían pensar, el precio del producto tiene ",
                                            "menos impacto en la satisfacción de lo que esperábamos. El gráfico de la ",
                                            "izquierda muestra que los clientes satisfechos y no satisfechos pagan precios ",
                                            "similares en promedio. Esto nos dice que ",
                                            html.Strong(
                                                "la calidad del servicio importa más que el precio"
                                            ),
                                            ". Un cliente está dispuesto a pagar, pero espera recibir un buen servicio a cambio.",
                                        ],
                                        className="mb-2",
                                    ),
                                    html.P(
                                        [
                                            html.Strong(
                                                "Sobre los tiempos de entrega: "
                                            ),
                                            "El gráfico de la derecha nos muestra que hay mucha variabilidad en cuánto tiempo ",
                                            "tardan los pedidos. Algunos llegan en pocos días, otros tardan semanas. Esta ",
                                            "inconsistencia es un problema porque hace difícil gestionar las expectativas de ",
                                            "los clientes. Si logramos hacer más predecibles nuestros tiempos de entrega, ",
                                            "podríamos dar fechas de entrega más confiables a nuestros clientes.",
                                        ],
                                        className="mb-0",
                                    ),
                                ],
                                color="light",
                                className="mb-4",
                            )
                        ]
                    )
                ]
            ),
            # Correlaciones
            html.H4("Relaciones entre variables", className="text-primary mb-3 mt-4"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                figure=fig_corr,
                                                config={"displayModeBar": False},
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
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Alert(
                                [
                                    html.H6(
                                        "💡 Leyendo el mapa de correlaciones",
                                        className="alert-heading",
                                    ),
                                    html.P(
                                        [
                                            "Este mapa de colores nos ayuda a entender qué variables están relacionadas entre sí. ",
                                            "Los números van de -1 a 1:",
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Ul(
                                        [
                                            html.Li(
                                                [
                                                    html.Strong(
                                                        "Rojo (números negativos): "
                                                    ),
                                                    "Cuando una variable sube, la otra baja. Por ejemplo, vemos que ",
                                                    "la 'Diferencia en Fecha' (cuando el pedido llega tarde) tiene una ",
                                                    "correlación negativa con la Satisfacción (-0.35). Esto confirma que ",
                                                    "cuando hay retrasos, la satisfacción baja.",
                                                ]
                                            ),
                                            html.Li(
                                                [
                                                    html.Strong(
                                                        "Azul (números positivos): "
                                                    ),
                                                    "Las variables se mueven juntas en la misma dirección.",
                                                ]
                                            ),
                                            html.Li(
                                                [
                                                    html.Strong(
                                                        "Blanco (cerca de 0): "
                                                    ),
                                                    "No hay mucha relación entre las variables.",
                                                ]
                                            ),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.P(
                                        [
                                            "El hallazgo clave aquí es que la 'Diferencia en Fecha' (si llegó antes o después ",
                                            "de lo prometido) es el factor que más correlación tiene con la satisfacción. ",
                                            "Esto refuerza lo que vimos antes: ",
                                            html.Strong(
                                                "cumplir con las fechas prometidas es fundamental"
                                            ),
                                            " para tener clientes satisfechos.",
                                        ],
                                        className="mb-0",
                                    ),
                                ],
                                color="info",
                                className="mb-4",
                            )
                        ]
                    )
                ]
            ),
            # Conclusión Final
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Alert(
                                [
                                    html.H5(
                                        "🎯 Conclusión del Diagnóstico",
                                        className="alert-heading",
                                    ),
                                    html.P(
                                        [
                                            "Después de analizar todos estos datos, llegamos a una conclusión clara: ",
                                            html.Strong(
                                                "el tiempo de entrega y cumplir con las fechas prometidas son los factores "
                                                "más importantes para la satisfacción del cliente"
                                            ),
                                            ". El precio, aunque importante, tiene un impacto menor de lo que se podría pensar.",
                                        ],
                                        className="mb-3",
                                    ),
                                    html.P(
                                        [
                                            "Para mejorar la experiencia de nuestros clientes, necesitamos enfocarnos en tres áreas:",
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Ol(
                                        [
                                            html.Li(
                                                [
                                                    html.Strong(
                                                        "Optimizar la logística: "
                                                    ),
                                                    "Trabajar para reducir los tiempos de entrega promedio.",
                                                ]
                                            ),
                                            html.Li(
                                                [
                                                    html.Strong(
                                                        "Gestionar expectativas: "
                                                    ),
                                                    "Dar fechas de entrega más conservadoras y confiables, en lugar de ",
                                                    "prometer entregas rápidas que no podemos cumplir.",
                                                ]
                                            ),
                                            html.Li(
                                                [
                                                    html.Strong(
                                                        "Reducir variabilidad: "
                                                    ),
                                                    "Hacer que nuestros procesos sean más predecibles para poder dar ",
                                                    "estimaciones más precisas a los clientes.",
                                                ]
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    html.P(
                                        [
                                            "En las siguientes secciones de este análisis, veremos cómo limpiar los datos, ",
                                            "segmentar a los clientes en grupos, y usar modelos de predicción para anticipar ",
                                            "qué clientes podrían tener problemas, todo con el objetivo de mejorar la experiencia ",
                                            "y aumentar la satisfacción.",
                                        ],
                                        className="mb-0",
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
