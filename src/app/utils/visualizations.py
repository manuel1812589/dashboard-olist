"""
Utilidades para visualizaciones consistentes
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paleta de colores del tema oscuro
COLORS = {
    "primary": "#3498db",  # Azul
    "secondary": "#2ecc71",  # Verde
    "warning": "#f39c12",  # Naranja
    "danger": "#e74c3c",  # Rojo
    "info": "#1abc9c",  # Turquesa
    "dark": "#2c3e50",  # Azul oscuro
    "light": "#ecf0f1",  # Gris claro
    "background": "#1e1e1e",  # Fondo oscuro
    "text": "#ecf0f1",  # Texto claro
}

# Configuración de layout oscuro mejorado
DARK_LAYOUT = {
    "plot_bgcolor": "#2c3e50",
    "paper_bgcolor": "#1a1a1a",
    "font": {"color": COLORS["text"], "family": "Arial, sans-serif", "size": 12},
    "xaxis": {
        "gridcolor": "#34495e",
        "linecolor": "#7f8c8d",
        "zerolinecolor": "#7f8c8d",
    },
    "yaxis": {
        "gridcolor": "#34495e",
        "linecolor": "#7f8c8d",
        "zerolinecolor": "#7f8c8d",
    },
    "margin": {"l": 60, "r": 40, "t": 80, "b": 60},
}


def aplicar_tema_oscuro(fig):
    """Aplica tema oscuro consistente y mejorado a una figura de Plotly"""
    fig.update_layout(
        plot_bgcolor=DARK_LAYOUT["plot_bgcolor"],
        paper_bgcolor=DARK_LAYOUT["paper_bgcolor"],
        font=DARK_LAYOUT["font"],
        xaxis=dict(
            gridcolor=DARK_LAYOUT["xaxis"]["gridcolor"],
            linecolor=DARK_LAYOUT["xaxis"]["linecolor"],
            zerolinecolor=DARK_LAYOUT["xaxis"]["zerolinecolor"],
            showgrid=True,
            gridwidth=1,
        ),
        yaxis=dict(
            gridcolor=DARK_LAYOUT["yaxis"]["gridcolor"],
            linecolor=DARK_LAYOUT["yaxis"]["linecolor"],
            zerolinecolor=DARK_LAYOUT["yaxis"]["zerolinecolor"],
            showgrid=True,
            gridwidth=1,
        ),
        margin=DARK_LAYOUT["margin"],
        hovermode="closest",
        hoverlabel=dict(bgcolor="#2c3e50", font_size=13, font_family="Arial"),
    )
    return fig


def crear_histograma(data, column, title, nbins=50, color=COLORS["primary"]):
    """Crea un histograma mejorado con tema oscuro y gradientes"""
    fig = px.histogram(
        data, x=column, nbins=nbins, title=title, color_discrete_sequence=[color]
    )

    # Añadir gradiente y sombra
    fig.update_traces(
        marker=dict(line=dict(color="#1a1a1a", width=1), opacity=0.85),
        hovertemplate="<b>Rango:</b> %{x}<br><b>Frecuencia:</b> %{y}<extra></extra>",
    )

    fig.update_layout(
        title=dict(font=dict(size=16, color=COLORS["text"]), x=0.5, xanchor="center"),
        bargap=0.05,
    )

    fig = aplicar_tema_oscuro(fig)
    return fig


def crear_boxplot(data, x, y, title, color=None):
    """Crea un boxplot con tema oscuro"""
    if color:
        fig = px.box(data, x=x, y=y, color=color, title=title)
    else:
        fig = px.box(
            data, x=x, y=y, title=title, color_discrete_sequence=[COLORS["primary"]]
        )
    fig = aplicar_tema_oscuro(fig)
    return fig


def crear_scatter(data, x, y, color=None, size=None, title="", hover_data=None):
    """Crea un scatter plot con tema oscuro"""
    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=color,
        size=size,
        title=title,
        hover_data=hover_data,
        opacity=0.7,
    )
    fig = aplicar_tema_oscuro(fig)
    return fig


def crear_barras(x, y, title, orientation="v", color=COLORS["primary"]):
    """Crea un gráfico de barras con tema oscuro"""
    if orientation == "h":
        fig = px.bar(
            x=y, y=x, orientation="h", title=title, color_discrete_sequence=[color]
        )
    else:
        fig = px.bar(x=x, y=y, title=title, color_discrete_sequence=[color])
    fig = aplicar_tema_oscuro(fig)
    return fig


def crear_matriz_correlacion(corr_matrix, title="Matriz de Correlación"):
    """Crea una matriz de correlación con tema oscuro (limitada a 5 variables)"""
    # Limitar a 5 variables si hay más
    if corr_matrix.shape[0] > 5:
        # Tomar las 5 primeras variables
        corr_matrix = corr_matrix.iloc[:5, :5]

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        title=title,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig = aplicar_tema_oscuro(fig)
    return fig


def crear_gauge(value, title, min_val=1, max_val=5):
    """Crea un gauge (medidor) para mostrar predicciones"""
    # Determinar color según valor
    if value >= 4:
        color = COLORS["secondary"]
        nivel = "Excelente"
    elif value >= 3:
        color = COLORS["warning"]
        nivel = "Aceptable"
    else:
        color = COLORS["danger"]
        nivel = "Crítico"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": f"Predicción: {nivel}", "font": {"color": COLORS["text"]}},
            number={
                "suffix": f"/{max_val}",
                "font": {"size": 50, "color": COLORS["text"]},
            },
            gauge={
                "axis": {
                    "range": [min_val, max_val],
                    "tickwidth": 1,
                    "tickcolor": COLORS["text"],
                },
                "bar": {"color": color, "thickness": 0.5},
                "steps": [
                    {"range": [1, 2.5], "color": "#5a2a2a"},
                    {"range": [2.5, 3.5], "color": "#5a4a2a"},
                    {"range": [3.5, 5], "color": "#2a5a2a"},
                ],
                "threshold": {
                    "line": {"color": COLORS["text"], "width": 4},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=DARK_LAYOUT["paper_bgcolor"],
        font={"color": COLORS["text"]},
        height=400,
    )
    return fig


def crear_radar_chart(data_dict, categories, title="Comparación"):
    """Crea un radar chart para comparar segmentos"""
    fig = go.Figure()

    colors_list = [
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["warning"],
        COLORS["danger"],
    ]

    for i, (name, values) in enumerate(data_dict.items()):
        # Cerrar el polígono
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                name=name,
                line_color=colors_list[i % len(colors_list)],
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#34495e"),
            bgcolor=DARK_LAYOUT["plot_bgcolor"],
        ),
        title=title,
        paper_bgcolor=DARK_LAYOUT["paper_bgcolor"],
        font={"color": COLORS["text"]},
        height=500,
    )
    return fig


def crear_tabla_comparativa(data, title="Comparación de Modelos"):
    """Crea una tabla comparativa de modelos"""
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(data.columns),
                    fill_color=COLORS["dark"],
                    align="left",
                    font=dict(color=COLORS["text"], size=12),
                ),
                cells=dict(
                    values=[data[col] for col in data.columns],
                    fill_color=DARK_LAYOUT["plot_bgcolor"],
                    align="left",
                    font=dict(color=COLORS["text"], size=11),
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        paper_bgcolor=DARK_LAYOUT["paper_bgcolor"],
        font={"color": COLORS["text"]},
        height=400,
    )
    return fig
