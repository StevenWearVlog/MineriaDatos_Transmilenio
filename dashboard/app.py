"""
Fase 4: Dashboard Ejecutivo y Técnico Interactivo — TransMilenio S.A. (2026)
Construido con Streamlit + Plotly para despliegue local y en Streamlit Community Cloud.

Consume exclusivamente artefactos livianos preprocesados:
  - data/processed/vista_minable.parquet (2.02 MB)
  - data/processed/reporte_etl.json
  - data/processed/perfil_estaciones.csv y perfil_estaciones_escalado.csv
  - data/output/*.csv y *.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y PALETA CORPORATIVA TRANSMILENIO
# ==============================================================================
st.set_page_config(
    page_title="TransMilenio S.A. | Analítica y Minería de Datos 2026",
    page_icon="🚍",
    layout="wide",
    initial_sidebar_state="expanded",
)

TM_RED = "#D31F26"
TM_DARK = "#1E293B"
TM_SLATE = "#475569"
TM_BLUE = "#1D4ED8"
TM_GREEN = "#059669"
TM_AMBER = "#D97706"
TM_LIGHT_BG = "#F8FAFC"

CLUSTER_COLORS = {
    "C0 - Mega-Portales y Cabeceras de Alta Congestión": "#D31F26",
    "C1 - Estaciones Troncales Fuertes / Nodos Principales": "#1D4ED8",
    "C2 - Estaciones de Demanda Intermedia": "#059669",
    "C3 - Estaciones de Baja Demanda / Periféricas": "#D97706",
}

DIAS_NOMBRES = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo",
}

TIPO_DIA_LABELS = {
    "Habil": "Día Hábil",
    "Sabado": "Sábado",
    "Domingo_Festivo": "Dom / Festivo",
}

FRANJA_LABELS = {
    "Pico_AM": "Pico AM (05–08h)",
    "Pico_PM": "Pico PM (16–19h)",
    "Valle": "Valle",
}

TIPOS_DIA_ALL = ["Habil", "Sabado", "Domingo_Festivo"]
FRANJAS_ALL = ["Pico_AM", "Pico_PM", "Valle"]

ESCENARIO_UI_MAP = {
    "Base (Moderado · L2)": (
        "Escenario_Base_Nov2026",
        "Pred_Base",
        "Total_Nov_Base",
        "Prom_Diario_Nov_Base",
    ),
    "Optimista (Q90 + Control Evasión)": (
        "Escenario_Optimista_Nov2026",
        "Pred_Optimista",
        "Total_Nov_Optimista",
        "Prom_Diario_Nov_Optimista",
    ),
    "Pesimista (Q10 + Obras/Choques)": (
        "Escenario_Pesimista_Nov2026",
        "Pred_Pesimista",
        "Total_Nov_Pesimista",
        "Prom_Diario_Nov_Pesimista",
    ),
}

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "output"

# ==============================================================================
# ESTILOS CSS SAAS EJECUTIVO (PESTAÑAS PILL, TARJETAS, OCULTAR DEPLOY)
# ==============================================================================
st.markdown(
    f"""
    <style>
        /* Reducir padding superior y ocultar botón Deploy nativo para presentación limpia */
        .block-container {{
            padding-top: 0.75rem !important;
            padding-bottom: 2rem !important;
            max-width: 1480px;
        }}
        [data-testid="stAppDeployButton"],
        .stAppDeployButton {{
            display: none !important;
        }}

        /* Contenedores nativos st.container(border=True) estilo tarjeta SaaS con sombra suave */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.035);
        }}

        /* Pestañas principales estilo Pill moderno con estado activo en Rojo TransMilenio */
        div[data-testid="stTabs"] > div[role="tablist"] {{
            gap: 0.45rem;
            background: #E2E8F0;
            padding: 0.38rem;
            border-radius: 12px;
            border: 1px solid #CBD5E1;
            margin-bottom: 0.65rem;
        }}
        div[data-testid="stTabs"] button[role="tab"] {{
            border-radius: 9px !important;
            padding: 0.48rem 1.05rem !important;
            font-weight: 600 !important;
            color: #334155 !important;
            background: transparent !important;
            border: none !important;
            transition: all 0.16s ease-in-out;
        }}
        div[data-testid="stTabs"] button[role="tab"]:hover {{
            background: rgba(255, 255, 255, 0.72) !important;
            color: #0F172A !important;
        }}
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
            background: {TM_RED} !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 8px rgba(211, 31, 38, 0.28);
        }}
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {{
            color: #FFFFFF !important;
        }}
        div[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
        div[data-testid="stTabs"] div[data-baseweb="tab-border"] {{
            display: none !important;
        }}

        /* Encabezado Hero compacto */
        .tm-hero {{
            background: linear-gradient(135deg, {TM_DARK} 0%, #0F172A 74%, {TM_RED} 100%);
            color: #FFFFFF;
            padding: 1.05rem 1.45rem;
            border-radius: 13px;
            border-left: 6px solid {TM_RED};
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.10);
            margin-bottom: 0.75rem;
        }}
        .tm-hero h1 {{
            color: #FFFFFF !important;
            font-size: 1.45rem;
            font-weight: 800;
            margin: 0 0 0.2rem 0;
            letter-spacing: -0.02em;
        }}
        .tm-hero p {{
            color: #E2E8F0;
            font-size: 0.88rem;
            margin: 0;
        }}

        /* Barra compacta de estado activo debajo de pestañas */
        .status-strip {{
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-left: 4px solid {TM_DARK};
            border-radius: 9px;
            padding: 0.48rem 0.9rem;
            font-size: 0.82rem;
            color: #334155;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
        }}
        .status-pill {{
            background: #F1F5F9;
            color: #0F172A;
            font-weight: 600;
            padding: 0.14rem 0.52rem;
            border-radius: 6px;
            border: 1px solid #E2E8F0;
        }}
        .status-pill-red {{
            background: #FEF2F2;
            color: {TM_RED};
            font-weight: 700;
            padding: 0.14rem 0.55rem;
            border-radius: 6px;
            border: 1px solid #FECACA;
        }}

        /* Tarjetas de los 5 KPIs Corporativos */
        .kpi-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-top: 4px solid {TM_RED};
            border-radius: 12px;
            padding: 0.9rem 1rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            min-height: 162px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .kpi-top-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.3rem;
            margin-bottom: 0.3rem;
        }}
        .kpi-help {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #F1F5F9;
            color: #475569;
            font-size: 0.7rem;
            font-weight: 700;
            border: 1px solid #CBD5E1;
            cursor: help;
        }}
        .kpi-title {{
            color: {TM_SLATE};
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.2rem;
        }}
        .kpi-value {{
            color: {TM_DARK};
            font-size: 1.42rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.28rem;
        }}
        .kpi-sub {{
            color: #64748B;
            font-size: 0.75rem;
            line-height: 1.32;
        }}
        .kpi-filter-highlight {{
            margin-top: 0.38rem;
            padding: 0.24rem 0.48rem;
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            color: #1E40AF;
            border-radius: 6px;
            font-size: 0.73rem;
            font-weight: 700;
        }}
        .badge-obs {{
            display: inline-block;
            background: #DCFCE7;
            color: #166534;
            font-size: 0.66rem;
            font-weight: 700;
            padding: 0.13rem 0.46rem;
            border-radius: 999px;
        }}
        .badge-param {{
            display: inline-block;
            background: #FEF3C7;
            color: #92400E;
            font-size: 0.66rem;
            font-weight: 700;
            padding: 0.13rem 0.46rem;
            border-radius: 999px;
        }}

        /* Tarjetas de fases y bloques de insight automático */
        .phase-card {{
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-left: 5px solid {TM_RED};
            border-radius: 12px;
            padding: 0.95rem 1.05rem;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
            height: 100%;
        }}
        .phase-step {{
            color: {TM_RED};
            font-weight: 800;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .phase-title {{
            color: {TM_DARK};
            font-weight: 700;
            font-size: 0.98rem;
            margin: 0.2rem 0 0.4rem 0;
        }}
        .insight-bar {{
            background: #F8FAFC;
            border-left: 3px solid {TM_BLUE};
            border-radius: 0 8px 8px 0;
            padding: 0.45rem 0.75rem;
            font-size: 0.82rem;
            color: #334155;
            margin-top: 0.2rem;
        }}
        .insight-alert {{
            background: #FEF2F2;
            border-left: 3px solid {TM_RED};
            border-radius: 0 8px 8px 0;
            padding: 0.48rem 0.78rem;
            font-size: 0.82rem;
            color: #7F1D1D;
            margin-top: 0.35rem;
        }}
        .sidebar-block-tag {{
            font-size: 0.7rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #475569;
            margin-bottom: 0.1rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# CARGA CACHEADA EXCLUSIVA DE ARTEFACTOS LIVIANOS (@st.cache_data)
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_json_artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with (PROCESSED_DIR / "reporte_etl.json").open("r", encoding="utf-8") as f:
        rep_etl = json.load(f)
    with (OUTPUT_DIR / "reporte_modelos.json").open("r", encoding="utf-8") as f:
        rep_mod = json.load(f)
    with (OUTPUT_DIR / "kpis_escenarios.json").open("r", encoding="utf-8") as f:
        kpis_json = json.load(f)
    return rep_etl, rep_mod, kpis_json


@st.cache_data(show_spinner=False)
def load_light_tables() -> dict[str, pd.DataFrame]:
    return {
        "perfil_raw": pd.read_csv(PROCESSED_DIR / "perfil_estaciones.csv"),
        "perfil_scaled": pd.read_csv(PROCESSED_DIR / "perfil_estaciones_escalado.csv"),
        "codo_sil": pd.read_csv(OUTPUT_DIR / "curva_codo_silhouette.csv"),
        "clusters": pd.read_csv(OUTPUT_DIR / "clusters_estaciones.csv"),
        "radar": pd.read_csv(OUTPUT_DIR / "radar_clusters.csv"),
        "backtest": pd.read_csv(OUTPUT_DIR / "backtest_validacion.csv"),
        "feat_imp": pd.read_csv(OUTPUT_DIR / "feature_importance.csv"),
        "proy_diaria": pd.read_csv(OUTPUT_DIR / "proyecciones_3_escenarios_diarias.csv"),
        "proy_estacion": pd.read_csv(OUTPUT_DIR / "proyecciones_3_escenarios_estacion.csv"),
        "kpis_csv": pd.read_csv(OUTPUT_DIR / "kpis_escenarios.csv"),
    }


@st.cache_data(show_spinner=False)
def load_vista_minable_parquet() -> pd.DataFrame:
    cols = [
        "Fecha",
        "Hora_Bloque",
        "ID_Estacion",
        "Nombre_Estacion",
        "Troncal",
        "Dia_Semana",
        "Mes",
        "Tipo_Dia",
        "Franja_Horaria",
        "Total_Validaciones",
        "tipo_esta",
        "area_est",
        "cap_art",
        "cap_biart",
    ]
    return pd.read_parquet(PROCESSED_DIR / "vista_minable.parquet", columns=cols)


# ==============================================================================
# UTILIDADES DE COMPONENTES VISUALES Y PLOTLY SIN SOLAPAMIENTOS
# ==============================================================================
def render_kpi_card(
    titulo: str,
    valor: str,
    subtitulo: str,
    es_observado: bool,
    tooltip_formula: str = "",
    extra_highlight_html: str = "",
) -> str:
    badge_cls = "badge-obs" if es_observado else "badge-param"
    badge_txt = "Observado · Datos Abiertos" if es_observado else "Paramétrico Oficial · Sensibilizado"
    safe_tip = tooltip_formula.replace('"', "&quot;")
    help_badge = f'<span class="kpi-help" title="{safe_tip}">?</span>' if safe_tip else ""
    return f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-top-row">
                <span class="{badge_cls}">{badge_txt}</span>
                {help_badge}
            </div>
            <div class="kpi-title">{titulo}</div>
            <div class="kpi-value">{valor}</div>
            <div class="kpi-sub">{subtitulo}</div>
        </div>
        {extra_highlight_html}
    </div>
    """


def apply_plotly_style(
    fig: go.Figure,
    height: int = 390,
    show_legend: bool = True,
    bottom_margin: int = 72,
    right_margin: int = 26,
) -> go.Figure:
    """
    Aplica estilo tipográfico uniforme y separa limpiamente título (arriba) y leyenda
    horizontal sin título redundante (abajo), eliminando cualquier solapamiento.
    """
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(
            t=65,
            b=bottom_margin if show_legend else 42,
            l=45,
            r=right_margin,
        ),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=TM_DARK),
        title=dict(
            x=0.0,
            xanchor="left",
            y=0.95,
            yanchor="top",
            font=dict(size=14, color=TM_DARK),
        ),
        showlegend=show_legend,
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(size=11, color=TM_SLATE),
        ),
    )
    return fig


def render_kpi_methodology_popover(popover_key: str) -> None:
    """Muestra en un popover limpio las fórmulas y fuentes exactas de los 5 KPIs."""
    with st.popover("ℹ️ Ver fórmulas y fuentes de los 5 KPIs", key=popover_key):
        st.markdown(
            """
            **Trazabilidad metodológica de los 5 KPIs Corporativos:**
            1. **Demanda Diaria Validada (`Observado · Datos Abiertos`):**
               $\\text{KPI}_1 = \\frac{1}{N_{\\text{días}}} \\sum_{d} \\sum_{e, h} \\text{Validaciones}_{d,e,h}$ sobre `vista_minable.parquet` (`400.84M` validaciones troncales 2026) y proyección `XGBRegressor` (`L2`, `Q10`, `Q90`) para noviembre 2026.
            2. **Índice de Cumplimiento de Programación — ICP (`Paramétrico Oficial`):**
               $\\text{ICP} = \\frac{\\text{Servicios Ejecutados}}{\\text{Servicios Programados}} \\times 100$. Línea base oficial Troncal `96.40%`, sensibilizada en escenarios (`93.50%` Pesimista / `97.80%` Optimista).
            3. **Tiempo Medio de Espera (`Paramétrico Oficial Sensibilizado`):**
               $\\bar{W} = w_{\\text{pico}} W_{\\text{pico}} + w_{\\text{valle}} W_{\\text{valle}}$ con participación observada Hora Pico `60.8%`, línea base `3.50 min` Pico / `7.20 min` Valle, sensibilizada por cumplimiento de flota ($\\text{ICP}_{\\text{base}}/\\text{ICP}_{\\text{esc}}$) y saturación proyectada.
            4. **Tasa de Evasión Focalizada (`Paramétrico Oficial`):**
               Línea base oficial reciente `15.32%` (frente al pico histórico crítico de `28.51%`), sensibilizada según control de torniquetes/barreras (`12.50%` Optimista / `19.00%` Pesimista).
            5. **Kilómetros Comerciales Recorridos (`Paramétrico Oficial Sensibilizado`):**
               $\\text{KM}_{\\text{esc}} = 11.30\\text{ M km} \\times \\frac{\\text{ICP}_{\\text{esc}}}{96.40\\%}$, ajustando la oferta comercial proporcionalmente al cumplimiento efectivo de flota.
            """
        )


def render_active_status_bar(
    troncal: str,
    estacion: str,
    tipos_dia: list[str],
    franjas: list[str],
    escenario_lbl: str,
) -> None:
    """Barra compacta de contexto activo ubicada justo debajo de las pestañas."""
    dias_txt = "Todos los días" if len(tipos_dia) == 3 else ", ".join(TIPO_DIA_LABELS.get(t, t) for t in tipos_dia)
    franjas_txt = "Todas las franjas" if len(franjas) == 3 else ", ".join(FRANJA_LABELS.get(f, f) for f in franjas)
    esc_corto = escenario_lbl.split(" (")[0]
    st.markdown(
        f"""
        <div class="status-strip">
            <div>
                🔎 <b>Vista histórica activa:</b>
                <span class="status-pill">{troncal}</span> ·
                <span class="status-pill">{estacion}</span> ·
                <span class="status-pill">{dias_txt}</span> ·
                <span class="status-pill">{franjas_txt}</span>
            </div>
            <div>
                🎯 <b>Simulador Nov 2026:</b>
                <span class="status-pill-red">Escenario {esc_corto}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# INICIALIZACIÓN DE SESSION STATE Y SIDEBAR INTUITIVO EN 2 BLOQUES
# ==============================================================================
rep_etl, rep_mod, kpis_json = load_json_artifacts()
tables = load_light_tables()
df_minable = load_vista_minable_parquet()

DEFAULT_ESCENARIO = list(ESCENARIO_UI_MAP.keys())[0]
st.session_state.setdefault("flt_escenario", DEFAULT_ESCENARIO)
st.session_state.setdefault("flt_troncal", "Todas las Troncales")
st.session_state.setdefault("flt_estacion", "Todas las Estaciones")
st.session_state.setdefault("flt_tipos_dia", list(TIPOS_DIA_ALL))
st.session_state.setdefault("flt_franjas", list(FRANJAS_ALL))


def _on_troncal_change() -> None:
    """Al cambiar de corredor troncal, restablece la estación a 'Todas las Estaciones'."""
    st.session_state["flt_estacion"] = "Todas las Estaciones"


def _reset_all_filters() -> None:
    """Restablece todos los selectores y pills del panel lateral a su estado inicial."""
    st.session_state["flt_escenario"] = DEFAULT_ESCENARIO
    st.session_state["flt_troncal"] = "Todas las Troncales"
    st.session_state["flt_estacion"] = "Todas las Estaciones"
    st.session_state["flt_tipos_dia"] = list(TIPOS_DIA_ALL)
    st.session_state["flt_franjas"] = list(FRANJAS_ALL)


with st.sidebar:
    st.markdown(
        f"""
        <div style="background:{TM_DARK};color:white;padding:0.8rem 0.95rem;border-radius:10px;border-left:5px solid {TM_RED};margin-bottom:0.75rem;">
            <div style="font-weight:800;font-size:1.02rem;letter-spacing:-0.01em;">🚍 TRANSMILENIO S.A.</div>
            <div style="font-size:0.75rem;color:#CBD5E1;">Plataforma Analítica · Red Troncal 2026</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # BLOQUE A: SIMULADOR DE ESCENARIOS
    with st.container(border=True):
        st.markdown('<div class="sidebar-block-tag">Bloque A · Proyección (Pestaña 5)</div>', unsafe_allow_html=True)
        escenario_label = st.selectbox(
            "Escenario de Simulación (Nov 2026):",
            options=list(ESCENARIO_UI_MAP.keys()),
            key="flt_escenario",
        )
        esc_json_key, esc_col_dia, esc_col_tot_est, esc_col_prom_est = ESCENARIO_UI_MAP[escenario_label]

    # BLOQUE B: FILTROS OPERATIVOS DEL HISTÓRICO Y EXPLORACIÓN
    with st.container(border=True):
        st.markdown('<div class="sidebar-block-tag">Bloque B · Filtros Operativos</div>', unsafe_allow_html=True)

        lista_troncales = ["Todas las Troncales"] + sorted(df_minable["Troncal"].unique().tolist())
        sel_troncal = st.selectbox(
            "Corredor Troncal:",
            options=lista_troncales,
            key="flt_troncal",
            on_change=_on_troncal_change,
        )

        if sel_troncal == "Todas las Troncales":
            estaciones_disp = sorted(df_minable["Nombre_Estacion"].unique().tolist())
        else:
            estaciones_disp = sorted(
                df_minable.loc[df_minable["Troncal"] == sel_troncal, "Nombre_Estacion"].unique().tolist()
            )
        est_options = ["Todas las Estaciones"] + estaciones_disp
        if st.session_state.get("flt_estacion") not in est_options:
            st.session_state["flt_estacion"] = "Todas las Estaciones"

        sel_estacion = st.selectbox(
            "Estación Troncal:",
            options=est_options,
            key="flt_estacion",
        )

        if hasattr(st, "pills"):
            raw_tipos = st.pills(
                "Tipo de Día:",
                options=TIPOS_DIA_ALL,
                selection_mode="multi",
                format_func=lambda x: TIPO_DIA_LABELS.get(x, x),
                key="flt_tipos_dia",
                wrap=True,
            )
            raw_franjas = st.pills(
                "Franja Horaria:",
                options=FRANJAS_ALL,
                selection_mode="multi",
                format_func=lambda x: FRANJA_LABELS.get(x, x),
                key="flt_franjas",
                wrap=True,
            )
        else:
            raw_tipos = st.multiselect(
                "Tipo de Día:",
                options=TIPOS_DIA_ALL,
                format_func=lambda x: TIPO_DIA_LABELS.get(x, x),
                key="flt_tipos_dia",
            )
            raw_franjas = st.multiselect(
                "Franja Horaria:",
                options=FRANJAS_ALL,
                format_func=lambda x: FRANJA_LABELS.get(x, x),
                key="flt_franjas",
            )

        sel_tipos_dia = list(raw_tipos) if raw_tipos else list(TIPOS_DIA_ALL)
        sel_franjas = list(raw_franjas) if raw_franjas else list(FRANJAS_ALL)

    st.button(
        "↺ Restablecer filtros",
        on_click=_reset_all_filters,
        width="stretch",
    )

    # Aplicar máscara filtrada sobre vista_minable.parquet
    mask = df_minable["Tipo_Dia"].isin(sel_tipos_dia) & df_minable["Franja_Horaria"].isin(sel_franjas)
    if sel_troncal != "Todas las Troncales":
        mask &= df_minable["Troncal"] == sel_troncal
    if sel_estacion != "Todas las Estaciones":
        mask &= df_minable["Nombre_Estacion"] == sel_estacion
    df_filtered = df_minable.loc[mask]

    is_custom_filter = (
        sel_troncal != "Todas las Troncales"
        or sel_estacion != "Todas las Estaciones"
        or len(sel_tipos_dia) < 3
        or len(sel_franjas) < 3
    )

    st.caption(
        f"**Filas activas:** `{len(df_filtered):,}` / `{len(df_minable):,}`  \n"
        f"**Estaciones activas:** `{df_filtered['ID_Estacion'].nunique()}` · **Corte:** `Ene–Oct 2026`"
    )

# ==============================================================================
# ENCABEZADO INSTITUCIONAL PRINCIPAL Y PESTAÑAS TIPO PILL
# ==============================================================================
st.markdown(
    """
    <div class="tm-hero">
        <h1>Plataforma de Minería de Datos y Proyección Operativa — TransMilenio S.A.</h1>
        <p>Trazabilidad End-to-End: Data Lake Abierto (ArcGIS / FCS Center) → Pipeline ETL Auditable → Clustering K-Means & Pronóstico XGBoost → Simulador de 3 Escenarios (2026)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏛️ 1. Contexto y Arquitectura",
        "🛠️ 2. Auditoría ETL",
        "📊 3. KPIs e Histórico",
        "🗺️ 4. Clústeres y Mapa Bogotá",
        "🤖 5. XGBoost y 3 Escenarios",
    ]
)

# ==============================================================================
# PESTAÑA 1: CONTEXTO CORPORATIVO Y ARQUITECTURA DE 4 FASES
# ==============================================================================
with tab1:
    render_active_status_bar(sel_troncal, sel_estacion, sel_tipos_dia, sel_franjas, escenario_label)

    col_corp1, col_corp2 = st.columns([1.22, 1.0])
    with col_corp1:
        with st.container(border=True):
            st.markdown(f"#### <span style='color:{TM_RED};'>Modelo Institucional de TransMilenio S.A.</span>", unsafe_allow_html=True)
            st.markdown(
                "**TransMilenio S.A.** actúa como el **Ente Gestor público** del Sistema Integrado de Transporte Público (SITP) de Bogotá D.C. bajo un esquema de **Asociación Público-Privada (APP)**: la entidad pública planea la red, regula frecuencias, audita niveles de servicio (ICP) y administra la infraestructura troncal, mientras concesionarios privados operan la flota articulada/biarticulada y el sistema centralizado de recaudo electrónico (*Tarjeta TuLlave · FCS Center*).\n\n"
                "* **Misión:** Integrar, planear, gestionar y controlar el transporte público masivo de Bogotá y su región metropolitana con criterios de calidad, accesibilidad, eficiencia operativa y sostenibilidad financiera y ambiental.\n"
                "* **Visión:** Consolidarse como el eje estructurador de movilidad multimodal de Bogotá-Región, articulando troncales BRT, cables aéreos y la Primera Línea del Metro mediante decisiones basadas en analítica avanzada de datos."
            )

    with col_corp2:
        with st.container(border=True):
            st.markdown(f"#### <span style='color:{TM_DARK};'>3 Objetivos Corporativos del Proyecto Analítico</span>", unsafe_allow_html=True)
            st.markdown(
                "1. **Segmentación Inteligente de Infraestructura:** Clasificar las `142` estaciones troncales activas según su asimetría horaria (generadoras AM vs. atractoras PM), capacidad de vagones e índice de saturación mediante *K-Means*.\n"
                "2. **Pronóstico Horario de Alta Precisión:** Anticipar la demanda validada por estación y bloque horario con *XGBoost Regressor* (partición cronológica sin fuga de datos) capturando cierres por obras del Metro y aperturas como Av. Ciudad de Cali.\n"
                "3. **Planeación Táctica bajo 3 Escenarios:** Simular la operación a 30 días (Noviembre 2026) en escenarios *Pesimista (Q10)*, *Base (L2)* y *Optimista (Q90 + control de evasión)* vinculados a 5 KPIs ejecutivos."
            )

    st.subheader("🔄 Arquitectura End-to-End del Pipeline de Minería de Datos (4 Fases)")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(
            """
            <div class="phase-card">
                <div class="phase-step">Fase 1 · Data Lake Crudo</div>
                <div class="phase-title">data/raw/ + src/eda_crudo.py</div>
                <ul style="font-size:0.83rem;color:#475569;padding-left:1.1rem;margin:0;">
                    <li><b>10 libros .xlsx</b> mensuales (Ene–Oct 2026, intervalo 15 min, <code>785,250</code> filas en plantilla).</li>
                    <li><b>Maestro ArcGIS</b> (<code>152</code> estaciones troncales en <code>.csv</code> y <code>.geojson</code>).</li>
                    <li>Detección de homónimos y desfase de códigos.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            """
            <div class="phase-card">
                <div class="phase-step">Fase 2 · ETL y Transformación</div>
                <div class="phase-title">src/etl.py → data/processed/</div>
                <ul style="font-size:0.83rem;color:#475569;padding-left:1.1rem;margin:0;">
                    <li>Eliminación de <code>220,499</code> filas vacías y <code>6</code> fechas desbordadas.</li>
                    <li>Imputación por mediana (<code>tipo_esta</code>), homologación de <code>6</code> nodos y <code>StandardScaler</code>.</li>
                    <li><b>400.84M validaciones</b> → <code>787,951</code> filas Hora–Estación (<b>0 nulos</b>).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            """
            <div class="phase-card">
                <div class="phase-step">Fase 3 · Modelos ML Python</div>
                <div class="phase-title">src/modelo.py → data/output/</div>
                <ul style="font-size:0.83rem;color:#475569;padding-left:1.1rem;margin:0;">
                    <li><b>K-Means (k=4):</b> Silhouette <code>0.2252</code> + proyección PCA 2D y perfil Radar.</li>
                    <li><b>XGBRegressor (hist):</b> <code>R² = 0.9724</code> horario y <code>WAPE = 2.58%</code> diario en Test.</li>
                    <li>Cuantiles residuales <code>Q10 / Q90</code> para Noviembre 2026.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p4:
        st.markdown(
            """
            <div class="phase-card">
                <div class="phase-step">Fase 4 · Visualización Web</div>
                <div class="phase-title">dashboard/app.py (Cloud)</div>
                <ul style="font-size:0.83rem;color:#475569;padding-left:1.1rem;margin:0;">
                    <li>Carga ultrarrápida con <code>@st.cache_data</code> sobre <code>vista_minable.parquet</code> (<code>2.02 MB</code>).</li>
                    <li>Mapa interactivo CARTO de Bogotá sin token.</li>
                    <li>Tablero de <b>5 KPIs Corporativos</b> y simulador de 3 escenarios.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        funnel_df = pd.DataFrame(
            {
                "Etapa": [
                    "1. Filas crudas leídas en 10 libros Excel (.xlsx)",
                    "2. Filas operativas tras limpiar plantilla y 'Total general'",
                    "3. Filas 15-min en alcance Troncal y horario 04:00–23:59",
                    "4. Vista Minable consolidada Hora–Estación (304 días × 142 est.)",
                ],
                "Registros": [
                    rep_etl["limpieza_ingesta_xlsx_10_meses"]["filas_crudas_totales_bajo_header"],
                    rep_etl["limpieza_ingesta_xlsx_10_meses"]["filas_operativas_evaluadas"],
                    rep_etl["filtro_horario_operativo"]["filas_15min_mensuales_en_alcance_comercial"],
                    rep_etl["salidas_generadas"]["vista_minable_parquet"]["filas"],
                ],
            }
        )
        fig_vol = px.bar(
            funnel_df,
            x="Registros",
            y="Etapa",
            orientation="h",
            text="Registros",
            color="Etapa",
            color_discrete_sequence=[TM_SLATE, TM_BLUE, TM_AMBER, TM_RED],
            title="Trazabilidad Volumétrica: De Plantillas Crudas Mensuales a Vista Minable Hora–Estación",
        )
        fig_vol.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
        fig_vol.update_layout(yaxis=dict(autorange="reversed", title=""), xaxis=dict(range=[0, 940000], title="Registros"))
        st.plotly_chart(apply_plotly_style(fig_vol, height=300, show_legend=False, right_margin=55), width="stretch")


# ==============================================================================
# PESTAÑA 2: AUDITORÍA DEL PIPELINE ETL (TRANSPARENCIA TÉCNICA)
# ==============================================================================
with tab2:
    render_active_status_bar(sel_troncal, sel_estacion, sel_tipos_dia, sel_franjas, escenario_label)
    st.subheader("🛡️ Auditoría de Calidad y Depuración del Proceso ETL (`reporte_etl.json`)")

    limp = rep_etl["limpieza_ingesta_xlsx_10_meses"]
    alc = rep_etl["filtro_alcance_y_homologacion"]
    cruce = rep_etl["cruce_y_calidad_final"]

    with st.container(border=True):
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric(
            "Filas Vacías Eliminadas",
            f"{limp['filas_vacias_plantilla_eliminadas']:,}",
            f"+ {limp['filas_resumen_total_general_eliminadas']} filas 'Total general'",
        )
        a2.metric(
            "Fechas Desbordadas",
            f"{limp['total_columnas_fecha_desbordadas_eliminadas']} cols",
            "Feb (3), Abr (1), Jun (1), Sep (1)",
        )
        a3.metric(
            "Excluidas por Alcance",
            f"{alc['exclusiones_por_alcance']['total_validaciones_excluidas_alcance'] / 1e6:.2f} M val",
            "TransMiCable + Buses Dual en calle",
        )
        a4.metric(
            "Match Llave cod_nodo",
            f"{cruce['pct_match_cod_nodo_en_alcance']:.2f}%",
            f"{alc['validaciones_recuperadas_por_homologacion']:,} val rescatadas",
        )
        a5.metric(
            "Valores Nulos Finales",
            f"{cruce['confirmacion_cero_nulos']['nulos_vista_minable']}",
            "100% libre de NaN (Parquet/CSV)",
        )

    col_bal, col_mes = st.columns([1.0, 1.15])
    with col_bal:
        with st.container(border=True):
            balance_df = pd.DataFrame(
                {
                    "Categoría": [
                        "Conservadas Troncal (04–23h)",
                        "Excluidas: Dual en Calle",
                        "Excluidas: TransMiCable",
                        "Filtradas: Madrugada (00–03h)",
                    ],
                    "Validaciones": [
                        cruce["total_validaciones_limpias_vista_minable"],
                        alc["exclusiones_por_alcance"]["dual_en_calle_fuera_de_troncal"]["validaciones_excluidas"],
                        alc["exclusiones_por_alcance"]["transmicable"]["validaciones_excluidas"],
                        rep_etl["filtro_horario_operativo"]["madrugada_00_03_filtrada"]["validaciones_residuales_filtradas"],
                    ],
                }
            )
            fig_bal = px.pie(
                balance_df,
                names="Categoría",
                values="Validaciones",
                hole=0.48,
                color="Categoría",
                color_discrete_map={
                    "Conservadas Troncal (04–23h)": TM_RED,
                    "Excluidas: Dual en Calle": "#64748B",
                    "Excluidas: TransMiCable": TM_AMBER,
                    "Filtradas: Madrugada (00–03h)": "#94A3B8",
                },
                title="Balance Auditado de Validaciones en los 10 Libros 2026",
            )
            st.plotly_chart(apply_plotly_style(fig_bal, height=365, show_legend=True, bottom_margin=85), width="stretch")

    with col_mes:
        with st.container(border=True):
            det_mes_df = pd.DataFrame(rep_etl["detalle_mensual"])
            det_mes_df["Millones_Val"] = (det_mes_df["validaciones_limpias_mes"] / 1e6).round(2)
            fig_mes_etl = px.bar(
                det_mes_df,
                x="mes_objetivo",
                y="Millones_Val",
                text="Millones_Val",
                color_discrete_sequence=[TM_DARK],
                title="Validaciones Troncales Limpias Procesadas por Mes (Millones)",
                labels={"mes_objetivo": "Mes 2026", "Millones_Val": "Millones de Validaciones"},
            )
            fig_mes_etl.update_traces(texttemplate="%{text:.2f}M", textposition="outside", cliponaxis=False)
            fig_mes_etl.update_layout(yaxis_range=[0, 54])
            st.plotly_chart(apply_plotly_style(fig_mes_etl, height=365, show_legend=False), width="stretch")

    st.markdown("### 🔬 Trazabilidad de las 3 Transformaciones Obligatorias del ETL")
    with st.container(border=True):
        sub_imp, sub_norm, sub_esc = st.tabs(
            [
                "1️⃣ Imputación de Ceros Estructurales",
                "2️⃣ Normalización, Homologación y Festivos",
                "3️⃣ Escalamiento (StandardScaler)",
            ]
        )

        with sub_imp:
            st.markdown(
                "En el maestro ArcGIS de estaciones se descartaron **11 columnas administrativas/constantes** "
                "(`esta_oper`, `observ`, `created_user`, `created_date`, `last_edited_user`, `last_edited_date`, `objectid`, `globalid`, `X`, `Y`, `num_est`) "
                "y los **falsos ceros estructurales** se convirtieron a `NaN` para imputarse mediante la **mediana agrupada por `tipo_esta`** "
                "(con respaldo de mediana global en los 9 Portales `tipo_esta = 1` donde el polígono GIS registraba área `0`):"
            )
            imp_dict = rep_etl["maestro_estaciones_e_imputacion"]["imputacion_ceros_estructurales"]
            imp_rows = [{"Variable_Infraestructura": k, **v} for k, v in imp_dict.items()]
            st.dataframe(pd.DataFrame(imp_rows), width="stretch", hide_index=True)

        with sub_norm:
            cn1, cn2 = st.columns([1.25, 1.0])
            with cn1:
                st.markdown("**Tabla de Homologación de los 6 Códigos Desfasados (`FCS Center .xlsx` → `ArcGIS cod_nodo`):**")
                homolog_df = pd.DataFrame(
                    [
                        {"Código en Excel (FCS)": "(15001) Tibanica - Primavera", "cod_nodo ArcGIS": 5011, "Estación Maestro": "Tibanica - Primavera", "Causa": "Nueva extensión Av. Ciudad de Cali (TZ022)"},
                        {"Código en Excel (FCS)": "(15002) Los Laureles", "cod_nodo ArcGIS": 5010, "Estación Maestro": "Los Laureles", "Causa": "Nueva extensión Av. Ciudad de Cali (TZ022)"},
                        {"Código en Excel (FCS)": "(15003) Islandia", "cod_nodo ArcGIS": 5009, "Estación Maestro": "Islandia", "Causa": "Nueva extensión Av. Ciudad de Cali (TZ022)"},
                        {"Código en Excel (FCS)": "(57503) Ampliacion San Mateo", "cod_nodo ArcGIS": 7503, "Estación Maestro": "San Mateo - CC Unisur", "Causa": "Plataforma de ampliación integrada al nodo"},
                        {"Código en Excel (FCS)": "(04004) La Granja", "cod_nodo ArcGIS": 4100, "Estación Maestro": "Granja - Kr 77", "Causa": "Unificación de vagones en polígono GIS"},
                        {"Código en Excel (FCS)": "(50003) Corral Molinos", "cod_nodo ArcGIS": 9001, "Estación Maestro": "Molinos", "Causa": "Corral operativo anexo a estación"},
                    ]
                )
                st.dataframe(homolog_df, width="stretch", hide_index=True)
                st.info(
                    "El cruce numérico por `cod_nodo` (`int64`) rescató además **62 estaciones** cuyo nombre comercial difiere entre recaudo y GIS "
                    "(ej. `(04000) Cabecera Calle 80` ↔ `Portal 80`, `(02001) C.C. Santa Fe` ↔ `Calle 187`) y desambiguó "
                    "el homónimo `Temporal AV. Jiménez - Inter Eléctricas` (`9110` Caracas vs `14003` Eje Ambiental)."
                )
            with cn2:
                st.markdown("**14 Festivos Oficiales de Colombia (Ene–Oct 2026) integrados en `Tipo_Dia = Domingo_Festivo`:**")
                fest_list = rep_etl["archivos_procesados"]["rango_fechas_vista_minable"]["festivos_colombia_2026_en_rango"]
                fest_nombres = {
                    "2026-01-01": "Año Nuevo",
                    "2026-01-12": "Reyes Magos (Ley Emiliani)",
                    "2026-03-23": "Día de San José (Ley Emiliani)",
                    "2026-04-02": "Jueves Santo",
                    "2026-04-03": "Viernes Santo",
                    "2026-05-01": "Día del Trabajo",
                    "2026-05-18": "Ascensión del Señor",
                    "2026-06-08": "Corpus Christi",
                    "2026-06-15": "Sagrado Corazón de Jesús",
                    "2026-06-29": "San Pedro y San Pablo",
                    "2026-07-20": "Día de la Independencia",
                    "2026-08-07": "Batalla de Boyacá",
                    "2026-08-17": "Asunción de la Virgen",
                    "2026-10-12": "Día de la Raza",
                }
                st.dataframe(
                    pd.DataFrame([{"Fecha": d, "Celebración Oficial": fest_nombres.get(d, "Festivo Nacional")} for d in fest_list]),
                    width="stretch",
                    hide_index=True,
                    height=310,
                )

        with sub_esc:
            vars_escaladas = rep_etl["salidas_generadas"]["perfil_estaciones_escalado_csv"]["variables_escaladas"]
            default_vars = ["prom_diario_habil", "max_validaciones_hora", "pct_pico_am", "area_est", "cap_art"]
            sel_vars_scale = st.multiselect(
                "Variables del perfil de estación a comparar Antes vs. Después de StandardScaler:",
                options=vars_escaladas,
                default=default_vars,
            )
            if not sel_vars_scale:
                sel_vars_scale = default_vars

            fig_scale = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    "Antes de Escalar (Magnitudes Originales)",
                    "Después de StandardScaler (Media = 0, Desv. = 1)",
                ),
            )
            raw_melt = tables["perfil_raw"][sel_vars_scale].melt(var_name="Variable", value_name="Valor")
            sc_melt = tables["perfil_scaled"][sel_vars_scale].melt(var_name="Variable", value_name="Z_Score")

            for var_name in sel_vars_scale:
                fig_scale.add_trace(
                    go.Box(
                        y=raw_melt.loc[raw_melt["Variable"] == var_name, "Valor"],
                        name=var_name,
                        marker_color=TM_SLATE,
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                fig_scale.add_trace(
                    go.Box(
                        y=sc_melt.loc[sc_melt["Variable"] == var_name, "Z_Score"],
                        name=var_name,
                        marker_color=TM_RED,
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
            fig_scale.update_yaxes(type="log", title_text="Escala original (log)", row=1, col=1)
            fig_scale.update_yaxes(title_text="Puntaje Z estandarizado", row=1, col=2)
            st.plotly_chart(apply_plotly_style(fig_scale, height=370, show_legend=False), width="stretch")

    with st.container(border=True):
        st.markdown("#### 📋 Explorador Interactivo de Muestra — `vista_minable.parquet` (Hora–Estación)")
        st.caption(f"Mostrando primeras 300 filas de `{len(df_filtered):,}` que coinciden con los filtros activos.")
        st.dataframe(df_filtered.head(300), width="stretch", hide_index=True, height=270)


# ==============================================================================
# PESTAÑA 3: DASHBOARD EJECUTIVO DE KPIS Y OPERACIÓN HISTÓRICA (ENE–OCT 2026)
# ==============================================================================
with tab3:
    render_active_status_bar(sel_troncal, sel_estacion, sel_tipos_dia, sel_franjas, escenario_label)

    hdr_kpi_col, pop_kpi_col = st.columns([3.2, 1.0], vertical_alignment="center")
    with hdr_kpi_col:
        st.subheader("📌 Tablero Ejecutivo de los 5 KPIs Corporativos (Referencia Global del Sistema)")
    with pop_kpi_col:
        render_kpi_methodology_popover("pop_kpis_tab3")

    hist_kpis = kpis_json["escenarios"]["Linea_Base_Historica_2026"]

    # Calcular promedio diario real de la selección activa cuando el usuario aplica un filtro
    n_dias_filtro = max(1, int(df_filtered["Fecha"].nunique()))
    total_val_filtro = int(df_filtered["Total_Validaciones"].sum())
    prom_diario_filtro = int(round(total_val_filtro / n_dias_filtro))
    pct_sobre_sistema = (
        total_val_filtro / max(1, int(df_minable["Total_Validaciones"].sum())) * 100.0
    )

    extra_kpi1_html = ""
    if is_custom_filter:
        extra_kpi1_html = (
            f'<div class="kpi-filter-highlight">'
            f'Filtro activo: {prom_diario_filtro:,} val/día ({pct_sobre_sistema:.2f}% del total sistema)'
            f"</div>"
        )

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        d1 = hist_kpis["KPI_1_Demanda_Validada_Troncal"]
        st.markdown(
            render_kpi_card(
                "1. Demanda Diaria Sistema",
                f"{d1['promedio_diario_global']:,}",
                f"Hábil: {d1['promedio_dia_habil']:,} | Sáb: {d1['promedio_sabado']:,} | Dom/Fest: {d1['promedio_domingo_festivo']:,}",
                es_observado=True,
                tooltip_formula="Promedio diario observado sobre 304 días en vista_minable.parquet (400.84M validaciones troncales).",
                extra_highlight_html=extra_kpi1_html,
            ),
            unsafe_allow_html=True,
        )
    with k2:
        d2 = hist_kpis["KPI_2_Indice_Cumplimiento_Programacion_ICP_pct"]
        st.markdown(
            render_kpi_card(
                "2. Cumplimiento ICP",
                f"{d2['valor_pct']:.2f}%",
                "Servicios troncales ejecutados vs. programados",
                es_observado=False,
                tooltip_formula="Línea base oficial Troncal TransMilenio S.A. (96.40% servicios ejecutados sobre programados).",
            ),
            unsafe_allow_html=True,
        )
    with k3:
        d3 = hist_kpis["KPI_3_Tiempo_Medio_Espera_Troncal_min"]
        st.markdown(
            render_kpi_card(
                "3. Tiempo Medio Espera",
                f"{d3['ponderado_sistema_min']:.2f} min",
                f"Hora Pico: {d3['hora_pico_min']:.2f} min | Valle: {d3['hora_valle_min']:.2f} min",
                es_observado=False,
                tooltip_formula="Ponderado según peso real observado de Horas Pico (60.8%) y Horas Valle (39.2%).",
            ),
            unsafe_allow_html=True,
        )
    with k4:
        d4 = hist_kpis["KPI_4_Tasa_Evasion_Focalizada_pct"]
        st.markdown(
            render_kpi_card(
                "4. Evasión Focalizada",
                f"{d4['valor_pct']:.2f}%",
                f"Reducción frente al pico crítico histórico ({d4['referencia_historica_critica_pct']}%)",
                es_observado=False,
                tooltip_formula="Medición oficial de evasión en componente Troncal (15.32% vs. máximo histórico de 28.51%).",
            ),
            unsafe_allow_html=True,
        )
    with k5:
        d5 = hist_kpis["KPI_5_Kilometros_Comerciales_Troncal_millones_mes"]
        st.markdown(
            render_kpi_card(
                "5. Km Comerciales / Mes",
                f"{d5['millones_km_mes']:.2f} M km",
                f"{d5['km_exactos_mes']:,} km mensuales flota troncal",
                es_observado=False,
                tooltip_formula="Oferta comercial mensual de flota troncal (11.30 millones km/mes en línea base).",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    # 1. Serie histórica diaria + Insight dinámico y detector automático de cierre/caída operativa
    with st.container(border=True):
        daily_filtered = (
            df_filtered.groupby(["Fecha", "Tipo_Dia"], as_index=False)["Total_Validaciones"]
            .sum()
            .sort_values("Fecha")
        )
        daily_filtered["Media_Movil_7d"] = daily_filtered["Total_Validaciones"].rolling(7, min_periods=1).mean()

        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=daily_filtered["Fecha"],
                y=daily_filtered["Total_Validaciones"],
                mode="lines",
                name="Validaciones Diarias",
                line=dict(color="#94A3B8", width=1.3),
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=daily_filtered["Fecha"],
                y=daily_filtered["Media_Movil_7d"],
                mode="lines",
                name="Tendencia (Media Móvil 7 días)",
                line=dict(color=TM_RED, width=2.8),
            )
        )
        fig_ts.update_layout(
            title=f"Evolución Diaria de Validaciones ({sel_troncal} · {sel_estacion})",
            xaxis_title="Fecha (Enero – Octubre 2026)",
            yaxis_title="Validaciones / Día",
        )
        st.plotly_chart(apply_plotly_style(fig_ts, height=360, show_legend=True), width="stretch")

        if len(daily_filtered) > 0:
            idx_max = daily_filtered["Total_Validaciones"].idxmax()
            peak_day_row = daily_filtered.loc[idx_max]
            st.markdown(
                f"""
                <div class="insight-bar">
                    💡 <b>Insight de Serie Temporal:</b> El pico máximo en la selección ocurrió el <b>{peak_day_row['Fecha']}</b>
                    ({TIPO_DIA_LABELS.get(str(peak_day_row['Tipo_Dia']), str(peak_day_row['Tipo_Dia']))}) con
                    <b>{int(peak_day_row['Total_Validaciones']):,} validaciones</b>, frente a un promedio de
                    <b>{prom_diario_filtro:,} val/día</b> en los <code>{n_dias_filtro}</code> días filtrados.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Detección automática de cierre o caída abrupta reciente (Octubre vs promedio previo)
            oct_vals = daily_filtered.loc[daily_filtered["Fecha"] >= "2026-10-01", "Total_Validaciones"]
            prev_vals = daily_filtered.loc[daily_filtered["Fecha"] < "2026-09-01", "Total_Validaciones"]
            oct_mean = float(oct_vals.mean()) if len(oct_vals) > 0 else 0.0
            prev_mean = float(prev_vals.mean()) if len(prev_vals) > 0 else 0.0

            if prev_mean > 100 and oct_mean == 0:
                st.markdown(
                    """
                    <div class="insight-alert">
                        ⚠️ <b>Alerta Operativa Detectada:</b> La selección presenta <b>cierre operativo total (0 validaciones en octubre 2026)</b>
                        tras haber registrado actividad en meses previos (asociado a cierres programados por obras de infraestructura / Línea 1 del Metro).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif prev_mean > 100 and oct_mean < 0.60 * prev_mean:
                caida_pct = (1.0 - oct_mean / prev_mean) * 100.0
                st.markdown(
                    f"""
                    <div class="insight-alert">
                        ⚠️ <b>Alerta de Variación Operativa:</b> En octubre 2026 la demanda media de esta selección cayó un
                        <b>{caida_pct:.1f}%</b> respecto al promedio Enero–Agosto (de <code>{prev_mean:,.0f}</code> a <code>{oct_mean:,.0f} val/día</code>).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # 2. Heatmap Hora x Día de Semana y Distribución por Franja (sin solapamientos)
    ch1, ch2 = st.columns([1.32, 1.0])
    with ch1:
        with st.container(border=True):
            hourly_day = (
                df_filtered.groupby(["Fecha", "Dia_Semana", "Hora_Bloque"], as_index=False)["Total_Validaciones"]
                .sum()
                .groupby(["Dia_Semana", "Hora_Bloque"], as_index=False)["Total_Validaciones"]
                .mean()
            )
            heat_piv = (
                hourly_day.pivot(index="Dia_Semana", columns="Hora_Bloque", values="Total_Validaciones")
                .reindex(index=range(7), columns=range(4, 24), fill_value=0.0)
            )
            heat_piv.index = [DIAS_NOMBRES[i] for i in heat_piv.index]

            fig_hm = px.imshow(
                heat_piv,
                labels=dict(x="Hora del Día (04:00 – 23:00)", y="Día de la Semana", color="Prom. Val"),
                color_continuous_scale="YlOrRd",
                aspect="auto",
                title="Mapa de Calor Horario: Intensidad Media Hora × Día de la Semana",
            )
            fig_hm.update_coloraxes(colorbar=dict(thickness=13, len=0.84))
            st.plotly_chart(apply_plotly_style(fig_hm, height=380, show_legend=False), width="stretch")

            if len(hourly_day) > 0 and hourly_day["Total_Validaciones"].max() > 0:
                top_hd = hourly_day.loc[hourly_day["Total_Validaciones"].idxmax()]
                dia_pico_nom = DIAS_NOMBRES.get(int(top_hd["Dia_Semana"]), "")
                hora_pico_int = int(top_hd["Hora_Bloque"])
                st.markdown(
                    f"""
                    <div class="insight-bar">
                        ⏱️ <b>Bloque Crítico:</b> Máxima presión horaria los <b>{dia_pico_nom} a las {hora_pico_int:02d}:00–{hora_pico_int:02d}:59</b>
                        con un promedio de <b>{top_hd['Total_Validaciones']:,.0f} validaciones/hora</b>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with ch2:
        with st.container(border=True):
            franja_agg = (
                df_filtered.groupby(["Tipo_Dia", "Franja_Horaria"], as_index=False)["Total_Validaciones"]
                .sum()
            )
            franja_agg["Tipo_Dia_Lbl"] = franja_agg["Tipo_Dia"].map(TIPO_DIA_LABELS).fillna(franja_agg["Tipo_Dia"])
            franja_agg["Franja_Lbl"] = franja_agg["Franja_Horaria"].map(FRANJA_LABELS).fillna(franja_agg["Franja_Horaria"])

            fig_fr = px.bar(
                franja_agg,
                x="Tipo_Dia_Lbl",
                y="Total_Validaciones",
                color="Franja_Lbl",
                barmode="group",
                color_discrete_map={
                    "Pico AM (05–08h)": TM_RED,
                    "Pico PM (16–19h)": TM_DARK,
                    "Valle": "#64748B",
                },
                title="Distribución por Tipo de Día y Franja Horaria",
                labels={"Total_Validaciones": "Validaciones Acumuladas", "Tipo_Dia_Lbl": "Tipo de Día", "Franja_Lbl": ""},
            )
            st.plotly_chart(apply_plotly_style(fig_fr, height=380, show_legend=True, bottom_margin=82), width="stretch")

            tot_sel = max(1, int(franja_agg["Total_Validaciones"].sum()))
            val_am = int(franja_agg.loc[franja_agg["Franja_Horaria"] == "Pico_AM", "Total_Validaciones"].sum())
            val_pm = int(franja_agg.loc[franja_agg["Franja_Horaria"] == "Pico_PM", "Total_Validaciones"].sum())
            pct_pico_tot = (val_am + val_pm) / tot_sel * 100.0
            st.markdown(
                f"""
                <div class="insight-bar">
                    📊 <b>Concentración Horaria:</b> Las horas pico agrupan el <b>{pct_pico_tot:.1f}%</b> de las validaciones
                    (<code>Pico AM: {val_am/tot_sel*100:.1f}%</code> · <code>Pico PM: {val_pm/tot_sel*100:.1f}%</code>) frente a <b>{100-pct_pico_tot:.1f}%</b> en Valle.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 3. Top 15 estaciones
    with st.container(border=True):
        top15_df = (
            df_filtered.groupby(["Nombre_Estacion", "Troncal"], as_index=False)["Total_Validaciones"]
            .sum()
            .sort_values("Total_Validaciones", ascending=False)
            .head(15)
        )
        max_bar_val = float(top15_df["Total_Validaciones"].max()) if len(top15_df) > 0 else 1.0
        fig_top = px.bar(
            top15_df,
            x="Total_Validaciones",
            y="Nombre_Estacion",
            color="Troncal",
            orientation="h",
            text="Total_Validaciones",
            title="Top 15 Estaciones Troncales con Mayor Demanda Acumulada (según filtros activos)",
            labels={"Total_Validaciones": "Validaciones Acumuladas (Ene–Oct 2026)", "Nombre_Estacion": ""},
        )
        fig_top.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
        fig_top.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis=dict(range=[0, max_bar_val * 1.18]),
        )
        st.plotly_chart(apply_plotly_style(fig_top, height=450, show_legend=True, bottom_margin=85, right_margin=45), width="stretch")

        if len(top15_df) > 0 and total_val_filtro > 0:
            share_top = top15_df["Total_Validaciones"].sum() / total_val_filtro * 100.0
            lider_nom = top15_df.iloc[0]["Nombre_Estacion"]
            lider_val = int(top15_df.iloc[0]["Total_Validaciones"])
            st.markdown(
                f"""
                <div class="insight-bar">
                    🏆 <b>Efecto Pareto Espacial:</b> El Top {len(top15_df)} concentra el <b>{share_top:.1f}%</b> de la demanda seleccionada;
                    lidera <b>{lider_nom}</b> con <b>{lider_val:,} validaciones</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# PESTAÑA 4: SEGMENTACIÓN K-MEANS Y MAPA GEOESPACIAL DE BOGOTÁ
# ==============================================================================
with tab4:
    render_active_status_bar(sel_troncal, sel_estacion, sel_tipos_dia, sel_franjas, escenario_label)
    st.subheader("🗺️ Segmentación No Supervisada K-Means ($k = 4$) y Distribución Espacial en Bogotá")

    df_cl = tables["clusters"].copy()
    if sel_troncal != "Todas las Troncales":
        df_cl_map = df_cl[df_cl["Troncal"] == sel_troncal].copy()
    else:
        df_cl_map = df_cl.copy()

    cm_left, cm_right = st.columns([1.32, 1.0])
    with cm_left:
        with st.container(border=True):
            scatter_geo_fn = getattr(px, "scatter_map", None)
            if scatter_geo_fn is not None:
                fig_map = scatter_geo_fn(
                    df_cl_map,
                    lat="latitud",
                    lon="longitud",
                    color="Etiqueta_Cluster",
                    size="prom_diario_habil",
                    size_max=26,
                    zoom=10.3,
                    center={"lat": 4.648, "lon": -74.105},
                    map_style="carto-positron",
                    hover_name="Nombre_Estacion",
                    hover_data={
                        "Troncal": True,
                        "prom_diario_habil": ":,.0f",
                        "max_validaciones_hora": ":,.0f",
                        "pct_pico_am": ":.1f",
                        "pct_pico_pm": ":.1f",
                        "capacidad_total_est": True,
                        "indice_saturacion_pico": ":.2f",
                        "latitud": False,
                        "longitud": False,
                    },
                    color_discrete_map=CLUSTER_COLORS,
                    title="Mapa Geoespacial de Estaciones Troncales por Clúster Operativo",
                )
            else:
                fig_map = px.scatter_mapbox(
                    df_cl_map,
                    lat="latitud",
                    lon="longitud",
                    color="Etiqueta_Cluster",
                    size="prom_diario_habil",
                    size_max=26,
                    zoom=10.3,
                    center={"lat": 4.648, "lon": -74.105},
                    mapbox_style="carto-positron",
                    hover_name="Nombre_Estacion",
                    color_discrete_map=CLUSTER_COLORS,
                    title="Mapa Geoespacial de Estaciones Troncales por Clúster Operativo",
                )
            fig_map.update_layout(
                height=490,
                margin=dict(l=0, r=0, t=48, b=72),
                title=dict(x=0.01, font=dict(size=14, color=TM_DARK)),
                legend=dict(
                    title_text="",
                    orientation="h",
                    yanchor="top",
                    y=-0.04,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10.5),
                ),
            )
            st.plotly_chart(fig_map, width="stretch")

    with cm_right:
        with st.container(border=True):
            df_rad = tables["radar"]
            radar_dims = [
                ("Demanda_Dia_Habil_norm", "Demanda Hábil"),
                ("Concentracion_Pico_AM_norm", "% Pico AM"),
                ("Concentracion_Pico_PM_norm", "% Pico PM"),
                ("Resiliencia_Fin_Semana_norm", "Dom/Festivo"),
                ("Capacidad_Instalada_norm", "Capacidad"),
                ("Indice_Saturacion_Pico_norm", "Saturación"),
            ]
            theta_labels = [lbl for _, lbl in radar_dims] + [radar_dims[0][1]]

            fig_radar = go.Figure()
            for _, r_row in df_rad.iterrows():
                r_vals = [float(r_row[col_n]) for col_n, _ in radar_dims]
                r_vals.append(r_vals[0])
                etq = str(r_row["Etiqueta_Cluster"])
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=r_vals,
                        theta=theta_labels,
                        fill="toself",
                        name=etq.split(" - ")[0],
                        line=dict(color=CLUSTER_COLORS.get(etq, TM_DARK), width=2.3),
                        opacity=0.72,
                    )
                )
            fig_radar.update_layout(
                template="plotly_white",
                polar=dict(radialaxis=dict(visible=True, range=[0, 1.05])),
                title=dict(text="Perfil Radar Normalizado (0–1) por Clúster", x=0.0, font=dict(size=14, color=TM_DARK)),
                height=490,
                margin=dict(l=48, r=48, t=58, b=68),
                legend=dict(title_text="", orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_radar, width="stretch")

    if len(df_cl_map) > 0:
        top_sat = df_cl_map.sort_values("indice_saturacion_pico", ascending=False).iloc[0]
        dom_cluster = df_cl_map["Etiqueta_Cluster"].value_counts().idxmax()
        st.markdown(
            f"""
            <div class="insight-bar" style="margin-bottom:0.8rem;">
                🧭 <b>Diagnóstico Espacial ({sel_troncal}):</b> Segmento predominante: <b>{dom_cluster}</b>.
                Mayor presión física en hora pico: <b>{top_sat['Nombre_Estacion']}</b> (índice de saturación <code>{top_sat['indice_saturacion_pico']:.2f}</code> val/plaza).
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(r"### 📐 Validación Técnica del Número Óptimo de Clústeres ($k = 2 \dots 8$) y Proyección PCA")
    cv1, cv2 = st.columns([1.02, 1.0])
    with cv1:
        with st.container(border=True):
            df_codo = tables["codo_sil"]
            fig_k = make_subplots(specs=[[{"secondary_y": True}]])
            fig_k.add_trace(
                go.Scatter(
                    x=df_codo["k"],
                    y=df_codo["inercia"],
                    mode="lines+markers",
                    name="Inercia (Codo)",
                    line=dict(color=TM_DARK, width=2.5),
                    marker=dict(size=8),
                ),
                secondary_y=False,
            )
            fig_k.add_trace(
                go.Scatter(
                    x=df_codo["k"],
                    y=df_codo["silhouette_score"],
                    mode="lines+markers",
                    name="Silhouette Score",
                    line=dict(color=TM_RED, width=2.5),
                    marker=dict(size=9),
                ),
                secondary_y=True,
            )
            fig_k.add_vline(
                x=4,
                line_dash="dash",
                line_color=TM_GREEN,
                annotation_text=" k=4 óptimo ",
                annotation_position="top right",
            )
            fig_k.update_xaxes(title_text="Número de Clústeres (k)", dtick=1)
            fig_k.update_yaxes(title_text="Inercia Intra-Clúster", secondary_y=False)
            fig_k.update_yaxes(title_text="Silhouette Score", secondary_y=True)
            fig_k.update_layout(title="Método del Codo (Inercia) + Coeficiente de Silhouette")
            st.plotly_chart(apply_plotly_style(fig_k, height=370, show_legend=True), width="stretch")

    with cv2:
        with st.container(border=True):
            pca_exp = rep_mod["clustering_estaciones"]["pca_varianza_explicada_2d_pct"]
            df_cl_pca = df_cl.copy()
            df_cl_pca["Cluster_Corto"] = df_cl_pca["Etiqueta_Cluster"].str.split(" - ").str[0]
            short_colors = {k.split(" - ")[0]: v for k, v in CLUSTER_COLORS.items()}
            fig_pca = px.scatter(
                df_cl_pca,
                x="pca_1",
                y="pca_2",
                color="Cluster_Corto",
                size="prom_diario_habil",
                hover_name="Nombre_Estacion",
                hover_data={"Etiqueta_Cluster": True, "Troncal": True},
                color_discrete_map=short_colors,
                title=f"Separación en Espacio PCA 2D ({sum(pca_exp):.1f}% Varianza Explicada)",
                labels={"pca_1": f"PC1 ({pca_exp[0]:.1f}%)", "pca_2": f"PC2 ({pca_exp[1]:.1f}%)"},
            )
            st.plotly_chart(apply_plotly_style(fig_pca, height=370, show_legend=True), width="stretch")

    with st.container(border=True):
        st.markdown("**Resumen Descriptivo de los 4 Segmentos Estratégicos de Estaciones:**")
        seg_df = pd.DataFrame(rep_mod["clustering_estaciones"]["perfiles_segmentos"])
        st.dataframe(seg_df, width="stretch", hide_index=True)


# ==============================================================================
# PESTAÑA 5: MODELO PREDICTIVO XGBOOST Y SIMULADOR DE 3 ESCENARIOS (NOV 2026)
# ==============================================================================
with tab5:
    render_active_status_bar(sel_troncal, sel_estacion, sel_tipos_dia, sel_franjas, escenario_label)
    st.subheader("🤖 Evaluación Predictiva Out-of-Sample (Train: Ene–Ago → Test: Sep–Oct 2026)")

    m_hora = rep_mod["validacion_predictiva_sep_oct_2026"]["metricas_hora_estacion"]
    m_dia = rep_mod["validacion_predictiva_sep_oct_2026"]["metricas_diario_sistema"]

    with st.container(border=True):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(
            "R² XGBoost (Hora–Estación)",
            f"{m_hora['XGBoost_Regressor']['R2']:.4f}",
            f"+{m_hora['XGBoost_Regressor']['R2'] - m_hora['Baseline_Promedio_Historico']['R2']:.4f} vs Baseline",
        )
        mc2.metric(
            "WAPE Hora–Estación",
            f"{m_hora['XGBoost_Regressor']['WAPE_pct']:.2f}%",
            f"-{m_hora['Baseline_Promedio_Historico']['WAPE_pct'] - m_hora['XGBoost_Regressor']['WAPE_pct']:.2f} pp vs Baseline",
            delta_color="inverse",
        )
        mc3.metric(
            "R² XGBoost (Diario Sistema)",
            f"{m_dia['XGBoost_Regressor']['R2']:.4f}",
            f"vs {m_dia['Baseline_Promedio_Historico']['R2']:.4f} Baseline Histórico",
        )
        mc4.metric(
            "MAE Diario Sistema",
            f"{m_dia['XGBoost_Regressor']['MAE']:,.0f} val",
            "-61.2% error vs Baseline Histórico",
            delta_color="inverse",
        )

    col_bt_tbl, col_fi = st.columns([1.05, 1.0])
    with col_bt_tbl:
        with st.container(border=True):
            st.markdown("**Comparativa Formal de Modelos en Test Cronológico (Sep–Oct 2026):**")
            comp_rows = []
            for nivel_lbl, m_dict in [("Hora–Estación (169,457 obs.)", m_hora), ("Diario–Sistema (61 días)", m_dia)]:
                for mod_name, vals in m_dict.items():
                    comp_rows.append({"Granularidad": nivel_lbl, "Modelo": mod_name, **vals})
            st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True, height=255)

            bt_franja_sel = st.radio(
                "Filtrar serie de Backtest (Sep–Oct 2026) por Franja:",
                options=["Todas (Sistema Diario)", "Pico_AM", "Pico_PM", "Valle"],
                horizontal=True,
                key="bt_franja_radio",
            )

    with col_fi:
        with st.container(border=True):
            df_fi = tables["feat_imp"].head(12).copy()
            fig_fi = px.bar(
                df_fi,
                x="importancia_pct",
                y="variable",
                orientation="h",
                text="importancia_pct",
                color_discrete_sequence=[TM_RED],
                title="Top 12 Variables más Influyentes en XGBRegressor (% Ganancia)",
                labels={"importancia_pct": "Importancia (%)", "variable": ""},
            )
            fig_fi.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
            fig_fi.update_layout(yaxis=dict(autorange="reversed"), xaxis_range=[0, 96])
            st.plotly_chart(apply_plotly_style(fig_fi, height=345, show_legend=False, right_margin=40), width="stretch")

    # Curva Backtest Real vs Predicho
    with st.container(border=True):
        df_bt = tables["backtest"]
        if bt_franja_sel.startswith("Todas"):
            bt_sub = df_bt[df_bt["Nivel"] == "Diario_Sistema"].sort_values("Fecha")
        else:
            bt_sub = df_bt[(df_bt["Nivel"] == "Diario_Franja") & (df_bt["Franja_Horaria"] == bt_franja_sel)].sort_values("Fecha")

        fig_bt = go.Figure()
        fig_bt.add_trace(
            go.Scatter(
                x=bt_sub["Fecha"],
                y=bt_sub["Real_Validaciones"],
                mode="lines+markers",
                name="Demanda Real Observada",
                line=dict(color=TM_DARK, width=2.5),
                marker=dict(size=5),
            )
        )
        fig_bt.add_trace(
            go.Scatter(
                x=bt_sub["Fecha"],
                y=bt_sub["Pred_XGBoost"],
                mode="lines",
                name="Predicción XGBoost (Sin Leakage)",
                line=dict(color=TM_RED, width=2.3, dash="solid"),
            )
        )
        fig_bt.add_trace(
            go.Scatter(
                x=bt_sub["Fecha"],
                y=bt_sub["Pred_Baseline_Hist"],
                mode="lines",
                name="Baseline Promedio Histórico",
                line=dict(color="#94A3B8", width=1.8, dash="dot"),
            )
        )
        fig_bt.update_layout(
            title=f"Backtest Cronológico Out-of-Sample (2026-09-01 a 2026-10-31) — Franja: {bt_franja_sel}",
            xaxis_title="Fecha de Validación",
            yaxis_title="Validaciones",
        )
        st.plotly_chart(apply_plotly_style(fig_bt, height=355, show_legend=True), width="stretch")

    st.subheader(f"🔮 Simulador de Proyecciones a 30 Días — Noviembre 2026 ({escenario_label})")

    with st.container(border=True):
        df_proy_d = tables["proy_diaria"]
        if sel_troncal == "Todas las Troncales":
            proy_nov_sub = df_proy_d[df_proy_d["Nivel"] == "Sistema_Total"].sort_values("Fecha")
            hist_oct_tail = (
                df_minable[df_minable["Fecha"] >= "2026-10-11"]
                .groupby("Fecha", as_index=False)["Total_Validaciones"]
                .sum()
                .sort_values("Fecha")
            )
            titulo_alcance_proy = "Sistema Troncal Completo"
        else:
            proy_nov_sub = df_proy_d[
                (df_proy_d["Nivel"] == "Troncal") & (df_proy_d["Troncal"] == sel_troncal)
            ].sort_values("Fecha")
            hist_oct_tail = (
                df_minable[(df_minable["Fecha"] >= "2026-10-11") & (df_minable["Troncal"] == sel_troncal)]
                .groupby("Fecha", as_index=False)["Total_Validaciones"]
                .sum()
                .sort_values("Fecha")
            )
            titulo_alcance_proy = f"Troncal: {sel_troncal}"

        fig_sim = go.Figure()

        if len(hist_oct_tail) > 0 and len(proy_nov_sub) > 0:
            fig_sim.add_trace(
                go.Scatter(
                    x=hist_oct_tail["Fecha"].tolist() + [proy_nov_sub["Fecha"].iloc[0]],
                    y=hist_oct_tail["Total_Validaciones"].tolist() + [proy_nov_sub["Pred_Base"].iloc[0]],
                    mode="lines+markers",
                    name="Histórico Reciente (Oct 11–31)",
                    line=dict(color=TM_DARK, width=2.5),
                    marker=dict(size=5),
                )
            )

        fig_sim.add_trace(
            go.Scatter(
                x=proy_nov_sub["Fecha"],
                y=proy_nov_sub["Pred_Optimista"],
                mode="lines",
                name="Optimista (Q90 + Control Evasión)",
                line=dict(color=TM_GREEN, width=2.1, dash="dash"),
            )
        )
        fig_sim.add_trace(
            go.Scatter(
                x=proy_nov_sub["Fecha"],
                y=proy_nov_sub["Pred_Pesimista"],
                mode="lines",
                name="Pesimista (Q10 + Obras/Choques)",
                fill="tonexty",
                fillcolor="rgba(5, 150, 105, 0.10)",
                line=dict(color=TM_AMBER, width=2.1, dash="dash"),
            )
        )
        fig_sim.add_trace(
            go.Scatter(
                x=proy_nov_sub["Fecha"],
                y=proy_nov_sub["Pred_Base"],
                mode="lines+markers",
                name="Base (Central L2)",
                line=dict(color=TM_RED, width=3.0),
                marker=dict(size=6),
            )
        )

        # Anotaciones escalonadas dentro del área de trazado para cero solapamiento con el título
        for fest_date, fest_lbl, y_pos in [
            ("2026-11-02", "Festivo: Todos los Santos", 0.94),
            ("2026-11-16", "Festivo: Indep. Cartagena", 0.84),
        ]:
            fig_sim.add_vline(x=fest_date, line_dash="dot", line_color="#7C3AED")
            fig_sim.add_annotation(
                x=fest_date,
                y=y_pos,
                yref="paper",
                text=fest_lbl,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#DDD6FE",
                borderwidth=1,
                font=dict(size=10, color="#6D28D9"),
            )

        fig_sim.update_layout(
            title=f"Transición Octubre → Proyección 3 Escenarios Noviembre 2026 ({titulo_alcance_proy})",
            xaxis_title="Fecha",
            yaxis_title="Validaciones Diarias",
        )
        st.plotly_chart(apply_plotly_style(fig_sim, height=410, show_legend=True, bottom_margin=78), width="stretch")

        if len(proy_nov_sub) > 0:
            sum_pes_s = int(proy_nov_sub["Pred_Pesimista"].sum())
            sum_base_s = int(proy_nov_sub["Pred_Base"].sum())
            sum_opt_s = int(proy_nov_sub["Pred_Optimista"].sum())
            peak_nov = proy_nov_sub.loc[proy_nov_sub[esc_col_dia].idxmax()]
            st.markdown(
                f"""
                <div class="insight-bar">
                    🎯 <b>Síntesis de Proyección ({titulo_alcance_proy}):</b> En el escenario activo se proyectan
                    <b>{int(proy_nov_sub[esc_col_dia].sum()):,} validaciones</b> en noviembre 2026
                    (rango <code>{sum_pes_s:,}</code> Pesimista a <code>{sum_opt_s:,}</code> Optimista; Base <code>{sum_base_s:,}</code>),
                    con pico esperado el <b>{peak_nov['Fecha']}</b> (<code>{int(peak_nov[esc_col_dia]):,} val/día</code>).
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Tarjetas de los 5 KPIs sensibilizados según el escenario seleccionado
    hdr_sim_col, pop_sim_col = st.columns([3.2, 1.0], vertical_alignment="center")
    with hdr_sim_col:
        st.markdown(f"#### 🎯 Los 5 KPIs Corporativos Sensibilizados — **{escenario_label}**")
    with pop_sim_col:
        render_kpi_methodology_popover("pop_kpis_tab5")

    esc_kpi_data = kpis_json["escenarios"][esc_json_key]

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        kd1 = esc_kpi_data["KPI_1_Demanda_Validada_Troncal"]
        st.markdown(
            render_kpi_card(
                "1. Demanda Proyectada / Día",
                f"{kd1['promedio_diario_global']:,}",
                f"Total 30d: {kd1['total_mensual_30d']:,} | Hábil: {kd1['promedio_dia_habil']:,}",
                es_observado=True,
                tooltip_formula="Proyección diaria agregada del sistema para noviembre 2026 con XGBoost.",
            ),
            unsafe_allow_html=True,
        )
    with s2:
        kd2 = esc_kpi_data["KPI_2_Indice_Cumplimiento_Programacion_ICP_pct"]
        st.markdown(
            render_kpi_card(
                "2. ICP Proyectado",
                f"{kd2['valor_pct']:.2f}%",
                "Meta operativa de cumplimiento de flota",
                es_observado=False,
                tooltip_formula="93.50% Pesimista | 96.40% Base | 97.80% Optimista.",
            ),
            unsafe_allow_html=True,
        )
    with s3:
        kd3 = esc_kpi_data["KPI_3_Tiempo_Medio_Espera_Troncal_min"]
        st.markdown(
            render_kpi_card(
                "3. Espera Proyectada",
                f"{kd3['ponderado_sistema_min']:.2f} min",
                f"Pico: {kd3['hora_pico_min']:.2f} min | Valle: {kd3['hora_valle_min']:.2f} min",
                es_observado=False,
                tooltip_formula="Sensibilizado según índice de saturación proyectado y cumplimiento ICP.",
            ),
            unsafe_allow_html=True,
        )
    with s4:
        kd4 = esc_kpi_data["KPI_4_Tasa_Evasion_Focalizada_pct"]
        st.markdown(
            render_kpi_card(
                "4. Evasión Proyectada",
                f"{kd4['valor_pct']:.2f}%",
                "Sensibilizada por control de accesos y torniquetes",
                es_observado=False,
                tooltip_formula="19.00% Pesimista | 15.32% Base | 12.50% Optimista.",
            ),
            unsafe_allow_html=True,
        )
    with s5:
        kd5 = esc_kpi_data["KPI_5_Kilometros_Comerciales_Troncal_millones_mes"]
        st.markdown(
            render_kpi_card(
                "5. Km Comerciales Nov",
                f"{kd5['millones_km_mes']:.3f} M km",
                f"{kd5['km_exactos_mes']:,} km ajustados por ICP",
                es_observado=False,
                tooltip_formula="Ajuste proporcional a la oferta ejecutada (11.30M * ICP_esc / 96.4%).",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    col_ktbl, col_est_proy = st.columns([1.0, 1.15])
    with col_ktbl:
        with st.container(border=True):
            st.markdown("**Matriz Comparativa Completa de los 5 KPIs en los 3 Escenarios:**")
            kpis_piv = (
                tables["kpis_csv"]
                .pivot(index=["KPI_ID", "Nombre_KPI", "Fuente_Dato", "Unidad"], columns="Escenario", values="Valor")
                .reset_index()
            )
            ordered_sc_cols = [
                "KPI_ID",
                "Nombre_KPI",
                "Fuente_Dato",
                "Unidad",
                "Linea_Base_Historica_2026",
                "Escenario_Pesimista_Nov2026",
                "Escenario_Base_Nov2026",
                "Escenario_Optimista_Nov2026",
            ]
            st.dataframe(kpis_piv[ordered_sc_cols], width="stretch", hide_index=True, height=255)

    with col_est_proy:
        with st.container(border=True):
            st.markdown(f"**Ranking de Estaciones Proyectadas en Noviembre 2026 ({titulo_alcance_proy}):**")
            df_pe = tables["proy_estacion"].copy()
            if sel_troncal != "Todas las Troncales":
                df_pe = df_pe[df_pe["Troncal"] == sel_troncal]
            df_pe = df_pe.sort_values(esc_col_tot_est, ascending=False)
            show_pe_cols = list(
                dict.fromkeys(
                    [
                        "Nombre_Estacion",
                        "Troncal",
                        "Etiqueta_Cluster",
                        esc_col_prom_est,
                        "Total_Nov_Pesimista",
                        "Total_Nov_Base",
                        "Total_Nov_Optimista",
                        "Indice_Saturacion_Nov_Base",
                    ]
                )
            )
            st.dataframe(df_pe[show_pe_cols], width="stretch", hide_index=True, height=255)
