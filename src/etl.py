"""
Fase 2: Pipeline ETL, Limpieza, Imputación, Feature Engineering y Escalamiento
Proyecto de Minería de Datos - TransMilenio S.A.

Entradas (data/raw/):
  - Estaciones_Troncales_de_TRANSMILENIO.csv
  - 01..10 TM Resumen de Validaciones Troncales ... Intervalo 15 Mint.xlsx (2026)

Salidas:
  - data/processed/vista_minable.parquet          (Principal Hora-Estación para Python/ML/Streamlit)
  - data/processed/vista_minable.csv              (Principal Hora-Estación en CSV)
  - data/processed/vista_diaria_estacion.csv      (Agregada Día-Estación para Power BI / pruebas)
  - data/processed/perfil_estaciones.csv          (Perfil operativo + físico por estación sin escalar)
  - data/processed/perfil_estaciones_escalado.csv (Perfil escalado con StandardScaler para Clustering)
  - data/processed/reporte_etl.json               (Reporte de auditoría completa del ETL)
  - models/scaler.joblib                          (Objeto StandardScaler entrenado sobre perfil de estación)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import openpyxl
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CONFIGURACIÓN Y CONSTANTES DE DOMINIO TRANSMILENIO 2026
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Tabla oficial de homologación de códigos FCS Center (.xlsx) -> cod_nodo ArcGIS (.csv)
HOMOLOGACION_COD_NODO: dict[int, int] = {
    15001: 5011,  # Tibanica - Primavera (Extensión Av. Ciudad de Cali TZ022)
    15002: 5010,  # Los Laureles (Extensión Av. Ciudad de Cali TZ022)
    15003: 5009,  # Islandia (Extensión Av. Ciudad de Cali TZ022)
    57503: 7503,  # Ampliación San Mateo -> San Mateo - CC Unisur
    4004: 4100,   # La Granja -> Granja - Kr 77
    50003: 9001,  # Corral Molinos -> Molinos
}

# Catálogo legible de troncales a partir de id_trazado del maestro GIS
MAPA_TRONCALES: dict[str, str] = {
    "TZ001": "A - Caracas",
    "TZ002": "B - AutoNorte",
    "TZ003": "C - Suba",
    "TZ005": "D - Calle 80",
    "TZ007": "E - NQS Central (Calle 6)",
    "TZ008": "E - NQS Central",
    "TZ009": "F - Américas",
    "TZ010": "G - NQS Sur",
    "TZ011": "G - Extensión Soacha",
    "TZ012": "H - Caracas Sur",
    "TZ013": "H - Caracas Sur (Usme)",
    "TZ014": "H - Ramal Tunal",
    "TZ015": "J - Eje Ambiental",
    "TZ016": "K - Calle 26",
    "TZ018": "L - Carrera 10",
    "TZ019": "M - Carrera 7",
    "TZ022": "F - Av. Ciudad de Cali",
}

# Festivos oficiales de Colombia en 2026 (Ley 51 de 1983 - Ley Emiliani)
FESTIVOS_COLOMBIA_2026: set[str] = {
    "2026-01-01",  # Año Nuevo
    "2026-01-12",  # Reyes Magos (trasladado del 6 ene)
    "2026-03-23",  # Día de San José (trasladado del 19 mar)
    "2026-04-02",  # Jueves Santo
    "2026-04-03",  # Viernes Santo
    "2026-05-01",  # Día del Trabajo
    "2026-05-18",  # Ascensión del Señor
    "2026-06-08",  # Corpus Christi
    "2026-06-15",  # Sagrado Corazón de Jesús
    "2026-06-29",  # San Pedro y San Pablo
    "2026-07-20",  # Día de la Independencia
    "2026-08-07",  # Batalla de Boyacá
    "2026-08-17",  # Asunción de la Virgen (trasladado del 15 ago)
    "2026-10-12",  # Día de la Raza
    "2026-11-02",  # Todos los Santos (trasladado del 1 nov)
    "2026-11-16",  # Independencia de Cartagena (trasladado del 11 nov)
    "2026-12-08",  # Inmaculada Concepción
    "2026-12-25",  # Navidad
}

# Columnas sin valor analítico a descartar del maestro de estaciones
COLS_DESCARTAR_MAESTRO: list[str] = [
    "esta_oper",
    "observ",
    "created_user",
    "created_date",
    "last_edited_user",
    "last_edited_date",
    "objectid",
    "globalid",
    "X",
    "Y",
    "num_est",
]

# Columnas físicas/operativas donde el valor 0 representa un falso cero estructural
COLS_CEROS_ESTRUCTURALES: list[str] = [
    "ancho_est",
    "area_est",
    "cap_biart",
    "cap_art",
    "num_vag",
]


# ==============================================================================
# 1. LIMPIEZA E IMPUTACIÓN DEL MAESTRO DE ESTACIONES
# ==============================================================================
def clean_and_impute_stations(csv_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Limpia el maestro de estaciones troncales:
      - Elimina columnas administrativas/constantes sin valor analítico.
      - Desambigua nombres homónimos ('Temporal AV. Jiménez - Inter Eléctricas').
      - Convierte falsos ceros estructurales en NaN e imputa con la mediana por 'tipo_esta'
        (con respaldo en mediana global cuando un tipo_esta completo carece de dato, ej. Portales tipo_esta=1).
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    raw_shape = df.shape

    cols_dropped = [c for c in COLS_DESCARTAR_MAESTRO if c in df.columns]
    df = df.drop(columns=cols_dropped)

    df["cod_nodo"] = df["cod_nodo"].astype("int64")
    df["Troncal"] = df["id_trazado"].map(MAPA_TRONCALES).fillna(df["id_trazado"])

    # Desambiguar estaciones homónimas en distintas troncales para evitar colisiones visuales
    dup_name_mask = df["nom_est"].duplicated(keep=False)
    homonimos_corregidos = int(dup_name_mask.sum())
    if homonimos_corregidos > 0:
        df.loc[dup_name_mask, "nom_est"] = (
            df.loc[dup_name_mask, "nom_est"] + " (" + df.loc[dup_name_mask, "Troncal"] + ")"
        )

    # Conversión de falsos ceros estructurales a NaN e imputación por mediana según tipo_esta
    imputation_audit: dict[str, Any] = {}
    for col in COLS_CEROS_ESTRUCTURALES:
        n_zeros = int((df[col] == 0).sum())
        df[col] = pd.to_numeric(df[col], errors="coerce").mask(df[col] == 0, np.nan)

        group_medians = df.groupby("tipo_esta")[col].transform("median")
        global_median = float(df[col].median())

        n_fallback_global = int((df[col].isna() & group_medians.isna()).sum())
        df[col] = df[col].fillna(group_medians).fillna(global_median)

        imputation_audit[col] = {
            "ceros_estructurales_detectados": n_zeros,
            "imputados_por_mediana_tipo_esta": n_zeros - n_fallback_global,
            "imputados_por_mediana_global_portales": n_fallback_global,
            "mediana_global_referencia": round(global_median, 2),
            "nulos_restantes": int(df[col].isna().sum()),
        }

    # Estandarizar tipos enteros/flotantes de infraestructura
    for int_col in ["eta_oper", "tipo_esta", "num_acc", "acc_esp", "acc_depr", "acc_puent", "num_vag", "cap_biart", "cap_art"]:
        df[int_col] = df[int_col].round().astype("int64")
    for flt_col in ["longitud", "latitud", "long_est", "ancho_est", "area_est"]:
        df[flt_col] = df[flt_col].round(6)

    df = df.rename(columns={"cod_nodo": "ID_Estacion", "nom_est": "Nombre_Estacion"})

    ordered_cols = [
        "ID_Estacion",
        "Nombre_Estacion",
        "Troncal",
        "id_trazado",
        "tipo_esta",
        "eta_oper",
        "ub_est",
        "longitud",
        "latitud",
        "long_est",
        "ancho_est",
        "area_est",
        "num_acc",
        "acc_esp",
        "acc_depr",
        "acc_puent",
        "num_vag",
        "cap_art",
        "cap_biart",
    ]
    df = df[ordered_cols].sort_values("ID_Estacion").reset_index(drop=True)

    audit_stations = {
        "dimensiones_crudas": {"filas": raw_shape[0], "columnas": raw_shape[1]},
        "dimensiones_limpias": {"filas": df.shape[0], "columnas": df.shape[1]},
        "columnas_descartadas": cols_dropped,
        "homonimos_desambiguados": homonimos_corregidos,
        "imputacion_ceros_estructurales": imputation_audit,
        "nulos_totales_maestro_limpio": int(df.isna().sum().sum()),
    }
    return df, audit_stations


# ==============================================================================
# 2. INGESTA Y LIMPIEZA ESTRUCTURAL POR ARCHIVO MENSUAL (.XLSX)
# ==============================================================================
def _parse_hour_from_interval(val: Any) -> int | None:
    """Extrae la hora entera (0..23) desde un objeto datetime.time, datetime o texto HH:MM."""
    if val is None:
        return None
    if isinstance(val, (dt.time, dt.datetime)):
        return int(val.hour)
    s = str(val).strip()
    if ":" in s:
        try:
            return int(s.split(":")[0])
        except ValueError:
            return None
    return None


def process_single_xlsx(args: tuple[str, set[int]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Procesa un archivo mensual .xlsx (hoja 'Validaciones Tullave', encabezado en fila índice 6):
      1. Ignora hoja 'Validacion Datos', columna A vacía, filas vacías de plantilla y 'Total general'.
      2. Descarta columnas de fecha cuyo año-mes no coincida con el mes del archivo.
      3. Extrae código entre paréntesis de 'Estación', aplica homologación de 6 códigos y filtra
         estrictamente estaciones troncales (registrando exclusiones de TransMiCable y Dual en calle).
      4. Filtra registros residuales fuera del horario comercial troncal (madrugada 00:00-03:59).
      5. Agrupa a nivel (ID_Estacion, Hora_Bloque) y despivota (melt) a formato largo Hora-Estación.
    """
    filepath_str, valid_nodos = args
    filepath = Path(filepath_str)

    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ignored_sheets = [s for s in wb.sheetnames if s != "Validaciones Tullave"]
    ws = wb["Validaciones Tullave"]

    # Leer fila de encabezado (índice 6 -> fila 7 en Excel)
    header_row = next(ws.iter_rows(min_row=7, max_row=7, values_only=True))

    # Identificar todas las columnas de fecha y el año-mes objetivo del archivo (primera columna de fecha)
    all_date_positions: list[tuple[int, str]] = []
    total_general_col_present = False
    for col_idx, cell_val in enumerate(header_row):
        if isinstance(cell_val, (dt.datetime, dt.date)):
            all_date_positions.append((col_idx, cell_val.strftime("%Y-%m-%d")))
        elif isinstance(cell_val, str) and cell_val.strip().lower() == "total general":
            total_general_col_present = True

    target_year_month = all_date_positions[0][1][:7]
    valid_date_positions = [(idx, d_str) for idx, d_str in all_date_positions if d_str.startswith(target_year_month)]
    overflow_dates_removed = [d_str for _, d_str in all_date_positions if not d_str.startswith(target_year_month)]

    valid_col_indices = [idx for idx, _ in valid_date_positions]
    valid_date_names = [d_str for _, d_str in valid_date_positions]
    n_valid_days = len(valid_date_names)

    # Contadores de auditoría
    raw_rows_after_header = 0
    empty_template_rows = 0
    total_general_rows = 0
    operational_rows = 0

    excl_cable_rows = 0
    excl_cable_validations = 0.0
    excl_dual_street_rows = 0
    excl_dual_street_validations = 0.0
    unmatched_other_codes: dict[str, float] = {}

    homologated_rows = 0
    homologated_validations = 0.0
    dual_in_troncal_rows = 0
    dual_in_troncal_validations = 0.0

    off_hours_00_03_rows = 0
    off_hours_00_03_validations = 0.0
    in_scope_commercial_rows = 0
    null_cells_in_valid_dates = 0

    # Acumulador rápido en memoria: (id_estacion, hora_bloque) -> np.ndarray de longitud n_valid_days
    agg_matrix: dict[tuple[int, int], np.ndarray] = {}
    code_regex = re.compile(r"^\((\d+)\)")

    for row in ws.iter_rows(min_row=8, values_only=True):
        raw_rows_after_header += 1
        fase_raw = row[1]
        est_raw = row[3]

        if fase_raw is None and est_raw is None:
            empty_template_rows += 1
            continue

        fase_str = str(fase_raw).strip() if fase_raw is not None else ""
        if fase_str.lower() == "total general":
            total_general_rows += 1
            continue

        operational_rows += 1

        # Extraer vector diario únicamente de los días válidos del mes
        day_vals = np.empty(n_valid_days, dtype=np.float64)
        for j, c_idx in enumerate(valid_col_indices):
            v = row[c_idx]
            if v is None:
                null_cells_in_valid_dates += 1
                day_vals[j] = 0.0
            else:
                day_vals[j] = float(v)
        row_month_sum = float(day_vals.sum())

        # Extraer código entre paréntesis y aplicar homologación oficial
        est_text = str(est_raw).strip() if est_raw is not None else ""
        m = code_regex.match(est_text)
        cod_extracted = int(m.group(1)) if m else -1
        cod_final = HOMOLOGACION_COD_NODO.get(cod_extracted, cod_extracted)
        was_homologated = cod_extracted in HOMOLOGACION_COD_NODO

        # Filtro estricto de alcance: Red de Estaciones Troncales
        if cod_final not in valid_nodos:
            if fase_str.upper() == "DUAL":
                excl_dual_street_rows += 1
                excl_dual_street_validations += row_month_sum
            elif (40000 <= cod_extracted <= 40999) or cod_extracted == 8100 or "cable" in est_text.lower():
                excl_cable_rows += 1
                excl_cable_validations += row_month_sum
            else:
                unmatched_other_codes[est_text] = unmatched_other_codes.get(est_text, 0.0) + row_month_sum
            continue

        if was_homologated:
            homologated_rows += 1
            homologated_validations += row_month_sum
        if fase_str.upper() == "DUAL":
            dual_in_troncal_rows += 1
            dual_in_troncal_validations += row_month_sum

        # Filtrar madrugada fuera del horario comercial troncal (00:00 - 03:59)
        hora = _parse_hour_from_interval(row[5])
        if hora is None or hora < 4 or hora > 23:
            off_hours_00_03_rows += 1
            off_hours_00_03_validations += row_month_sum
            continue

        in_scope_commercial_rows += 1
        key = (cod_final, hora)
        if key in agg_matrix:
            agg_matrix[key] += day_vals
        else:
            agg_matrix[key] = day_vals

    wb.close()

    # Construir DataFrame ancho agregado (ID_Estacion, Hora_Bloque) y despivotar (melt) a formato largo
    keys_list = sorted(agg_matrix.keys())
    matrix_2d = np.vstack([agg_matrix[k] for k in keys_list]) if keys_list else np.empty((0, n_valid_days))

    df_wide = pd.DataFrame(matrix_2d, columns=valid_date_names)
    df_wide.insert(0, "Hora_Bloque", [k[1] for k in keys_list])
    df_wide.insert(0, "ID_Estacion", [k[0] for k in keys_list])

    df_long = df_wide.melt(
        id_vars=["ID_Estacion", "Hora_Bloque"],
        value_vars=valid_date_names,
        var_name="Fecha",
        value_name="Total_Validaciones",
    )
    df_long["Total_Validaciones"] = df_long["Total_Validaciones"].round().astype("int64")
    df_long["ID_Estacion"] = df_long["ID_Estacion"].astype("int64")
    df_long["Hora_Bloque"] = df_long["Hora_Bloque"].astype("int64")

    file_audit = {
        "archivo": filepath.name,
        "mes_objetivo": target_year_month,
        "hojas_ignoradas": ignored_sheets,
        "columna_A_vacia_ignorada": header_row[0] is None,
        "columna_total_general_ignorada": total_general_col_present,
        "dias_validos_conservados": n_valid_days,
        "fechas_desbordadas_eliminadas": overflow_dates_removed,
        "filas_crudas_bajo_header": raw_rows_after_header,
        "filas_plantilla_vacias_eliminadas": empty_template_rows,
        "filas_total_general_eliminadas": total_general_rows,
        "filas_operativas_totales": operational_rows,
        "filas_excluidas_transmicable": excl_cable_rows,
        "validaciones_excluidas_transmicable": int(round(excl_cable_validations)),
        "filas_excluidas_dual_calle": excl_dual_street_rows,
        "validaciones_excluidas_dual_calle": int(round(excl_dual_street_validations)),
        "codigos_no_mapeados_residuales": unmatched_other_codes,
        "filas_homologadas_6_codigos": homologated_rows,
        "validaciones_recuperadas_homologacion": int(round(homologated_validations)),
        "filas_dual_en_estacion_troncal_incluidas": dual_in_troncal_rows,
        "validaciones_dual_en_estacion_troncal_incluidas": int(round(dual_in_troncal_validations)),
        "filas_madrugada_00_03_filtradas": off_hours_00_03_rows,
        "validaciones_madrugada_00_03_filtradas": int(round(off_hours_00_03_validations)),
        "filas_15min_troncal_horario_comercial": in_scope_commercial_rows,
        "celdas_nulas_en_fechas_validas": null_cells_in_valid_dates,
        "filas_hora_estacion_generadas": int(len(df_long)),
        "validaciones_limpias_mes": int(df_long["Total_Validaciones"].sum()),
    }
    return df_long, file_audit


# ==============================================================================
# 3. CONSOLIDACIÓN, JOIN EXACTO Y FEATURE ENGINEERING
# ==============================================================================
def build_minable_views(
    monthly_dfs: list[pd.DataFrame],
    df_stations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Une los 10 meses a nivel Hora-Estación, crea variables temporales/operativas
    y ejecuta el JOIN exacto por ID_Estacion (cod_nodo int64) contra el maestro limpio.
    """
    df_val = pd.concat(monthly_dfs, ignore_index=True)

    # Garantizar unicidad (Fecha, Hora_Bloque, ID_Estacion) por seguridad ante cualquier solapamiento
    df_val = (
        df_val.groupby(["Fecha", "Hora_Bloque", "ID_Estacion"], as_index=False)["Total_Validaciones"]
        .sum()
    )

    # Feature Engineering temporal vectorizado sobre fechas únicas (304 días -> instantáneo)
    unique_dates = pd.DataFrame({"Fecha": df_val["Fecha"].unique()})
    dt_series = pd.to_datetime(unique_dates["Fecha"], format="%Y-%m-%d")
    unique_dates["Dia_Semana"] = dt_series.dt.day_of_week.astype("int64")  # 0=Lunes .. 6=Domingo
    unique_dates["Mes"] = dt_series.dt.month.astype("int64")
    is_holiday = unique_dates["Fecha"].isin(FESTIVOS_COLOMBIA_2026)

    unique_dates["Tipo_Dia"] = np.select(
        [
            is_holiday | (unique_dates["Dia_Semana"] == 6),
            unique_dates["Dia_Semana"] == 5,
        ],
        [
            "Domingo_Festivo",
            "Sabado",
        ],
        default="Habil",
    )

    df_val = df_val.merge(unique_dates, on="Fecha", how="left")

    # Franja_Horaria: Pico_AM (05:00-08:59), Pico_PM (16:00-19:59), Valle (resto del horario comercial)
    df_val["Franja_Horaria"] = np.select(
        [
            df_val["Hora_Bloque"].between(5, 8),
            df_val["Hora_Bloque"].between(16, 19),
        ],
        [
            "Pico_AM",
            "Pico_PM",
        ],
        default="Valle",
    )

    # JOIN exacto 1:N contra maestro limpio de estaciones por ID_Estacion (cod_nodo int64)
    df_minable = df_val.merge(df_stations, on="ID_Estacion", how="left", validate="many_to_one")

    # Orden exacto del esquema del proyecto
    core_cols = [
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
    ]
    infra_cols = [c for c in df_stations.columns if c not in {"ID_Estacion", "Nombre_Estacion", "Troncal"}]
    df_minable = (
        df_minable[core_cols + infra_cols]
        .sort_values(["Fecha", "Hora_Bloque", "ID_Estacion"])
        .reset_index(drop=True)
    )

    # Tabla agregada Día-Estación vectorizada en Cython (ultrarrápida) para Power BI y análisis ágil
    df_diaria = (
        df_minable.assign(
            _val_am=np.where(df_minable["Franja_Horaria"] == "Pico_AM", df_minable["Total_Validaciones"], 0),
            _val_pm=np.where(df_minable["Franja_Horaria"] == "Pico_PM", df_minable["Total_Validaciones"], 0),
            _val_valle=np.where(df_minable["Franja_Horaria"] == "Valle", df_minable["Total_Validaciones"], 0),
        )
        .groupby(
            ["Fecha", "ID_Estacion", "Nombre_Estacion", "Troncal", "Dia_Semana", "Mes", "Tipo_Dia"],
            as_index=False,
        )
        .agg(
            Total_Validaciones_Dia=("Total_Validaciones", "sum"),
            Validaciones_Pico_AM=("_val_am", "sum"),
            Validaciones_Pico_PM=("_val_pm", "sum"),
            Validaciones_Valle=("_val_valle", "sum"),
            Max_Validaciones_Hora=("Total_Validaciones", "max"),
        )
        .merge(df_stations.drop(columns=["Nombre_Estacion", "Troncal"]), on="ID_Estacion", how="left")
        .sort_values(["Fecha", "ID_Estacion"])
        .reset_index(drop=True)
    )

    return df_minable, df_diaria


# ==============================================================================
# 4. PERFILAMIENTO POR ESTACIÓN Y ESCALAMIENTO (StandardScaler)
# ==============================================================================
def build_and_scale_station_profiles(
    df_diaria: pd.DataFrame,
    df_stations: pd.DataFrame,
    scaler_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Construye el perfil operativo y físico de cada estación activa y aplica StandardScaler
    sobre las variables numéricas clave para Clustering / ML, guardando models/scaler.joblib.
    """
    # Promedios diarios por Tipo_Dia
    pivot_tipo_dia = (
        df_diaria.pivot_table(
            index="ID_Estacion",
            columns="Tipo_Dia",
            values="Total_Validaciones_Dia",
            aggfunc="mean",
            fill_value=0.0,
        )
        .rename(
            columns={
                "Habil": "prom_diario_habil",
                "Sabado": "prom_diario_sabado",
                "Domingo_Festivo": "prom_diario_dom_fest",
            }
        )
        .reset_index()
    )

    agg_general = (
        df_diaria.groupby("ID_Estacion", as_index=False)
        .agg(
            total_validaciones_periodo=("Total_Validaciones_Dia", "sum"),
            prom_diario_general=("Total_Validaciones_Dia", "mean"),
            max_validaciones_hora=("Max_Validaciones_Hora", "max"),
            sum_pico_am=("Validaciones_Pico_AM", "sum"),
            sum_pico_pm=("Validaciones_Pico_PM", "sum"),
            sum_valle=("Validaciones_Valle", "sum"),
            dias_activos=("Total_Validaciones_Dia", lambda s: int((s > 0).sum())),
        )
    )

    denom = agg_general["total_validaciones_periodo"].replace(0, np.nan)
    agg_general["pct_pico_am"] = (agg_general["sum_pico_am"] / denom * 100).fillna(0.0)
    agg_general["pct_pico_pm"] = (agg_general["sum_pico_pm"] / denom * 100).fillna(0.0)
    agg_general["pct_valle"] = (agg_general["sum_valle"] / denom * 100).fillna(0.0)
    agg_general = agg_general.drop(columns=["sum_pico_am", "sum_pico_pm", "sum_valle"])

    df_perfil = (
        df_stations.merge(agg_general, on="ID_Estacion", how="inner")
        .merge(pivot_tipo_dia, on="ID_Estacion", how="left")
    )
    df_perfil["ratio_dom_fest_vs_habil"] = (
        df_perfil["prom_diario_dom_fest"] / df_perfil["prom_diario_habil"].replace(0, np.nan)
    ).fillna(0.0)

    # Variables numéricas clave a estandarizar con StandardScaler
    scale_cols = [
        "prom_diario_habil",
        "prom_diario_sabado",
        "prom_diario_dom_fest",
        "max_validaciones_hora",
        "pct_pico_am",
        "pct_pico_pm",
        "pct_valle",
        "ratio_dom_fest_vs_habil",
        "longitud",
        "latitud",
        "long_est",
        "ancho_est",
        "area_est",
        "num_acc",
        "num_vag",
        "cap_art",
        "cap_biart",
    ]

    df_perfil[scale_cols] = df_perfil[scale_cols].round(4)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_perfil[scale_cols])

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    id_meta_cols = ["ID_Estacion", "Nombre_Estacion", "Troncal", "id_trazado", "tipo_esta"]
    df_scaled = pd.concat(
        [
            df_perfil[id_meta_cols].reset_index(drop=True),
            pd.DataFrame(np.round(scaled_values, 6), columns=scale_cols),
        ],
        axis=1,
    )

    return df_perfil, df_scaled, scale_cols


# ==============================================================================
# 5. ORQUESTADOR PRINCIPAL Y REPORTE DE AUDITORÍA JSON
# ==============================================================================
def main() -> None:
    t0 = time.perf_counter()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print(" INICIANDO FASE 2: PIPELINE ETL TRANSMILENIO S.A. (2026)")
    print("=" * 95)

    # Paso 1: Maestro de Estaciones
    station_csv = RAW_DIR / "Estaciones_Troncales_de_TRANSMILENIO.csv"
    df_stations, audit_stations = clean_and_impute_stations(station_csv)
    valid_nodos = set(df_stations["ID_Estacion"].astype(int).tolist())
    print(
        f"[1/5] Maestro de estaciones limpio e imputado: {len(df_stations)} estaciones | "
        f"Nulos restantes: {audit_stations['nulos_totales_maestro_limpio']}"
    )

    # Paso 2: Ingesta paralela de los 10 libros mensuales .xlsx
    xlsx_files = sorted(RAW_DIR.glob("*.xlsx"))
    max_workers = min(len(xlsx_files), max(1, (os.cpu_count() or 4) - 2))
    print(f"[2/5] Procesando {len(xlsx_files)} archivos mensuales .xlsx en paralelo ({max_workers} workers)...")

    worker_args = [(str(xf), valid_nodos) for xf in xlsx_files]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_xlsx, worker_args))

    monthly_dfs = [r[0] for r in results]
    monthly_audits = [r[1] for r in results]

    for ma in monthly_audits:
        print(
            f"      - {ma['mes_objetivo']} ({ma['dias_validos_conservados']} días) | "
            f"Desborde eliminado: {ma['fechas_desbordadas_eliminadas'] or 'Ninguno'} | "
            f"Validaciones limpias: {ma['validaciones_limpias_mes']:,}"
        )

    # Paso 3: Consolidación, Feature Engineering y JOIN
    print("[3/5] Consolidando vista minable Hora-Estación y tabla diaria...")
    df_minable, df_diaria = build_minable_views(monthly_dfs, df_stations)

    # Paso 4: Perfilamiento por estación y StandardScaler
    scaler_path = MODELS_DIR / "scaler.joblib"
    print("[4/5] Generando perfiles por estación y ajustando StandardScaler -> models/scaler.joblib...")
    df_perfil, df_perfil_scaled, scaled_cols = build_and_scale_station_profiles(
        df_diaria, df_stations, scaler_path
    )

    # Paso 5: Exportación de artefactos y Reporte de Auditoría JSON
    parquet_path = PROCESSED_DIR / "vista_minable.parquet"
    csv_minable_path = PROCESSED_DIR / "vista_minable.csv"
    csv_diaria_path = PROCESSED_DIR / "vista_diaria_estacion.csv"
    csv_perfil_path = PROCESSED_DIR / "perfil_estaciones.csv"
    csv_perfil_scaled_path = PROCESSED_DIR / "perfil_estaciones_escalado.csv"
    report_json_path = PROCESSED_DIR / "reporte_etl.json"

    print("[5/5] Guardando archivos en data/processed/ y models/...")
    df_minable.to_parquet(parquet_path, index=False)
    df_minable.to_csv(csv_minable_path, index=False, encoding="utf-8")
    df_diaria.to_csv(csv_diaria_path, index=False, encoding="utf-8")
    df_perfil.to_csv(csv_perfil_path, index=False, encoding="utf-8")
    df_perfil_scaled.to_csv(csv_perfil_scaled_path, index=False, encoding="utf-8")

    # Totales agregados de auditoría sobre los 10 meses
    tot_raw_rows = sum(m["filas_crudas_bajo_header"] for m in monthly_audits)
    tot_empty_rows = sum(m["filas_plantilla_vacias_eliminadas"] for m in monthly_audits)
    tot_total_gen_rows = sum(m["filas_total_general_eliminadas"] for m in monthly_audits)
    tot_op_rows = sum(m["filas_operativas_totales"] for m in monthly_audits)
    tot_excl_cable_rows = sum(m["filas_excluidas_transmicable"] for m in monthly_audits)
    tot_excl_cable_val = sum(m["validaciones_excluidas_transmicable"] for m in monthly_audits)
    tot_excl_dual_rows = sum(m["filas_excluidas_dual_calle"] for m in monthly_audits)
    tot_excl_dual_val = sum(m["validaciones_excluidas_dual_calle"] for m in monthly_audits)
    tot_homolog_val = sum(m["validaciones_recuperadas_homologacion"] for m in monthly_audits)
    tot_dual_troncal_val = sum(m["validaciones_dual_en_estacion_troncal_incluidas"] for m in monthly_audits)
    tot_madrugada_rows = sum(m["filas_madrugada_00_03_filtradas"] for m in monthly_audits)
    tot_madrugada_val = sum(m["validaciones_madrugada_00_03_filtradas"] for m in monthly_audits)
    tot_in_scope_15m_rows = sum(m["filas_15min_troncal_horario_comercial"] for m in monthly_audits)

    overflow_by_month = {
        m["mes_objetivo"]: m["fechas_desbordadas_eliminadas"]
        for m in monthly_audits
        if m["fechas_desbordadas_eliminadas"]
    }
    residual_unmatched: dict[str, float] = {}
    for m in monthly_audits:
        for k, v in m["codigos_no_mapeados_residuales"].items():
            residual_unmatched[k] = residual_unmatched.get(k, 0.0) + v

    nulls_vista_minable = int(df_minable.isna().sum().sum())
    nulls_vista_diaria = int(df_diaria.isna().sum().sum())
    nulls_perfil_scaled = int(df_perfil_scaled.isna().sum().sum())

    matched_rows = int(df_minable["Nombre_Estacion"].notna().sum())
    pct_match_cod_nodo = round(matched_rows / len(df_minable) * 100, 4) if len(df_minable) > 0 else 0.0

    elapsed_sec = round(time.perf_counter() - t0, 2)

    reporte_etl: dict[str, Any] = {
        "proyecto": "Minería de Datos - TransMilenio S.A. (2026)",
        "fase": "Fase 2 - ETL, Imputación, Feature Engineering y Escalamiento",
        "tiempo_ejecucion_segundos": elapsed_sec,
        "archivos_procesados": {
            "maestro_estaciones": station_csv.name,
            "archivos_validaciones_xlsx": len(xlsx_files),
            "rango_fechas_vista_minable": {
                "fecha_inicio": str(df_minable["Fecha"].min()),
                "fecha_fin": str(df_minable["Fecha"].max()),
                "dias_unicos": int(df_minable["Fecha"].nunique()),
                "festivos_colombia_2026_en_rango": sorted(
                    [d for d in FESTIVOS_COLOMBIA_2026 if df_minable["Fecha"].min() <= d <= df_minable["Fecha"].max()]
                ),
            },
        },
        "limpieza_ingesta_xlsx_10_meses": {
            "filas_crudas_totales_bajo_header": tot_raw_rows,
            "filas_vacias_plantilla_eliminadas": tot_empty_rows,
            "filas_resumen_total_general_eliminadas": tot_total_gen_rows,
            "filas_operativas_evaluadas": tot_op_rows,
            "fechas_desbordadas_eliminadas_por_mes": overflow_by_month,
            "total_columnas_fecha_desbordadas_eliminadas": sum(len(v) for v in overflow_by_month.values()),
        },
        "filtro_alcance_y_homologacion": {
            "codigos_homologados_aplicados": {str(k): v for k, v in HOMOLOGACION_COD_NODO.items()},
            "validaciones_recuperadas_por_homologacion": tot_homolog_val,
            "validaciones_dual_en_estacion_troncal_incluidas": tot_dual_troncal_val,
            "exclusiones_por_alcance": {
                "transmicable": {
                    "filas_15min_mensuales": tot_excl_cable_rows,
                    "validaciones_excluidas": tot_excl_cable_val,
                },
                "dual_en_calle_fuera_de_troncal": {
                    "filas_15min_mensuales": tot_excl_dual_rows,
                    "validaciones_excluidas": tot_excl_dual_val,
                },
                "otros_no_mapeados": residual_unmatched,
                "total_validaciones_excluidas_alcance": int(
                    tot_excl_cable_val + tot_excl_dual_val + sum(residual_unmatched.values())
                ),
            },
        },
        "filtro_horario_operativo": {
            "horario_conservado": "04:00 - 23:59 (Hora_Bloque 4 a 23)",
            "madrugada_00_03_filtrada": {
                "filas_15min_mensuales_filtradas": tot_madrugada_rows,
                "validaciones_residuales_filtradas": tot_madrugada_val,
            },
            "filas_15min_mensuales_en_alcance_comercial": tot_in_scope_15m_rows,
        },
        "maestro_estaciones_e_imputacion": audit_stations,
        "cruce_y_calidad_final": {
            "llave_join": "ID_Estacion (cod_nodo int64)",
            "pct_match_cod_nodo_en_alcance": pct_match_cod_nodo,
            "estaciones_troncales_activas_en_2026": int(df_minable["ID_Estacion"].nunique()),
            "estaciones_maestro_totales": int(len(df_stations)),
            "total_validaciones_limpias_vista_minable": int(df_minable["Total_Validaciones"].sum()),
            "confirmacion_cero_nulos": {
                "nulos_vista_minable": nulls_vista_minable,
                "nulos_vista_diaria_estacion": nulls_vista_diaria,
                "nulos_perfil_estaciones_escalado": nulls_perfil_scaled,
                "cumple_cero_nulos": (
                    nulls_vista_minable == 0 and nulls_vista_diaria == 0 and nulls_perfil_scaled == 0
                ),
            },
        },
        "salidas_generadas": {
            "vista_minable_parquet": {
                "ruta": "data/processed/vista_minable.parquet",
                "filas": int(df_minable.shape[0]),
                "columnas": int(df_minable.shape[1]),
                "size_mb": round(parquet_path.stat().st_size / (1024 * 1024), 2),
            },
            "vista_minable_csv": {
                "ruta": "data/processed/vista_minable.csv",
                "filas": int(df_minable.shape[0]),
                "columnas": int(df_minable.shape[1]),
                "size_mb": round(csv_minable_path.stat().st_size / (1024 * 1024), 2),
            },
            "vista_diaria_estacion_csv": {
                "ruta": "data/processed/vista_diaria_estacion.csv",
                "filas": int(df_diaria.shape[0]),
                "columnas": int(df_diaria.shape[1]),
                "size_mb": round(csv_diaria_path.stat().st_size / (1024 * 1024), 2),
            },
            "perfil_estaciones_escalado_csv": {
                "ruta": "data/processed/perfil_estaciones_escalado.csv",
                "filas": int(df_perfil_scaled.shape[0]),
                "columnas": int(df_perfil_scaled.shape[1]),
                "variables_escaladas": scaled_cols,
                "scaler_guardado_en": "models/scaler.joblib",
            },
        },
        "detalle_mensual": monthly_audits,
    }

    with report_json_path.open("w", encoding="utf-8") as f:
        json.dump(reporte_etl, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 95)
    print(f" ETL COMPLETADO EN {elapsed_sec} SEGUNDOS")
    print("=" * 95)
    print(f"  - vista_minable.parquet / .csv : {df_minable.shape[0]:,} filas x {df_minable.shape[1]} columnas")
    print(f"  - vista_diaria_estacion.csv    : {df_diaria.shape[0]:,} filas x {df_diaria.shape[1]} columnas")
    print(f"  - perfil_estaciones_escalado   : {df_perfil_scaled.shape[0]:,} estaciones x {df_perfil_scaled.shape[1]} columnas")
    print(f"  - Match por cod_nodo           : {pct_match_cod_nodo:.2f}%")
    print(f"  - Nulos en vista_minable       : {nulls_vista_minable}")
    print(f"  - Reporte JSON guardado en     : {report_json_path}")
    print("\n[PRIMERAS 8 FILAS DE vista_minable (COLUMNAS PRINCIPALES)]:")
    preview_cols = [
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
        "area_est",
        "cap_art",
    ]
    print(df_minable[preview_cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
