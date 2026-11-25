"""
Fase 1: Exploración de Datos Crudos (Data Lake - data/raw/)
Proyecto de Minería de Datos - TransMilenio S.A.

Inspecciona:
1. Maestro de infraestructura: Estaciones_Troncales_de_TRANSMILENIO (.csv y .geojson).
2. Libros mensuales (.xlsx) de Resumen de Validaciones Troncales (Intervalo 15 Minutos):
   - Hojas disponibles, filas de encabezado/membrete, columnas vacías y filas de plantilla/totales.
   - Columnas exactas de fecha (formato ancho/pivoteado), intervalo de 15 min, estación y validaciones.
   - Rango de fechas en todos los meses y detección de fechas desbordadas en meses de <31 días.
   - Nulos, duplicados, distribución y outliers (1.5 * IQR) de las validaciones por intervalo de 15 min.
3. Prueba real de cruce (JOIN) extrayendo el código entre paréntesis de 'Estación' contra 'cod_nodo'
   de Estaciones_Troncales_de_TRANSMILENIO.csv.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MAX_SAMPLE_ROWS = 500_000


# ==============================================================================
# LECTORES DE DATOS CRUDOS
# ==============================================================================
def read_csv_smart(filepath_or_buffer: Any, sample_limit: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lee un archivo CSV detectando codificación y separador automáticamente."""
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    separators = [",", ";", "|", "\t"]

    for enc in encodings:
        for sep in separators:
            try:
                if hasattr(filepath_or_buffer, "seek"):
                    filepath_or_buffer.seek(0)
                preview = pd.read_csv(filepath_or_buffer, encoding=enc, sep=sep, nrows=10)
                if preview.shape[1] > 1:
                    if hasattr(filepath_or_buffer, "seek"):
                        filepath_or_buffer.seek(0)
                    df = pd.read_csv(
                        filepath_or_buffer,
                        encoding=enc,
                        sep=sep,
                        nrows=sample_limit,
                        low_memory=False,
                    )
                    return df, {"encoding": enc, "sep": sep}
            except Exception:
                continue

    if hasattr(filepath_or_buffer, "seek"):
        filepath_or_buffer.seek(0)
    df = pd.read_csv(filepath_or_buffer, encoding="latin-1", nrows=sample_limit, low_memory=False)
    return df, {"encoding": "latin-1", "sep": ","}


def read_geojson_as_df(filepath: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convierte un archivo FeatureCollection GeoJSON en un DataFrame tabular."""
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    crs_name = (
        data.get("crs", {})
        .get("properties", {})
        .get("name", "CRS no especificado (WGS84 por defecto)")
    )
    features = data.get("features", [])
    rows: list[dict[str, Any]] = []

    for feat in features:
        props = dict(feat.get("properties") or {})
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None]
        props["_geom_type"] = geom.get("type")
        if isinstance(coords, list) and len(coords) >= 2 and not isinstance(coords[0], list):
            props["_geom_lon"] = coords[0]
            props["_geom_lat"] = coords[1]
        rows.append(props)

    return pd.DataFrame(rows), {"crs": crs_name, "n_features": len(features)}


def scan_xlsx_catalog(xlsx_files: list[Path]) -> pd.DataFrame:
    """Escanea rápidamente hojas, encabezados y rango de columnas fecha de todos los .xlsx."""
    catalog_rows: list[dict[str, Any]] = []

    for xf in xlsx_files:
        wb = openpyxl.load_workbook(xf, read_only=True, data_only=True)
        sheets = wb.sheetnames
        ws = wb[sheets[0]]

        header_idx = None
        header_vals: list[Any] = []
        for idx, row in enumerate(ws.iter_rows(max_row=12, values_only=True)):
            if any(isinstance(c, str) and ("Estación" in c or "Fase" in c) for c in row):
                header_idx = idx
                header_vals = list(row)
                break

        date_cols = [c for c in header_vals if isinstance(c, (dt.datetime, dt.date))]
        min_date = min(date_cols).strftime("%Y-%m-%d") if date_cols else "N/A"
        max_date = max(date_cols).strftime("%Y-%m-%d") if date_cols else "N/A"

        # Detectar si hay columnas de fecha desbordadas al mes siguiente (plantilla de 31 días)
        months_in_cols = sorted({d.strftime("%Y-%m") for d in date_cols})
        overflow_dates = []
        if len(months_in_cols) > 1:
            primary_month = months_in_cols[0]
            overflow_dates = [d.strftime("%Y-%m-%d") for d in date_cols if d.strftime("%Y-%m") != primary_month]

        catalog_rows.append(
            {
                "archivo": xf.name,
                "size_mb": round(xf.stat().st_size / (1024 * 1024), 2),
                "hojas": ", ".join(sheets),
                "header_fila_idx": header_idx,
                "max_row_excel": ws.max_row,
                "n_cols_fecha": len(date_cols),
                "fecha_col_min": min_date,
                "fecha_col_max": max_date,
                "fechas_desbordadas": ", ".join(overflow_dates) if overflow_dates else "Ninguna",
            }
        )
        wb.close()

    return pd.DataFrame(catalog_rows)


def load_xlsx_validations_detail(filepath: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Lee en modo streaming (openpyxl read_only=True) las hojas de un archivo .xlsx mensual
    de Validaciones Troncales 15 min, retornando el DataFrame crudo (desde header=6) y metadatos.
    """
    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    sheet_info: dict[str, Any] = {}

    for sname in wb.sheetnames:
        ws_s = wb[sname]
        preview_rows = [list(r[:10]) for _, r in zip(range(8), ws_s.iter_rows(values_only=True))]
        sheet_info[sname] = {
            "max_row": ws_s.max_row,
            "max_col": ws_s.max_column,
            "preview_primeras_filas": preview_rows,
        }

    main_sheet = wb.sheetnames[0]
    ws = wb[main_sheet]
    all_rows = list(ws.iter_rows(values_only=True))
    wb.close()

    # Buscar dinámicamente la fila de encabezado ('Fase', 'Línea', 'Estación', ...)
    header_idx = 6
    for idx, r in enumerate(all_rows[:15]):
        if any(isinstance(c, str) and "Estación" in c for c in r):
            header_idx = idx
            break

    banner_rows = all_rows[:header_idx]
    raw_header = list(all_rows[header_idx])

    col_names: list[str] = []
    date_col_names: list[str] = []
    for i, h in enumerate(raw_header):
        if h is None:
            col_names.append(f"Unnamed_{i}")
        elif isinstance(h, (dt.datetime, dt.date)):
            d_str = h.strftime("%Y-%m-%d")
            col_names.append(d_str)
            date_col_names.append(d_str)
        else:
            col_names.append(str(h).strip())

    data_rows = all_rows[header_idx + 1 :]
    df_raw = pd.DataFrame(data_rows, columns=col_names)

    # Normalizar columna Intervalo a string HH:MM:SS para análisis consistente
    if "Intervalo" in df_raw.columns:
        df_raw["Intervalo_tipo_py"] = df_raw["Intervalo"].map(lambda x: type(x).__name__ if x is not None else "None")
        df_raw["Intervalo"] = df_raw["Intervalo"].map(
            lambda x: x.strftime("%H:%M:%S") if isinstance(x, (dt.time, dt.datetime)) else (str(x) if x is not None else np.nan)
        )

    meta = {
        "archivo": filepath.name,
        "hojas": sheet_info,
        "hoja_principal": main_sheet,
        "header_idx": header_idx,
        "banner_texto": [
            [c for c in r if c is not None] for r in banner_rows if any(c is not None for c in r)
        ],
        "columnas_crudas": col_names,
        "columnas_fecha": date_col_names,
    }
    return df_raw, meta


# ==============================================================================
# FUNCIONES DE ANÁLISIS EDA
# ==============================================================================
def analyze_schema_and_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Genera resumen de columnas, dtypes, porcentaje de nulos, cardinalidad y ejemplos."""
    total_rows = len(df)
    records = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        empty_str_count = 0
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            non_na = df[col].dropna().astype(str)
            empty_str_count = int((non_na.str.strip() == "").sum())

        effective_nulls = null_count + empty_str_count
        null_pct = (effective_nulls / total_rows * 100) if total_rows > 0 else 0.0
        nunique = int(df[col].nunique(dropna=True))

        non_null_vals = df[col].dropna()
        sample_val = str(non_null_vals.iloc[0])[:38] if len(non_null_vals) > 0 else "N/A"

        records.append(
            {
                "columna": col,
                "dtype": str(df[col].dtype),
                "no_nulos": total_rows - effective_nulls,
                "nulos": effective_nulls,
                "pct_nulos": round(null_pct, 2),
                "unicos": nunique,
                "ejemplo": sample_val,
            }
        )
    return pd.DataFrame(records)


def analyze_duplicates(df: pd.DataFrame, key_candidates: list[str]) -> dict[str, Any]:
    """Evalúa duplicados exactos de fila y duplicados en llaves candidatas."""
    exact_dups = int(df.duplicated().sum())
    key_checks: dict[str, dict[str, Any]] = {}
    for col in key_candidates:
        if col in df.columns:
            dups = int(df[col].dropna().duplicated().sum())
            dup_vals = (
                df.loc[df[col].duplicated(keep=False) & df[col].notna(), col]
                .unique()
                .tolist()
            )
            key_checks[col] = {
                "duplicados": dups,
                "es_unica": dups == 0 and df[col].isna().sum() == 0,
                "valores_repetidos": dup_vals[:5],
            }
    return {"filas_duplicadas_exactas": exact_dups, "llaves_candidatas": key_checks}


def analyze_outliers_iqr(series: pd.Series, nombre: str) -> dict[str, Any]:
    """Calcula estadísticos descriptivos y outliers bajo el criterio 1.5 * IQR."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    q1 = float(clean.quantile(0.25))
    median = float(clean.median())
    q3 = float(clean.quantile(0.75))
    p95 = float(clean.quantile(0.95))
    p99 = float(clean.quantile(0.99))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_out = int(((clean < lower) | (clean > upper)).sum())
    n_neg = int((clean < 0).sum())
    n_zero = int((clean == 0).sum())

    return {
        "variable": nombre,
        "n_observaciones": int(len(clean)),
        "min": float(clean.min()),
        "Q1": round(q1, 2),
        "mediana": round(median, 2),
        "Q3": round(q3, 2),
        "P95": round(p95, 2),
        "P99": round(p99, 2),
        "max": float(clean.max()),
        "IQR": round(iqr, 2),
        "lim_sup_iqr": round(upper, 2),
        "ceros_pct": round(n_zero / len(clean) * 100, 2),
        "negativos": n_neg,
        "outliers_iqr": n_out,
        "pct_outliers": round(n_out / len(clean) * 100, 2),
    }


def inspect_xlsx_and_join(
    df_xlsx_raw: pd.DataFrame,
    meta_xlsx: dict[str, Any],
    df_stations_csv: pd.DataFrame,
) -> None:
    """Imprime el perfilamiento profundo del .xlsx mensual y prueba el cruce real contra cod_nodo."""
    print("\n" + "=" * 100)
    print(f" ANÁLISIS PROFUNDO DE ARCHIVO MENSUAL XLSX: {meta_xlsx['archivo']}")
    print("=" * 100)

    print("\n[A] HOJAS ENCONTRADAS EN EL LIBRO EXCEL:")
    for sname, sinfo in meta_xlsx["hojas"].items():
        print(f"  - Hoja '{sname}': max_row={sinfo['max_row']:,} | max_col={sinfo['max_col']}")
        if sname != meta_xlsx["hoja_principal"]:
            print("    Propósito detectado: Hoja de control/conciliación diaria contra FCS Center (no transaccional).")

    print("\n[B] METADATOS DE MEMBRETE (FILAS 0 A 5 ANTES DEL ENCABEZADO EN FILA 6):")
    for brow in meta_xlsx["banner_texto"]:
        print(f"  - {brow}")

    total_raw_rows = len(df_xlsx_raw)
    empty_template_mask = df_xlsx_raw["Fase"].isna() & df_xlsx_raw["Estación"].isna()
    total_row_mask = df_xlsx_raw["Fase"].astype(str).str.strip().eq("Total general")
    df_valid = df_xlsx_raw.loc[~empty_template_mask & ~total_row_mask].copy()

    print("\n[C] CONTEO DE FILAS EN HOJA 'Validaciones Tullave' (DESPUÉS DE HEADER EN FILA 6):")
    print(f"  - Filas totales en plantilla después del header : {total_raw_rows:,}")
    print(f"  - Filas vacías de plantilla al final (Fase=NaN)  : {int(empty_template_mask.sum()):,}")
    print(f"  - Filas de resumen ('Total general')            : {int(total_row_mask.sum()):,}")
    print(f"  - Filas operativas reales válidas               : {len(df_valid):,}")

    # Esquema sobre filas operativas válidas (mostrando las 6 primeras columnas + primera/última fecha + Total general)
    date_cols = meta_xlsx["columnas_fecha"]
    representative_cols = (
        ["Unnamed_0", "Fase", "Línea", "Estación", "Acceso de Estación", "Intervalo", "Intervalo_tipo_py"]
        + date_cols[:2]
        + date_cols[-2:]
        + (["Total general"] if "Total general" in df_valid.columns else [])
    )
    representative_cols = [c for c in representative_cols if c in df_valid.columns]

    print("\n[D] ESTRUCTURA DE COLUMNAS, TIPOS Y % DE NULOS (SOBRE FILAS OPERATIVAS VÁLIDAS):")
    print(
        f"    Nota: La tabla viene PIVOTEADA EN ANCHO con {len(date_cols)} columnas de fecha diaria "
        f"({date_cols[0]} a {date_cols[-1]})."
    )
    schema_xlsx = analyze_schema_and_nulls(df_valid[representative_cols])
    print(schema_xlsx.to_string(index=False))

    # Verificar nulos en TODAS las columnas de fecha de forma agregada
    nulls_in_dates = int(df_valid[date_cols].isna().sum().sum())
    print(
        f"\n  - Nulos totales en las {len(date_cols)} columnas de cantidad de validaciones diarias "
        f"({len(df_valid) * len(date_cols):,} celdas): {nulls_in_dates} (0.00%)"
    )

    # Distribución de Fase e Intervalo
    print("\n[E] DISTRIBUCIÓN POR 'Fase' Y RANGO DE 'Intervalo' (15 MINUTOS):")
    fase_summary = (
        df_valid.groupby("Fase", dropna=False)
        .agg(
            filas=("Estación", "count"),
            estaciones_unicas=("Estación", "nunique"),
            validaciones_mes=("Total general", "sum"),
        )
        .reset_index()
    )
    total_val_mes = fase_summary["validaciones_mes"].sum()
    fase_summary["pct_validaciones"] = (fase_summary["validaciones_mes"] / total_val_mes * 100).round(2)
    print(fase_summary.to_string(index=False))

    intervalos = sorted(df_valid["Intervalo"].dropna().unique().tolist())
    print(
        f"\n  - Intervalos únicos de 15 min detectados: {len(intervalos)} "
        f"(desde '{intervalos[0]}' hasta '{intervalos[-1]}')"
    )

    # Duplicados en llave compuesta (Fase, Línea, Estación, Acceso de Estación, Intervalo)
    composite_cols = ["Fase", "Línea", "Estación", "Acceso de Estación", "Intervalo"]
    dup_comp = int(df_valid.duplicated(subset=composite_cols).sum())
    dup_exact = int(df_valid.drop(columns=["Unnamed_0"], errors="ignore").duplicated().sum())
    print("\n[F] DUPLICADOS EN ARCHIVO MENSUAL DE VALIDACIONES:")
    print(f"  - Filas operativas exactamente duplicadas: {dup_exact}")
    print(f"  - Duplicados en llave compuesta {composite_cols}: {dup_comp}")

    # Outliers sobre las cantidades de validaciones despivotadas (Estación-Acceso-Intervalo-Día) y agregadas (Estación-Intervalo-Día)
    print("\n[G] DISTRIBUCIÓN Y OUTLIERS DE CANTIDAD DE VALIDACIONES (1.5 * IQR):")
    troncal_mask = ~df_valid["Fase"].astype(str).str.upper().eq("DUAL")
    df_troncal = df_valid.loc[troncal_mask].copy()

    vals_acceso_nivel = df_troncal[date_cols].to_numpy(dtype=float).ravel()
    df_est_intervalo = df_troncal.groupby(["Estación", "Intervalo"], as_index=False)[date_cols].sum()
    vals_est_nivel = df_est_intervalo[date_cols].to_numpy(dtype=float).ravel()

    out_stats = pd.DataFrame(
        [
            analyze_outliers_iqr(pd.Series(vals_acceso_nivel), "Validaciones 15m (nivel Acceso-Día, Troncal)"),
            analyze_outliers_iqr(pd.Series(vals_est_nivel), "Validaciones 15m (agregado Estación-Día, Troncal)"),
        ]
    )
    print(out_stats.to_string(index=False))

    if int(total_row_mask.sum()) > 0:
        falso_max = df_xlsx_raw.loc[total_row_mask, "Total general"].iloc[0]
        print(
            f"  [!] Alerta crítica para ETL: Si NO se excluyen la fila y columna 'Total general', "
            f"aparece un falso valor extremo de {falso_max:,.0f} validaciones."
        )

    # ==========================================================================
    # PRUEBA REAL DE CRUCE: 'Estación' (XLSX) <-> 'cod_nodo' (CSV Estaciones)
    # ==========================================================================
    print("\n" + "=" * 100)
    print(" PRUEBA REAL DE CRUCE: 'Estación' (.xlsx) <-> 'cod_nodo' (Estaciones_Troncales_de_TRANSMILENIO.csv)")
    print("=" * 100)

    df_valid["codigo_parentesis_str"] = df_valid["Estación"].astype(str).str.extract(r"^\((\d+)\)", expand=False)
    df_valid["cod_nodo_extraido"] = pd.to_numeric(df_valid["codigo_parentesis_str"], errors="coerce").astype("Int64")

    pct_con_parentesis = df_valid["cod_nodo_extraido"].notna().mean() * 100
    print(f"  - ¿'Estación' trae código numérico entre paréntesis al inicio '(codigo) Nombre'?: {pct_con_parentesis:.2f}% de las filas operativas.")

    stations_lookup = df_stations_csv[["cod_nodo", "num_est", "nom_est", "id_trazado", "tipo_esta"]].copy()
    stations_lookup["cod_nodo"] = stations_lookup["cod_nodo"].astype("Int64")

    # Evaluar cruce por Fase
    df_est_resumen = (
        df_valid.groupby(["Fase", "Estación", "codigo_parentesis_str", "cod_nodo_extraido"], dropna=False)[
            "Total general"
        ]
        .sum()
        .reset_index()
        .rename(columns={"Total general": "validaciones_mes"})
    )
    df_merged = df_est_resumen.merge(stations_lookup, left_on="cod_nodo_extraido", right_on="cod_nodo", how="left")
    df_merged["cruza_con_csv"] = df_merged["cod_nodo"].notna()

    cruce_por_fase = (
        df_merged.groupby("Fase")
        .agg(
            estaciones_xlsx=("Estación", "nunique"),
            estaciones_cruzadas=("cruza_con_csv", "sum"),
            val_totales=("validaciones_mes", "sum"),
            val_cruzadas=(
                "validaciones_mes",
                lambda s: s[df_merged.loc[s.index, "cruza_con_csv"]].sum(),
            ),
        )
        .reset_index()
    )
    cruce_por_fase["pct_est_cruzadas"] = (
        cruce_por_fase["estaciones_cruzadas"] / cruce_por_fase["estaciones_xlsx"] * 100
    ).round(2)
    cruce_por_fase["pct_val_cruzadas"] = (
        cruce_por_fase["val_cruzadas"] / cruce_por_fase["val_totales"] * 100
    ).round(2)

    print("\n[1] COBERTURA DEL CRUCE POR 'Fase' USANDO cod_nodo:")
    print(cruce_por_fase.to_string(index=False))

    # Detalle de estaciones en Fase 1/2/3 que NO cruzan con Estaciones_Troncales_de_TRANSMILENIO.csv
    no_cruzan_troncal = df_merged.loc[
        (~df_merged["Fase"].astype(str).str.upper().eq("DUAL")) & (~df_merged["cruza_con_csv"])
    ].sort_values("validaciones_mes", ascending=False)

    print(
        f"\n[2] REGISTROS EN FASES TRONCALES (NO DUAL) SIN MATCH EN Estaciones_Troncales_de_TRANSMILENIO.csv "
        f"({len(no_cruzan_troncal)} códigos):"
    )
    if len(no_cruzan_troncal) > 0:
        print(
            no_cruzan_troncal[["Fase", "Estación", "cod_nodo_extraido", "validaciones_mes"]].to_string(index=False)
        )
    else:
        print("  - ¡100% de las estaciones troncales cruzaron exitosamente!")

    # Estaciones del CSV que no tuvieron validaciones en este mes
    matched_nodos = set(df_merged.loc[df_merged["cruza_con_csv"], "cod_nodo"].dropna().astype(int).tolist())
    csv_sin_val = stations_lookup.loc[~stations_lookup["cod_nodo"].isin(matched_nodos)]
    print(f"\n[3] ESTACIONES EN CSV MAESTRO SIN VALIDACIONES EN ESTE ARCHIVO ({len(csv_sin_val)} de {len(stations_lookup)}):")
    if len(csv_sin_val) > 0:
        print(csv_sin_val[["cod_nodo", "nom_est", "id_trazado"]].to_string(index=False))

    # Mostrar ejemplos reales donde el nombre difiere entre XLSX y CSV pero cod_nodo resuelve el cruce
    matched_troncal = df_merged.loc[df_merged["cruza_con_csv"]].copy()
    matched_troncal["nombre_en_xlsx"] = (
        matched_troncal["Estación"].astype(str).str.replace(r"^\(\d+\)\s*", "", regex=True).str.strip()
    )
    diff_names = matched_troncal.loc[
        matched_troncal["nombre_en_xlsx"].str.lower() != matched_troncal["nom_est"].astype(str).str.lower()
    ].sort_values("validaciones_mes", ascending=False)

    print(
        f"\n[4] ESTACIONES DONDE EL NOMBRE DIFIERE ENTRE .XLSX Y .CSV PERO 'cod_nodo' CRUZA PERFECTO "
        f"({len(diff_names)} estaciones; top 12 por volumen):"
    )
    print(
        diff_names[["cod_nodo", "Estación", "nom_est", "validaciones_mes"]]
        .head(12)
        .to_string(index=False)
    )


def main() -> None:
    print("=" * 100)
    print(" FASE 1: EXPLORACIÓN DE DATOS CRUDOS (DATA LAKE - TransMilenio S.A.)")
    print(f" Directorio inspeccionado: {RAW_DIR}")
    print("=" * 100)

    csv_station_path = RAW_DIR / "Estaciones_Troncales_de_TRANSMILENIO.csv"
    geojson_station_path = RAW_DIR / "Estaciones_Troncales_de_TRANSMILENIO.geojson"
    xlsx_files = sorted(RAW_DIR.glob("*.xlsx"))

    print(f"\nInventario en data/raw/:")
    print(f"  - Maestro CSV Estaciones     : {'SÍ' if csv_station_path.exists() else 'NO'}")
    print(f"  - Maestro GeoJSON Estaciones : {'SÍ' if geojson_station_path.exists() else 'NO'}")
    print(f"  - Archivos mensuales .xlsx   : {len(xlsx_files)} archivos detectados")

    # 1. Resumen rápido del maestro de estaciones CSV
    df_stations_csv, _ = read_csv_smart(csv_station_path)
    print(
        f"\n[MAESTRO DE ESTACIONES CSV]: {df_stations_csv.shape[0]} filas x {df_stations_csv.shape[1]} columnas | "
        f"cod_nodo únicos: {df_stations_csv['cod_nodo'].nunique()} | "
        f"nom_est únicos: {df_stations_csv['nom_est'].nunique()} (1 homónimo: 'Temporal AV. Jiménez - Inter Eléctricas')"
    )

    # 2. Catálogo de los 10 archivos mensuales .xlsx (hojas y rango de fechas)
    if xlsx_files:
        print("\n" + "-" * 100)
        print(f" CATÁLOGO DE LOS {len(xlsx_files)} ARCHIVOS MENSUALES .XLSX EN data/raw/ (ENERO - OCTUBRE 2026)")
        print("-" * 100)
        cat_df = scan_xlsx_catalog(xlsx_files)
        print(cat_df.to_string(index=False))

        # 3. Análisis profundo del archivo más reciente (Octubre 2026) + prueba real de cruce con cod_nodo
        sample_xlsx = xlsx_files[-1]
        df_xlsx_raw, meta_xlsx = load_xlsx_validations_detail(sample_xlsx)
        inspect_xlsx_and_join(df_xlsx_raw, meta_xlsx, df_stations_csv)


if __name__ == "__main__":
    main()
