"""
Generador de Presentación Ejecutiva y Técnica de Sustentación (.pptx)
Proyecto de Minería de Datos — TransMilenio S.A. (2026)
Ingeniería de Sistemas · UNIMINUTO · Bogotá D.C.
Autores: Juan José Morales Benítez · Johan Steven Sarmiento Cruz

Lee los resultados reales generados por src/etl.py y src/modelo.py en data/processed/ y data/output/,
renderiza gráficas a 300 DPI y fórmulas matemáticas LaTeX (matplotlib mathtext), y ensambla la
presentación final widescreen 16:9 en:
  presentacion/Sustentacion_Mineria_Datos_TransMilenio_2026.pptx
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# ==============================================================================
# RUTAS Y PALETA INSTITUCIONAL TRANSMILENIO / UNIMINUTO
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "output"
PRES_DIR = BASE_DIR / "presentacion"
ASSETS_DIR = PRES_DIR / "assets"
PPTX_OUT = PRES_DIR / "Sustentacion_Mineria_Datos_TransMilenio_2026.pptx"

C_RED = RGBColor(0xD3, 0x1F, 0x26)      # Rojo TransMilenio
C_RED_SOFT = RGBColor(0xFE, 0xF2, 0xF2) # Fondo rojo suave
C_DARK = RGBColor(0x0F, 0x17, 0x2A)     # Azul marino profundo
C_SLATE = RGBColor(0x33, 0x41, 0x55)    # Gris pizarra texto
C_MUTED = RGBColor(0x64, 0x74, 0x8B)    # Gris secundario
C_BLUE = RGBColor(0x02, 0x84, 0xC7)     # Azul técnico
C_BLUE_SOFT = RGBColor(0xEC, 0xF6, 0xFF)
C_GREEN = RGBColor(0x05, 0x96, 0x69)    # Esmeralda
C_GREEN_SOFT = RGBColor(0xEC, 0xFD, 0xF5)
C_AMBER = RGBColor(0xD9, 0x77, 0x06)    # Ámbar
C_AMBER_SOFT = RGBColor(0xFF, 0xFB, 0xEB)
C_BG_CARD = RGBColor(0xF8, 0xFA, 0xFC)  # Fondo tarjetas limpias
C_BORDER = RGBColor(0xE2, 0xE8, 0xF0)   # Borde suave
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)

HEX_RED = "#D31F26"
HEX_DARK = "#0F172A"
HEX_SLATE = "#475569"
HEX_BLUE = "#0284C7"
HEX_GREEN = "#059669"
HEX_AMBER = "#D97706"

CLUSTER_HEX = {
    0: "#D31F26",
    1: "#1D4ED8",
    2: "#059669",
    3: "#D97706",
}

FONT_MAIN = "Calibri"
TOTAL_SLIDES = 17

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.7,
    }
)


# ==============================================================================
# UTILIDADES DE RENDERIZADO (IMÁGENES REDONDEADAS, FÓRMULAS LATEX Y GRÁFICAS)
# ==============================================================================
def round_corners_image(src_path: Path, dst_path: Path, radius: int = 36, target_ratio: float = 4 / 3) -> Path:
    """Recorta a proporción objetivo y aplica esquinas redondeadas transparentes."""
    img = Image.open(src_path).convert("RGBA")
    w, h = img.size
    curr_ratio = w / h
    if curr_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), (w, h)], radius=radius, fill=255)
    img.putalpha(mask)
    img.save(dst_path, format="PNG")
    return dst_path


def render_formula_png(
    latex_lines: list[str],
    filename: str,
    fontsize: int = 15,
    bg_hex: str = "#FFFFFF",
    text_color: str = "#0F172A",
    line_spacing: float = 1.45,
) -> Path:
    """Renderiza una o más fórmulas matemáticas en PNG a 300 DPI usando matplotlib mathtext."""
    out_path = ASSETS_DIR / filename
    n_lines = len(latex_lines)
    fig_h = max(0.55, 0.44 * n_lines * (fontsize / 14.0))
    fig = plt.figure(figsize=(8.5, fig_h), dpi=300)
    fig.patch.set_facecolor(bg_hex)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_facecolor(bg_hex)

    if n_lines == 1:
        ax.text(0.01, 0.5, latex_lines[0], fontsize=fontsize, color=text_color, va="center", ha="left")
    else:
        ys = np.linspace(0.84, 0.16, n_lines)
        for y, expr in zip(ys, latex_lines):
            ax.text(0.01, y, expr, fontsize=fontsize, color=text_color, va="center", ha="left")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.06, facecolor=bg_hex)
    plt.close(fig)

    # Recortar márgenes vacíos sobrantes del lienzo para maximizar nitidez y tamaño en la diapositiva
    with Image.open(out_path).convert("RGB") as im:
        bg_rgb = tuple(int(bg_hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        arr = np.asarray(im)
        diff = np.any(np.abs(arr.astype(int) - np.array(bg_rgb)) > 6, axis=2)
        if np.any(diff):
            ys_idx, xs_idx = np.where(diff)
            pad = 18
            y0 = max(0, int(ys_idx.min()) - pad)
            y1 = min(arr.shape[0], int(ys_idx.max()) + pad)
            x0 = max(0, int(xs_idx.min()) - pad)
            x1 = min(arr.shape[1], int(xs_idx.max()) + pad)
            im.crop((x0, y0, x1, y1)).save(out_path)
    return out_path


def add_picture_fit(slide: Any, img_path: Path, left: float, top: float, max_w: float, max_h: float) -> None:
    """Inserta una imagen preservando exactamente su relación de aspecto dentro del cuadro (max_w, max_h)."""
    with Image.open(img_path) as im:
        iw, ih = im.size
    aspect = iw / ih
    if max_w / max_h > aspect:
        h = max_h
        w = h * aspect
    else:
        w = max_w
        h = w / aspect
    slide.shapes.add_picture(str(img_path), Inches(left), Inches(top), width=Inches(w), height=Inches(h))


def generate_all_charts(data: dict[str, Any]) -> dict[str, Path]:
    """Genera todas las gráficas reales del proyecto a 300 DPI en presentacion/assets/."""
    charts: dict[str, Path] = {}
    rep_etl = data["rep_etl"]
    df_min = data["df_min"]
    df_raw_p = data["perfil_raw"]
    df_sc_p = data["perfil_scaled"]
    df_codo = data["codo_sil"]
    df_cl = data["clusters"]
    df_rad = data["radar"]
    df_bt = data["backtest"]
    df_fi = data["feat_imp"]
    df_proy_d = data["proy_diaria"]

    # 1. Balance ETL y evolución mensual limpia (Slide 6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.6), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.35]})
    cruce = rep_etl["cruce_y_calidad_final"]
    alc = rep_etl["filtro_alcance_y_homologacion"]["exclusiones_por_alcance"]
    mad = rep_etl["filtro_horario_operativo"]["madrugada_00_03_filtrada"]

    cats = ["Troncal limpia\n(04:00–23:59)", "Dual en calle\n(excluido)", "TransMiCable\n(excluido)", "Madrugada\n(00:00–03:59)"]
    vals_m = [
        cruce["total_validaciones_limpias_vista_minable"] / 1e6,
        alc["dual_en_calle_fuera_de_troncal"]["validaciones_excluidas"] / 1e6,
        alc["transmicable"]["validaciones_excluidas"] / 1e6,
        mad["validaciones_residuales_filtradas"] / 1e6,
    ]
    colors_b = [HEX_RED, "#64748B", HEX_AMBER, "#94A3B8"]
    bars = ax1.barh(cats[::-1], vals_m[::-1], color=colors_b[::-1], height=0.58)
    for b, v in zip(bars, vals_m[::-1]):
        ax1.text(v + 6, b.get_y() + b.get_height() / 2, f"{v:.2f}M", va="center", fontsize=8.5, fontweight="bold", color=HEX_DARK)
    ax1.set_xlim(0, 495)
    ax1.set_title("Balance de Validaciones (Millones)", fontsize=10, fontweight="bold", color=HEX_DARK, pad=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    ax1.tick_params(labelsize=8)

    det_mes = pd.DataFrame(rep_etl["detalle_mensual"])
    mes_lbl = [m.split("-")[1] for m in det_mes["mes_objetivo"]]
    mes_names = {"01": "Ene", "02": "Feb", "03": "Mar", "04": "Abr", "05": "May", "06": "Jun", "07": "Jul", "08": "Ago", "09": "Sep", "10": "Oct"}
    x_mes = [mes_names[m] for m in mes_lbl]
    y_mes = det_mes["validaciones_limpias_mes"] / 1e6
    bar_cols = [HEX_RED if m in ("09", "10") else HEX_DARK for m in mes_lbl]
    b2 = ax2.bar(x_mes, y_mes, color=bar_cols, width=0.62)
    for b, v in zip(b2, y_mes):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}M", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=HEX_DARK)
    ax2.set_ylim(0, 52)
    ax2.set_title("Validaciones Troncales Limpias por Mes (Rojo = Test Sep–Oct)", fontsize=9.8, fontweight="bold", color=HEX_DARK, pad=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.tick_params(labelsize=8)

    plt.tight_layout(pad=1.2)
    p = ASSETS_DIR / "chart_etl_balance.png"
    fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)
    charts["etl_balance"] = p

    # 2. Comparativa StandardScaler Antes vs Después (Slide 7)
    vars_sc = ["prom_diario_habil", "max_validaciones_hora", "pct_pico_am", "area_est", "cap_art"]
    short_lbls = ["Dem.\nHábil", "Máx.\nHora", "% Pico\nAM", "Área\nEst.", "Cap.\nArtic."]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 3.3), dpi=300)
    bp1 = ax1.boxplot([df_raw_p[c].values for c in vars_sc], tick_labels=short_lbls, patch_artist=True, medianprops=dict(color="white", lw=1.4))
    for patch in bp1["boxes"]:
        patch.set_facecolor("#475569")
        patch.set_alpha(0.85)
    ax1.set_yscale("log")
    ax1.set_title("Antes (Escala Log Original)", fontsize=9.5, fontweight="bold", color=HEX_DARK)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.tick_params(labelsize=7.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    bp2 = ax2.boxplot([df_sc_p[c].values for c in vars_sc], tick_labels=short_lbls, patch_artist=True, medianprops=dict(color="white", lw=1.4))
    for patch in bp2["boxes"]:
        patch.set_facecolor(HEX_RED)
        patch.set_alpha(0.85)
    ax2.axhline(0, color=HEX_DARK, linestyle="--", lw=0.9)
    ax2.set_title("Después (StandardScaler: μ=0, σ=1)", fontsize=9.5, fontweight="bold", color=HEX_DARK)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.tick_params(labelsize=7.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.0)
    p = ASSETS_DIR / "chart_scaler.png"
    fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)
    charts["scaler"] = p

    # 3. Serie Histórica Diaria Ene-Oct 2026 (Slide 8 izquierda)
    daily = df_min.groupby("Fecha", as_index=False)["Total_Validaciones"].sum().sort_values("Fecha")
    daily["Fecha_dt"] = pd.to_datetime(daily["Fecha"])
    daily["MM7"] = daily["Total_Validaciones"].rolling(7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(6.3, 3.65), dpi=300)
    ax.plot(daily["Fecha_dt"], daily["Total_Validaciones"] / 1e6, color="#94A3B8", lw=0.9, alpha=0.85, label="Validaciones diarias")
    ax.plot(daily["Fecha_dt"], daily["MM7"] / 1e6, color=HEX_RED, lw=2.3, label="Tendencia (Media Móvil 7d)")
    ax.fill_between(daily["Fecha_dt"], daily["MM7"] / 1e6, alpha=0.08, color=HEX_RED)
    ax.set_ylabel("Millones de validaciones / día", fontsize=8.5, color=HEX_DARK)
    ax.set_title("Evolución Diaria Troncal (Enero – Octubre 2026 · 304 días)", fontsize=10, fontweight="bold", color=HEX_DARK, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=8.5)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    p = ASSETS_DIR / "chart_hist_series.png"
    fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)
    charts["hist_series"] = p

    # 4. Heatmap Hora x Día de Semana (Slide 8 derecha)
    hd = (
        df_min.groupby(["Fecha", "Dia_Semana", "Hora_Bloque"], as_index=False)["Total_Validaciones"]
        .sum()
        .groupby(["Dia_Semana", "Hora_Bloque"], as_index=False)["Total_Validaciones"]
        .mean()
    )
    piv = hd.pivot(index="Dia_Semana", columns="Hora_Bloque", values="Total_Validaciones").reindex(index=range(7), columns=range(4, 24), fill_value=0)
    dias_lbl = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    fig, ax = plt.subplots(figsize=(6.0, 3.65), dpi=300)
    im = ax.imshow(piv.values / 1e3, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7))
    ax.set_yticklabels(dias_lbl, fontsize=8.5)
    ax.set_xticks(range(0, 20, 2))
    ax.set_xticklabels([f"{h:02d}h" for h in range(4, 24, 2)], fontsize=8)
    ax.set_xlabel("Hora del Día (Operación Comercial 04:00–23:59)", fontsize=8.5, color=HEX_DARK)
    ax.set_title("Mapa de Calor Horario (Miles de Validaciones / Hora)", fontsize=10, fontweight="bold", color=HEX_DARK, pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label("Miles val/h", fontsize=8)
    plt.tight_layout()
    p = ASSETS_DIR / "chart_heatmap.png"
    fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)
    charts["heatmap"] = p

    # 5. Curva Dual Codo (Inercia) + Silhouette (Slide 9 derecha)
    fig, ax1 = plt.subplots(figsize=(5.8, 3.9), dpi=300)
    ax2 = ax1.twinx()
    l1 = ax1.plot(df_codo["k"], df_codo["inercia"], marker="o", color=HEX_DARK, lw=2.3, markersize=6, label="Inercia (WCSS)")
    l2 = ax2.plot(df_codo["k"], df_codo["silhouette_score"], marker="s", color=HEX_RED, lw=2.3, markersize=6, label="Silhouette Score")
    ax1.axvline(4, color=HEX_GREEN, linestyle="--", lw=1.8)
    ax2.annotate(
        "k = 4 óptimo\nSil = 0,2252\nDB = 1,4131",
        xy=(4, 0.2252),
        xytext=(4.65, 0.31),
        fontsize=8.2,
        fontweight="bold",
        color=HEX_GREEN,
        bbox=dict(boxstyle="round,pad=0.35", fc="#ECFDF5", ec=HEX_GREEN, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=HEX_GREEN, lw=1.3),
    )
    ax1.set_xticks(df_codo["k"])
    ax1.set_xlabel("Número de Clústeres (k)", fontsize=9, color=HEX_DARK)
    ax1.set_ylabel("Inercia Intra-Clúster (WCSS)", fontsize=9, color=HEX_DARK)
    ax2.set_ylabel("Coeficiente de Silhouette", fontsize=9, color=HEX_RED)
    ax1.set_title("Selección de k Óptimo: Método del Codo + Silhouette", fontsize=10, fontweight="bold", color=HEX_DARK, pad=10)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=8.5)
    plt.tight_layout()
    p = ASSETS_DIR / "chart_elbow_sil.png"
    fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)
    charts["elbow_sil"] = p

    # 6. Mapa Geoespacial de Bogotá — 142 Estaciones por Clúster (Slide 10 izquierda)
    fig, ax = plt.subplots(figsize=(5.6, 4.9), dpi=300)
    ax.set_facecolor("#F8FAFC")
    short_names = {0: "C0: Mega-Portales (10)", 1: "C1: Troncales Fuertes (21)", 2: "C2: Intermedias PM (72)", 3: "C3: Periféricas AM (39)"}
    for cl_id in [2, 3, 1, 0]:
        sub = df_cl[df_cl["Cluster"] == cl_id]
        sizes = np.clip(sub["prom_diario_habil"] / 230.0, 22, 310)
        ax.scatter(
            sub["longitud"],
            sub["latitud"],
            s=sizes,
            c=CLUSTER_HEX[cl_id],
            alpha=0.86 if cl_id == 0 else 0.78,
            edgecolors="white",
            linewidths=0.8,
            label=short_names[cl_id],
            zorder=4 if cl_id == 0 else 3,
        )
    # Etiquetar hitos clave sin solaparse
    key_labels = {
        "Portal Norte – Unicervantes": (-72, 4, "Portal Norte"),
        "Portal Suba": (-60, 6, "Portal Suba"),
        "Portal 80": (-54, 5, "Portal 80"),
        "Portal El Dorado – C.C. Nuestro Bogotá": (-76, -12, "Portal Dorado"),
        "Portal Américas": (-20, 15, "Portal Américas"),
        "San Mateo - CC Unisur": (6, -12, "San Mateo"),
        "Portal Sur - JFK Coop. Financiera": (8, -6, "Portal Sur"),
        "Portal Usme": (7, -2, "Portal Usme"),
        "Calle 100 - Marketmedios": (-62, -14, "Calle 100"),
    }
    for orig_name, (dx, dy, clean_lbl) in key_labels.items():
        row_m = df_cl[df_cl["Nombre_Estacion"] == orig_name]
        if len(row_m) > 0:
            r0 = row_m.iloc[0]
            ax.annotate(
                clean_lbl,
                xy=(r0["longitud"], r0["latitud"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.3,
                fontweight="bold",
                color=HEX_DARK,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CBD5E1", alpha=0.88),
            )
    ax.set_xlabel("Longitud (WGS84)", fontsize=8.5, color=HEX_DARK)
    ax.set_ylabel("Latitud (WGS84)", fontsize=8.5, color=HEX_DARK)
    ax.set_title("Mapa Geoespacial de Bogotá · 142 Estaciones por Clúster", fontsize=9.8, fontweight="bold", color=HEX_DARK, pad=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=7.8, markerscale=0.7)
    ax.tick_params(labelsize=7.5)
    plt.tight_layout()
    p = ASSETS_DIR / "chart_bogota_map.png"
    fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)
    charts["bogota_map"] = p

    # 7. Gráfico de Radar de los 4 Clústeres (Slide 10 derecha superior)
    radar_dims = [
        ("Demanda_Dia_Habil_norm", "Demanda\nHábil"),
        ("Concentracion_Pico_AM_norm", "% Pico AM\n(Origen)"),
        ("Concentracion_Pico_PM_norm", "% Pico PM\n(Destino)"),
        ("Resiliencia_Fin_Semana_norm", "Resiliencia\nDom/Fest"),
        ("Capacidad_Instalada_norm", "Capacidad\nFísica"),
        ("Indice_Saturacion_Pico_norm", "Saturación\nPico"),
    ]
    angles = np.linspace(0, 2 * np.pi, len(radar_dims), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(5.6, 3.65), dpi=300)
    ax = fig.add_subplot(111, polar=True)
    for _, row in df_rad.iterrows():
        c_id = int(row["Cluster"])
        vals = [float(row[col]) for col, _ in radar_dims]
        vals += vals[:1]
        ax.plot(angles, vals, lw=2.1, color=CLUSTER_HEX[c_id], label=f"C{c_id} ({int(row['n_estaciones'])} est.)")
        ax.fill(angles, vals, color=CLUSTER_HEX[c_id], alpha=0.14)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([lbl for _, lbl in radar_dims], fontsize=8.0, color=HEX_DARK, fontweight="bold")
    ax.tick_params(axis="x", pad=12)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=6.5, color="#64748B")
    ax.set_title("Perfil Radar Normalizado (0–1): Asimetría AM vs PM", fontsize=9.8, fontweight="bold", color=HEX_DARK, pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=7.6)
    fig.subplots_adjust(top=0.80, bottom=0.16)
    p = ASSETS_DIR / "chart_radar.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    charts["radar"] = p

    # 8. Backtest Cronológico Test Sep-Oct 2026 (Slide 12 izquierda)
    bt_sys = df_bt[df_bt["Nivel"] == "Diario_Sistema"].sort_values("Fecha").copy()
    bt_sys["Fecha_dt"] = pd.to_datetime(bt_sys["Fecha"])
    fig, ax = plt.subplots(figsize=(6.3, 3.05), dpi=300)
    ax.plot(bt_sys["Fecha_dt"], bt_sys["Real_Validaciones"] / 1e6, color=HEX_DARK, lw=2.1, marker="o", markersize=2.8, label="Real Observado")
    ax.plot(bt_sys["Fecha_dt"], bt_sys["Pred_XGBoost"] / 1e6, color=HEX_RED, lw=2.0, label="XGBoost (WAPE 2,58%)")
    ax.plot(bt_sys["Fecha_dt"], bt_sys["Pred_Baseline_Hist"] / 1e6, color="#94A3B8", lw=1.4, linestyle=":", label="Baseline Histórico (WAPE 6,65%)")
    ax.set_ylabel("Millones val / día", fontsize=8.5, color=HEX_DARK)
    ax.set_title("Backtest Out-of-Sample (Sep–Oct 2026 · 61 días sin fuga)", fontsize=9.8, fontweight="bold", color=HEX_DARK, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.29), ncol=3, frameon=False, fontsize=7.6)
    ax.tick_params(labelsize=7.5)
    plt.tight_layout()
    p = ASSETS_DIR / "chart_backtest.png"
    fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)
    charts["backtest"] = p

    # 9. Feature Importance Top 10 (Slide 12 derecha)
    top10 = df_fi.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.05), dpi=300)
    cols_fi = [HEX_RED if i >= 8 else HEX_DARK for i in range(len(top10))]
    bars = ax.barh(top10["variable"], top10["importancia_pct"], color=cols_fi, height=0.62)
    for b, v in zip(bars, top10["importancia_pct"]):
        ax.text(v + 1.0, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center", fontsize=7.6, fontweight="bold", color=HEX_DARK)
    ax.set_xlim(0, 96)
    ax.set_xlabel("Importancia por Ganancia (%)", fontsize=8.2, color=HEX_DARK)
    ax.set_title("Top 10 Variables más Influyentes en XGBRegressor", fontsize=9.8, fontweight="bold", color=HEX_DARK, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.45)
    ax.tick_params(labelsize=7.6)
    plt.tight_layout()
    p = ASSETS_DIR / "chart_feat_imp.png"
    fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)
    charts["feat_imp"] = p

    # 10. Proyección 3 Escenarios Noviembre 2026 (Slide 13 derecha)
    proy_sys = df_proy_d[df_proy_d["Nivel"] == "Sistema_Total"].sort_values("Fecha").copy()
    proy_sys["Fecha_dt"] = pd.to_datetime(proy_sys["Fecha"])
    oct_tail = (
        df_min[df_min["Fecha"] >= "2026-10-15"]
        .groupby("Fecha", as_index=False)["Total_Validaciones"]
        .sum()
        .sort_values("Fecha")
    )
    oct_tail["Fecha_dt"] = pd.to_datetime(oct_tail["Fecha"])

    fig, ax = plt.subplots(figsize=(6.4, 3.85), dpi=300)
    ax.plot(
        oct_tail["Fecha_dt"].tolist() + [proy_sys["Fecha_dt"].iloc[0]],
        (oct_tail["Total_Validaciones"] / 1e6).tolist() + [proy_sys["Pred_Base"].iloc[0] / 1e6],
        color=HEX_DARK,
        lw=2.1,
        marker="o",
        markersize=2.8,
        label="Histórico Oct 15–31",
    )
    ax.fill_between(
        proy_sys["Fecha_dt"],
        proy_sys["Pred_Pesimista"] / 1e6,
        proy_sys["Pred_Optimista"] / 1e6,
        color=HEX_GREEN,
        alpha=0.13,
        label="Banda Q10–Q90",
    )
    ax.plot(proy_sys["Fecha_dt"], proy_sys["Pred_Optimista"] / 1e6, color=HEX_GREEN, lw=1.8, linestyle="--", label="Optimista (44,70M)")
    ax.plot(proy_sys["Fecha_dt"], proy_sys["Pred_Base"] / 1e6, color=HEX_RED, lw=2.5, marker="o", markersize=3.2, label="Base L2 (42,07M)")
    ax.plot(proy_sys["Fecha_dt"], proy_sys["Pred_Pesimista"] / 1e6, color=HEX_AMBER, lw=1.8, linestyle="--", label="Pesimista (37,99M)")

    for f_str, f_txt, y_frac in [("2026-11-02", "2 Nov: Todos los Santos", 0.94), ("2026-11-16", "16 Nov: Indep. Cartagena", 0.94)]:
        f_dt = pd.to_datetime(f_str)
        ax.axvline(f_dt, color="#7C3AED", linestyle=":", lw=1.3)
        ax.text(f_dt, y_frac, f" {f_txt}", transform=ax.get_xaxis_transform(), fontsize=7.2, color="#6D28D9", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#DDD6FE", alpha=0.95))

    ax.set_ylim(0.40, 2.12)
    ax.set_ylabel("Millones validaciones / día", fontsize=8.5, color=HEX_DARK)
    ax.set_title("Transición Octubre → Proyección 3 Escenarios (Noviembre 2026)", fontsize=9.8, fontweight="bold", color=HEX_DARK, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=7.4)
    ax.tick_params(labelsize=7.5)
    plt.tight_layout()
    p = ASSETS_DIR / "chart_scenarios_nov.png"
    fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)
    charts["scenarios_nov"] = p

    return charts


# ==============================================================================
# UTILIDADES DE DISEÑO POWERPOINT (SLIDE MASTER LIMPIO ESTILO UNIMINUTO)
# ==============================================================================
def add_rect(
    slide: Any,
    left: float,
    top: float,
    width: float,
    height: float,
    fill_color: RGBColor,
    border_color: RGBColor | None = None,
    border_pt: float = 0.75,
    rounded: bool = False,
) -> Any:
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, Inches(left), Inches(top), Inches(width), Inches(height))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill_color
    if border_color is not None:
        shp.line.color.rgb = border_color
        shp.line.width = Pt(border_pt)
    else:
        shp.line.fill.background()
    if rounded:
        shp.adjustments[0] = 0.06
    return shp


def add_textbox(
    slide: Any,
    left: float,
    top: float,
    width: float,
    height: float,
    text: str,
    font_size: float = 12,
    bold: bool = False,
    color: RGBColor = C_DARK,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    italic: bool = False,
    v_anchor: MSO_ANCHOR = MSO_ANCHOR.TOP,
) -> Any:
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = v_anchor
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = FONT_MAIN
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tb


def add_bullets(
    slide: Any,
    left: float,
    top: float,
    width: float,
    height: float,
    items: list[tuple[str, str]],
    font_size: float = 11.2,
    bullet_color: RGBColor = C_RED,
    space_after: float = 6,
) -> Any:
    """Agrega viñetas limpias con prefijo en negrita y cuerpo en gris pizarra."""
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    for idx, (lead, body) in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = 1.16

        r_dot = p.add_run()
        r_dot.text = "▸ "
        r_dot.font.name = FONT_MAIN
        r_dot.font.size = Pt(font_size)
        r_dot.font.bold = True
        r_dot.font.color.rgb = bullet_color

        if lead:
            r_lead = p.add_run()
            r_lead.text = lead + " "
            r_lead.font.name = FONT_MAIN
            r_lead.font.size = Pt(font_size)
            r_lead.font.bold = True
            r_lead.font.color.rgb = C_DARK

        r_body = p.add_run()
        r_body.text = body
        r_body.font.name = FONT_MAIN
        r_body.font.size = Pt(font_size)
        r_body.font.color.rgb = C_SLATE
    return tb


def setup_content_slide(prs: Presentation, kicker: str, title: str, slide_num: int) -> Any:
    """Crea un slide 16:9 limpio con identidad visual TransMilenio/UNIMINUTO y pie de página."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    # Fondo blanco limpio
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = C_WHITE

    # Franja roja institucional superior
    add_rect(slide, 0, 0, 13.333, 0.08, C_RED)
    # Kicker y título
    add_textbox(slide, 0.55, 0.22, 12.2, 0.26, kicker.upper(), font_size=9.5, bold=True, color=C_RED)
    add_textbox(slide, 0.55, 0.46, 12.2, 0.48, title, font_size=22.5, bold=True, color=C_DARK)
    add_rect(slide, 0.55, 0.99, 12.23, 0.018, C_BORDER)

    # Footer
    add_rect(slide, 0.55, 7.05, 12.23, 0.012, C_BORDER)
    add_textbox(
        slide,
        0.55,
        7.10,
        9.5,
        0.28,
        "Minería de Datos en TransMilenio S.A. (2026)  ·  Ingeniería de Sistemas · UNIMINUTO Bogotá",
        font_size=8.5,
        color=C_MUTED,
    )
    add_textbox(
        slide,
        11.3,
        7.10,
        1.48,
        0.28,
        f"{slide_num:02d} / {TOTAL_SLIDES:02d}",
        font_size=8.5,
        bold=True,
        color=C_SLATE,
        align=PP_ALIGN.RIGHT,
    )
    return slide


def add_styled_table(
    slide: Any,
    left: float,
    top: float,
    width: float,
    height: float,
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[float] | None = None,
    font_size: float = 9.3,
    highlight_row_idx: int | None = None,
) -> Any:
    """Agrega una tabla corporativa limpia con encabezado azul oscuro y filas alternadas."""
    n_rows = len(rows) + 1
    n_cols = len(headers)
    table_shape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top), Inches(width), Inches(height))
    tbl = table_shape.table

    if col_widths is not None:
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Inches(cw)

    # Encabezado
    for j, h_text in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_DARK
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if j <= 1 else PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h_text
        r.font.name = FONT_MAIN
        r.font.size = Pt(font_size)
        r.font.bold = True
        r.font.color.rgb = C_WHITE

    # Cuerpo
    for i, r_vals in enumerate(rows, start=1):
        is_hl = highlight_row_idx is not None and (i - 1) == highlight_row_idx
        bg_c = C_RED_SOFT if is_hl else (C_BG_CARD if i % 2 == 0 else C_WHITE)
        for j, val in enumerate(r_vals):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_c
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j <= 1 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            r.font.name = FONT_MAIN
            r.font.size = Pt(font_size)
            r.font.bold = is_hl or (j == 0)
            r.font.color.rgb = C_RED if (is_hl and j == 0) else C_DARK
    return table_shape


# ==============================================================================
# CONSTRUCCIÓN DE LAS 17 DIAPOSITIVAS
# ==============================================================================
def build_presentation() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar datos reales del proyecto
    with (PROCESSED_DIR / "reporte_etl.json").open("r", encoding="utf-8") as f:
        rep_etl = json.load(f)
    with (OUTPUT_DIR / "reporte_modelos.json").open("r", encoding="utf-8") as f:
        rep_mod = json.load(f)
    with (OUTPUT_DIR / "kpis_escenarios.json").open("r", encoding="utf-8") as f:
        kpis_json = json.load(f)

    data = {
        "rep_etl": rep_etl,
        "rep_mod": rep_mod,
        "kpis_json": kpis_json,
        "df_min": pd.read_parquet(PROCESSED_DIR / "vista_minable.parquet"),
        "perfil_raw": pd.read_csv(PROCESSED_DIR / "perfil_estaciones.csv"),
        "perfil_scaled": pd.read_csv(PROCESSED_DIR / "perfil_estaciones_escalado.csv"),
        "codo_sil": pd.read_csv(OUTPUT_DIR / "curva_codo_silhouette.csv"),
        "clusters": pd.read_csv(OUTPUT_DIR / "clusters_estaciones.csv"),
        "radar": pd.read_csv(OUTPUT_DIR / "radar_clusters.csv"),
        "backtest": pd.read_csv(OUTPUT_DIR / "backtest_validacion.csv"),
        "feat_imp": pd.read_csv(OUTPUT_DIR / "feature_importance.csv"),
        "proy_diaria": pd.read_csv(OUTPUT_DIR / "proyecciones_3_escenarios_diarias.csv"),
    }

    print("[1/3] Generando 10 gráficas de alta resolución (300 DPI) desde datos reales...")
    charts = generate_all_charts(data)

    print("[2/3] Renderizando fórmulas matemáticas LaTeX a 300 DPI...")
    formulas = {
        "eda": render_formula_png(
            [
                r"$\mathrm{IQR} = Q_3 - Q_1 \quad\Longrightarrow\quad \mathrm{Rango\ normal} = [\,Q_1 - 1.5\cdot\mathrm{IQR},\;\; Q_3 + 1.5\cdot\mathrm{IQR}\,]$",
                r"$\%\mathrm{Nulos}_j = \frac{n_{\mathrm{NaN},j} + n_{\mathrm{vacio},j}}{N}\times 100\%,\qquad \mathrm{cod\_nodo} = \mathrm{int}(\mathrm{regex}(\mathrm{Estacion},\; \mathrm{patron\ codigo}))$",
            ],
            "formula_eda.png",
            fontsize=13,
            bg_hex="#F8FAFC",
        ),
        "etl_imp": render_formula_png(
            [
                r"$\tilde{x}_{i,j} = \mathrm{Mediana}\!\left(x_{k,j}\mid \mathrm{tipo\_esta}_k=\mathrm{tipo\_esta}_i,\; x_{k,j}>0\right)\;\;(\mathrm{respaldo\ global\ Portales})$",
                r"$Y_{d,h,e} = \sum_{a \in \mathcal{A}_e} \sum_{q \in h} V_{d,e,a,q}\qquad (h \in \{4,5,\dots,23\},\;\; d \in \mathrm{Ene-Oct\ 2026})$",
            ],
            "formula_etl_imp.png",
            fontsize=12.5,
            bg_hex="#F8FAFC",
        ),
        "etl_feat": render_formula_png(
            [
                r"$h_{\sin}=\sin\!\left(\frac{2\pi h}{24}\right),\;\; h_{\cos}=\cos\!\left(\frac{2\pi h}{24}\right),\qquad w_{\sin}=\sin\!\left(\frac{2\pi w}{7}\right),\;\; w_{\cos}=\cos\!\left(\frac{2\pi w}{7}\right)$",
                r"$z_{i,j}=\frac{x_{i,j}-\mu_j}{\sigma_j},\qquad S_{e,h}=\frac{\bar{Y}^{(28d)}_{e,h}}{\mathrm{cap\_art}_e+\mathrm{cap\_biart}_e+1},\qquad R_{e,t}=\frac{\bar{Y}^{(28d)}_{e,t}+1}{\bar{Y}^{(\mathrm{hist})}_{e,t}+1}$",
            ],
            "formula_etl_feat.png",
            fontsize=12.5,
            bg_hex="#F8FAFC",
        ),
        "kmeans": render_formula_png(
            [
                r"$\mathrm{Inercia\ (WCSS)}:\;\; \min_{\mathcal{C}} \sum_{c=1}^{k}\sum_{x_i \in C_c} \|x_i - \mu_c\|_2^2$",
                r"$\mathrm{Silhouette}:\;\; s(i) = \frac{b(i) - a(i)}{\max\{a(i),\,b(i)\}},\qquad \bar{s} = \frac{1}{n}\sum_{i=1}^{n} s(i)$",
                r"$\mathrm{Davies-Bouldin}:\;\; \mathrm{DB}=\frac{1}{k}\sum_{c=1}^{k}\max_{j\neq c}\frac{\sigma_c+\sigma_j}{\|\mu_c-\mu_j\|_2}$",
                r"$\mathrm{PCA\ 2D}:\;\; Z = X W_2\;\;(\Sigma v_r=\lambda_r v_r),\qquad \mathrm{Radar}_{c,d}=\frac{m_{c,d}-\min_c m_{c,d}}{\max_c m_{c,d}-\min_c m_{c,d}}$",
            ],
            "formula_kmeans.png",
            fontsize=12.5,
            bg_hex="#F8FAFC",
        ),
        "xgb": render_formula_png(
            [
                r"$\mathrm{XGBoost}:\;\; \hat{y}_i = \sum_{m=1}^{M} f_m(x_i),\qquad \mathcal{L} = \sum_{i=1}^{n}\frac{1}{2}(y_i-\hat{y}_i)^2 + \sum_{m=1}^{M}\!\left(\gamma T_m + \frac{1}{2}\lambda\|w_m\|_2^2 + \alpha\|w_m\|_1\right)$",
                r"$\mathrm{Ridge\ (Baseline)}:\;\; \min_{\beta}\; \|y - X\beta\|_2^2 + \alpha_{\mathrm{R}}\|\beta\|_2^2\qquad (\alpha_{\mathrm{R}}=10,\; \mathrm{ajustado\ solo\ en\ Train})$",
                r"$\mathrm{MAE}=\frac{1}{n}\sum|y_i-\hat{y}_i|,\quad \mathrm{RMSE}=\sqrt{\frac{1}{n}\sum(y_i-\hat{y}_i)^2},\quad R^2=1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2},\quad \mathrm{WAPE}=\frac{\sum|y_i-\hat{y}_i|}{\sum|y_i|}\times 100\%$",
            ],
            "formula_xgb.png",
            fontsize=12.0,
            bg_hex="#F8FAFC",
        ),
        "scenarios": render_formula_png(
            [
                r"$\mathrm{Perdida\ Pinball\ (Residuos\ } e_i = y_i - \hat{y}_{\mathrm{base},i}\mathrm{)}:\quad \rho_\tau(u) = u\cdot(\tau - \mathbf{1}_{u<0}),\;\;\tau\in\{0.10, 0.90\}$",
                r"$\hat{y}_{\mathrm{opt}}(x) = \max\!\left(\hat{y}_{\mathrm{base}}(x)\cdot\frac{1-0.1250}{1-0.1532}\cdot\sqrt{\frac{97.8}{96.4}} + 0.35\,\Delta^+(x),\;\; \hat{y}_{\mathrm{base}}(x)\right)$",
                r"$\hat{y}_{\mathrm{pes}}(x) = \max\!\left(0,\;\min\!\left(\hat{y}_{\mathrm{base}}(x)\cdot\frac{1-0.1900}{1-0.1532}\cdot\frac{93.5}{96.4} - 0.35\,\Delta^-(x),\;\; \hat{y}_{\mathrm{base}}(x)\right)\right)$",
            ],
            "formula_scenarios.png",
            fontsize=11.8,
            bg_hex="#F8FAFC",
        ),
        "kpis": render_formula_png(
            [
                r"$\mathrm{KPI}_1=\frac{1}{N_d}\sum_{d} Y_d,\qquad \mathrm{KPI}_2:\;\mathrm{ICP}=\frac{\mathrm{Despachos\ Ejecutados}}{\mathrm{Despachos\ Programados}}\times 100\%,\qquad \mathrm{KPI}_4:\;\mathrm{Evasion}=\frac{\mathrm{Accesos}-\mathrm{Validaciones}}{\mathrm{Accesos}}\times 100\%$",
                r"$\mathrm{KPI}_3:\;\bar{W} = 0.608\,W_{\mathrm{pico}} + 0.392\,W_{\mathrm{valle}}\;\;(\mathrm{sensibilizado\ por\ }\frac{\mathrm{ICP}_{\mathrm{base}}}{\mathrm{ICP}_{\mathrm{esc}}}\ \mathrm{y\ saturacion}),\qquad \mathrm{KPI}_5:\;\mathrm{KM}_{\mathrm{esc}}=11.30\times\frac{\mathrm{ICP}_{\mathrm{esc}}}{96.40}$",
            ],
            "formula_kpis.png",
            fontsize=11.5,
            bg_hex="#F8FAFC",
        ),
    }

    # Procesar imagen hero con esquinas redondeadas si existe
    hero_raw = ASSETS_DIR / "hero_transmilenio.png"
    hero_rounded = ASSETS_DIR / "hero_transmilenio_rounded.png"
    if hero_raw.exists():
        round_corners_image(hero_raw, hero_rounded, radius=36, target_ratio=4 / 3)

    print("[3/3] Ensamblando presentación de 17 diapositivas (.pptx)...")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ==========================================================================
    # SLIDE 1 — PORTADA (ESTILO REFERENCIA UNIMINUTO)
    # ==========================================================================
    s1 = prs.slides.add_slide(prs.slide_layouts[6])
    s1.background.fill.solid()
    s1.background.fill.fore_color.rgb = C_WHITE
    add_rect(s1, 0, 0, 13.333, 0.12, C_RED)
    add_rect(s1, 0, 7.38, 13.333, 0.12, C_DARK)

    add_textbox(s1, 0.72, 0.95, 6.2, 0.35, "INGENIERÍA DE SISTEMAS · UNIMINUTO · BOGOTÁ D.C.", font_size=10.5, bold=True, color=C_RED)
    add_textbox(s1, 0.72, 1.42, 6.2, 1.75, "Minería de Datos y\nAnalítica Predictiva en\nTransMilenio S.A.", font_size=34, bold=True, color=C_DARK)
    add_rect(s1, 0.72, 3.45, 1.35, 0.08, C_RED)
    add_textbox(
        s1,
        0.72,
        3.72,
        6.1,
        1.15,
        "Arquitectura End-to-End (Data Lake → ETL → K-Means & XGBoost → Dashboard de 3 Escenarios) aplicada sobre 400,84 millones de validaciones reales del Componente Troncal (Enero – Octubre 2026).",
        font_size=13.5,
        color=C_SLATE,
    )
    add_rect(s1, 0.72, 5.25, 5.95, 0.95, C_BG_CARD, border_color=C_BORDER, rounded=True)
    add_textbox(s1, 0.90, 5.36, 5.6, 0.25, "AUTORES DEL PROYECTO", font_size=8.5, bold=True, color=C_RED)
    add_textbox(s1, 0.90, 5.62, 5.6, 0.45, "Juan José Morales Benítez  ·  Johan Steven Sarmiento Cruz", font_size=13.5, bold=True, color=C_DARK)

    # Panel derecho con imagen redondeada
    add_rect(s1, 7.15, 1.05, 5.55, 5.15, C_BG_CARD, border_color=C_BORDER, rounded=True)
    if hero_rounded.exists():
        add_picture_fit(s1, hero_rounded, 7.30, 1.25, 5.25, 4.05)
    # Insignia cuantitativa inferior derecha
    add_rect(s1, 7.35, 5.38, 5.15, 0.65, C_DARK, rounded=True)
    add_textbox(
        s1,
        7.50,
        5.50,
        4.85,
        0.40,
        "400,84M Validaciones  ·  142 Estaciones  ·  R² = 0,9834  ·  5 KPIs",
        font_size=11.0,
        bold=True,
        color=C_WHITE,
        align=PP_ALIGN.CENTER,
    )

    # ==========================================================================
    # SLIDE 2 — 01 · CONTEXTO ORGANIZACIONAL Y MODELO DE NEGOCIO APP
    # ==========================================================================
    s2 = setup_content_slide(prs, "01 · Contexto Organizacional y Modelo de Negocio", "TransMilenio S.A.: Operación APP y Radiografía Oficial del SITP", 2)

    # Izquierda: Modelo de Negocio APP
    add_rect(s2, 0.55, 1.20, 6.25, 3.65, C_BG_CARD, border_color=C_BORDER, rounded=True)
    add_rect(s2, 0.55, 1.20, 0.10, 3.65, C_RED)
    add_textbox(s2, 0.80, 1.32, 5.8, 0.35, "Modelo de Asociación Público-Privada (APP)", font_size=14.5, bold=True, color=C_DARK)
    add_bullets(
        s2,
        0.80,
        1.75,
        5.8,
        2.95,
        [
            ("Ente Gestor Público:", "TransMilenio S.A. planea rutas, regula frecuencias, controla el Índice de Cumplimiento (ICP) y gestiona la infraestructura troncal."),
            ("27 Concesionarios Privados:", "9 en operación Troncal, 19 en Zonal y 1 en TransMiCable, remunerados según kilómetros comerciales y nivel de servicio."),
            ("Recaudo Centralizado SIRCI / FCS Center:", "Tarjeta TuLlave genera >103 millones de validaciones mensuales en el SITP integrado (49,19% Troncal · 50,81% Zonal)."),
            ("Reto de Sostenibilidad Financiera:", "Equilibrar oferta de flota, mitigar congestión en hora pico y reducir fugas por evasión (> $265.000M COP/año en su pico crítico)."),
        ],
        font_size=11.0,
        space_after=7,
    )

    # Flujo económico de valor abajo izquierda
    add_rect(s2, 0.55, 5.05, 6.25, 1.78, C_WHITE, border_color=C_RED, border_pt=1.2, rounded=True)
    add_textbox(s2, 0.75, 5.16, 5.85, 0.30, "CADENA DE VALOR ANALÍTICA EN EL NEGOCIO", font_size=9.5, bold=True, color=C_RED)
    steps_biz = [
        ("Demanda Validada\n(Tarjeta TuLlave)", C_DARK),
        ("Programación Flota\n(ICP & Km Comerc.)", C_BLUE),
        ("Control Congestión\ny Evasión", C_AMBER),
        ("Sostenibilidad\nFinanciera (FET)", C_GREEN),
    ]
    for idx, (lbl, col) in enumerate(steps_biz):
        bx = 0.75 + idx * 1.48
        add_rect(s2, bx, 5.56, 1.32, 1.05, col, rounded=True)
        add_textbox(s2, bx + 0.04, 5.65, 1.24, 0.88, lbl, font_size=9.0, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER, v_anchor=MSO_ANCHOR.MIDDLE)
        if idx < 3:
            add_textbox(s2, bx + 1.32, 5.88, 0.18, 0.35, "→", font_size=12, bold=True, color=C_SLATE, align=PP_ALIGN.CENTER)

    # Derecha: 4 Tarjetas de Radiografía Oficial SITP (estilo referencia)
    stats_sitp = [
        ("10.494", "Vehículos Vinculados", "Flota total distribuida en troncal, zonal y alimentación.", C_RED),
        ("152 Nodos", "Estaciones y 9 Portales", "Maestro GIS 2026 (142 estaciones activas Ene–Oct) + 7.531 paraderos.", C_BLUE),
        ("114,4 km", "Red Troncal Construida", "Corredores exclusivos BRT más 2.558 km de cobertura zonal.", C_DARK),
        ("1.485", "Buses 100% Eléctricos", "Operando en 9 patios eléctricos de cero emisiones en Bogotá.", C_GREEN),
    ]
    for idx, (big_num, subt, desc, accent) in enumerate(stats_sitp):
        row = idx // 2
        col = idx % 2
        cx = 7.05 + col * 2.95
        cy = 1.20 + row * 2.88
        add_rect(s2, cx, cy, 2.78, 2.68, C_BG_CARD, border_color=C_BORDER, rounded=True)
        add_rect(s2, cx, cy, 2.78, 0.09, accent)
        add_textbox(s2, cx + 0.16, cy + 0.25, 2.46, 0.72, big_num, font_size=28, bold=True, color=accent)
        add_textbox(s2, cx + 0.16, cy + 1.02, 2.46, 0.38, subt, font_size=12.2, bold=True, color=C_DARK)
        add_textbox(s2, cx + 0.16, cy + 1.46, 2.46, 1.05, desc, font_size=10.2, color=C_SLATE)

    # ==========================================================================
    # SLIDE 3 — 01 · DIRECCIONAMIENTO ESTRATÉGICO (MISIÓN, VISIÓN Y 3 OBJETIVOS)
    # ==========================================================================
    s3 = setup_content_slide(prs, "01 · Direccionamiento Estratégico", "Misión, Visión Corporativa y 3 Objetivos del Proyecto Analítico", 3)

    # Misión y Visión
    add_rect(s3, 0.55, 1.25, 5.95, 2.45, C_BG_CARD, border_color=C_BORDER, rounded=True)
    add_rect(s3, 0.55, 1.25, 5.95, 0.08, C_RED)
    add_textbox(s3, 0.80, 1.44, 5.4, 0.25, "PROPÓSITO ACTUAL", font_size=9, bold=True, color=C_RED)
    add_textbox(s3, 0.80, 1.70, 5.4, 0.38, "Misión Institucional", font_size=17, bold=True, color=C_DARK)
    add_textbox(
        s3,
        0.80,
        2.15,
        5.45,
        1.35,
        "Garantizar un servicio de transporte masivo seguro, eficiente, equitativo y sostenible para toda la ciudadanía, consolidándose como eje estructurador de la movilidad intermodal de Bogotá-Región.",
        font_size=11.8,
        color=C_SLATE,
    )

    add_rect(s3, 6.83, 1.25, 5.95, 2.45, C_BG_CARD, border_color=C_BORDER, rounded=True)
    add_rect(s3, 6.83, 1.25, 5.95, 0.08, C_BLUE)
    add_textbox(s3, 7.08, 1.44, 5.4, 0.25, "HORIZONTE TECNOLÓGICO", font_size=9, bold=True, color=C_BLUE)
    add_textbox(s3, 7.08, 1.70, 5.4, 0.38, "Visión Estratégica", font_size=17, bold=True, color=C_DARK)
    add_textbox(
        s3,
        7.08,
        2.15,
        5.45,
        1.35,
        "Ser referente global en movilidad inteligente e integración multimodal (BRT, TransMiCable y Primera Línea del Metro), apalancada en analítica predictiva, datos abiertos y flota cero emisiones.",
        font_size=11.8,
        color=C_SLATE,
    )

    # 3 Objetivos Corporativos
    add_textbox(s3, 0.55, 3.92, 12.2, 0.32, "OBJETIVOS CORPORATIVOS ARTICULADOS CON MINERÍA DE DATOS", font_size=10.5, bold=True, color=C_DARK)
    objs = [
        (
            "01 · OPTIMIZACIÓN OPERATIVA",
            "Despacho Inteligente de Flota",
            "Segmentar las 142 estaciones activas (K-Means) y predecir la demanda horaria (XGBoost) para asignar flota biarticulada en picos reales y sostener ICP ≥ 96,4%.",
            C_RED,
            C_RED_SOFT,
        ),
        (
            "02 · CONTROL DE EVASIÓN",
            "Auditoría Focalizada en Nodos",
            "Identificar estaciones de alta congestión y franjas valle/festivo para focalizar barreras físicas y reducir la evasión de 15,32% hacia la meta Optimista de 12,50%.",
            C_BLUE,
            C_BLUE_SOFT,
        ),
        (
            "03 · PLANEACIÓN Y SOSTENIBILIDAD",
            "Escenarios Tácticos y Km Comerciales",
            "Proyectar 30 días de operación en 3 escenarios cuantiles (Q10, Base L2, Q90) vinculando demanda, espera y 11,3M km/mes de oferta troncal.",
            C_AMBER,
            C_AMBER_SOFT,
        ),
    ]
    for idx, (tag, head, desc, col_b, col_bg) in enumerate(objs):
        ox = 0.55 + idx * 4.18
        add_rect(s3, ox, 4.30, 3.88, 2.52, col_bg, border_color=col_b, border_pt=1.0, rounded=True)
        add_textbox(s3, ox + 0.18, 4.44, 3.52, 0.25, tag, font_size=8.8, bold=True, color=col_b)
        add_textbox(s3, ox + 0.18, 4.72, 3.52, 0.36, head, font_size=13.5, bold=True, color=C_DARK)
        add_textbox(s3, ox + 0.18, 5.14, 3.52, 1.50, desc, font_size=10.8, color=C_SLATE)

    # ==========================================================================
    # SLIDE 4 — 02 · ARQUITECTURA END-TO-END DE 4 FASES
    # ==========================================================================
    s4 = setup_content_slide(prs, "02 · Arquitectura de Ingeniería y Ciencia de Datos", "Pipeline Reproducible de 4 Fases con Volumetría Real Auditada", 4)

    phases = [
        (
            "FASE 1 · DATA LAKE",
            "data/raw/\nsrc/eda_crudo.py",
            C_SLATE,
            [
                "10 libros .xlsx (Ene–Oct 2026, ~138 MB, 15 min).",
                "Maestro GIS ArcGIS: 152 estaciones (.csv / .geojson).",
                "785.250 filas crudas auditadas.",
                "Detección de 6 trampas de calidad.",
            ],
        ),
        (
            "FASE 2 · ETL AUDITABLE",
            "src/etl.py\ndata/processed/",
            C_BLUE,
            [
                "Limpieza en paralelo (10 workers, 107 s).",
                "Homologación 6 códigos → 100% match cod_nodo.",
                "Imputación mediana + 14 festivos + StandardScaler.",
                "400,84M val → 787.951 filas (0 nulos).",
            ],
        ),
        (
            "FASE 3 · MACHINE LEARNING",
            "src/modelo.py\ndata/output/ & models/",
            C_RED,
            [
                "K-Means (k=4, Sil=0,2252) + PCA 2D + Radar.",
                "XGBRegressor cronológico (Train Ene–Ago / Test Sep–Oct).",
                "R²=0,9724 horario · R²=0,9834 diario (WAPE 2,58%).",
                "Proyección Nov 2026 en 3 escenarios (Q10/L2/Q90).",
            ],
        ),
        (
            "FASE 4 · DASHBOARD WEB",
            "dashboard/app.py\nStreamlit Cloud",
            C_GREEN,
            [
                "5 pestañas SaaS + 15 gráficos Plotly sin solapamientos.",
                "Mapa geoespacial CARTO de Bogotá sin token.",
                "Simulador interactivo de 5 KPIs y alertas de cierre.",
                "Repositorio liviano Cloud-ready: solo 2,38 MB.",
            ],
        ),
    ]
    for idx, (ptag, psub, pcol, bullets) in enumerate(phases):
        px_x = 0.55 + idx * 3.12
        add_rect(s4, px_x, 1.25, 2.86, 4.35, C_BG_CARD, border_color=C_BORDER, rounded=True)
        add_rect(s4, px_x, 1.25, 2.86, 0.42, pcol, rounded=False)
        add_textbox(s4, px_x + 0.08, 1.32, 2.70, 0.28, ptag, font_size=10.2, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        add_textbox(s4, px_x + 0.12, 1.76, 2.62, 0.52, psub, font_size=10.5, bold=True, color=pcol, align=PP_ALIGN.CENTER)
        add_rect(s4, px_x + 0.22, 2.34, 2.42, 0.015, C_BORDER)
        add_bullets(s4, px_x + 0.10, 2.46, 2.66, 3.05, [("", b) for b in bullets], font_size=9.8, bullet_color=pcol, space_after=6)
        if idx < 3:
            add_textbox(s4, px_x + 2.86, 3.10, 0.26, 0.45, "➔", font_size=16, bold=True, color=C_MUTED, align=PP_ALIGN.CENTER)

    # Franja inferior de resumen de volumetría y tecnología
    add_rect(s4, 0.55, 5.82, 12.23, 1.02, C_DARK, rounded=True)
    metrics_bar = [
        ("785.250", "Filas crudas Excel"),
        ("400.836.739", "Validaciones limpias"),
        ("787.951 × 26", "Vista Minable (2,02 MB)"),
        ("100,00%", "Match cod_nodo"),
        ("0 Nulos", "Calidad certificada"),
    ]
    for i, (mv, ml) in enumerate(metrics_bar):
        mx = 0.72 + i * 2.40
        add_textbox(s4, mx, 5.92, 2.25, 0.45, mv, font_size=17, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        add_textbox(s4, mx, 6.42, 2.25, 0.30, ml, font_size=9.2, color=RGBColor(0xCB, 0xD5, 0xE1), align=PP_ALIGN.CENTER)

    # ==========================================================================
    # SLIDE 5 — 03 · FASE 1: DATA LAKE, EDA CRUDO Y ANOMALÍAS REALES
    # ==========================================================================
    s5 = setup_content_slide(prs, "03 · Fase 1 — Exploración de Datos Crudos (src/eda_crudo.py)", "Diagnóstico del Data Lake: 6 Trampas Reales Detectadas Antes del ETL", 5)

    add_rect(s5, 0.55, 1.18, 6.65, 5.65, C_BG_CARD, border_color=C_BORDER, rounded=True)
    add_textbox(s5, 0.75, 1.28, 6.25, 0.34, "Hallazgos Empíricos sobre los 10 Libros .xlsx + Maestro GIS", font_size=13.5, bold=True, color=C_DARK)
    add_bullets(
        s5,
        0.75,
        1.70,
        6.25,
        5.0,
        [
            ("1. Matriz pivoteada en ancho (header=6):", "Cada libro tiene 2 hojas ('Validaciones Tullave' y 'Validacion Datos'), columna A vacía, 220.499 filas vacías de plantilla y 10 filas 'Total general' (~50,5M) que inducirían outliers falsos."),
            ("2. Desbordamiento de fechas en meses cortos:", "Plantilla fija de 31 días arrastra fechas al mes siguiente: Febrero trae Mar 1–3; Abril trae May 1; Junio trae Jul 1; Septiembre trae Oct 1."),
            ("3. Homonimia física comprobada:", "'Temporal AV. Jiménez - Inter Eléctricas' aparece 2 veces en el GIS (cod_nodo 9110 en Caracas TZ001 y 14003 en Eje Ambiental TZ015)."),
            ("4. Divergencia comercial en 62 estaciones:", "FCS Center registra '(04000) Cabecera Calle 80' y '(02001) C.C. Santa Fe', mientras ArcGIS usa 'Portal 80' y 'Calle 187'. Cruzar por texto fallaría."),
            ("5. Ceros estructurales en infraestructura:", "17 estaciones (incluidos los 9 Portales) registran ancho_est=0 y area_est=0; 3 estaciones en obra tienen capacidad=0 y Danubio tiene num_vag=0."),
            ("6. Outliers legítimos conservados:", "Picos hasta 4.245 val/15m en Portales y coordenadas occidentales de Soacha/Av. Cali son reales, no errores."),
        ],
        font_size=10.2,
        space_after=7,
    )

    # Derecha: Fórmulas EDA + Mini-tabla de anomalías
    add_rect(s5, 7.42, 1.18, 5.36, 2.38, C_BG_CARD, border_color=C_BLUE, border_pt=1.0, rounded=True)
    add_textbox(s5, 7.60, 1.28, 5.0, 0.30, "FÓRMULAS APLICADAS EN EDA CRUDO", font_size=9.5, bold=True, color=C_BLUE)
    add_picture_fit(s5, formulas["eda"], 7.56, 1.66, 5.08, 1.75)

    add_textbox(s5, 7.42, 3.74, 5.36, 0.30, "RESUMEN DE LLAVES CANDIDATAS EN MAESTRO GIS (N = 152)", font_size=9.5, bold=True, color=C_DARK)
    add_styled_table(
        s5,
        7.42,
        4.08,
        5.36,
        2.74,
        ["Campo", "Tipo CSV", "% Nulos", "Únicos", "Veredicto PK"],
        [
            ["cod_nodo", "int64", "0,00%", "152", "PK Óptima ✓"],
            ["num_est", "str (4–5 dig)", "0,00%", "152", "4 sin cero inicial"],
            ["nom_est", "str", "0,00%", "151", "1 Homónimo ✗"],
            ["created_date", "str", "96,71%", "5", "Descartar ✗"],
            ["last_edited_date", "str", "67,76%", "46", "Descartar ✗"],
            ["esta_oper / observ", "int / str", "0,00%", "1", "Constantes ✗"],
        ],
        col_widths=[1.42, 1.00, 0.78, 0.72, 1.44],
        font_size=8.8,
        highlight_row_idx=0,
    )

    # ==========================================================================
    # SLIDE 6 — 04 · FASE 2: AUDITORÍA ETL, HOMOLOGACIÓN Y BALANCE VOLUMÉTRICO
    # ==========================================================================
    s6 = setup_content_slide(prs, "04 · Fase 2 — Ingeniería ETL (src/etl.py)", "Depuración Estructural, Homologación cod_nodo y Balance de 10 Meses", 6)

    # Arriba: Gráfica dual real de balance ETL + evolución mensual
    add_rect(s6, 0.55, 1.15, 12.23, 3.45, C_WHITE, border_color=C_BORDER, rounded=True)
    add_picture_fit(s6, charts["etl_balance"], 0.72, 1.22, 11.90, 3.30)

    # Abajo izquierda: Tabla de los 6 códigos homologados
    add_textbox(s6, 0.55, 4.72, 7.6, 0.28, "TABLA DE HOMOLOGACIÓN OFICIAL (5.720.812 VALIDACIONES RECUPERADAS → 100,00% MATCH)", font_size=9.2, bold=True, color=C_RED)
    add_styled_table(
        s6,
        0.55,
        5.02,
        7.85,
        1.92,
        ["Código Excel (FCS)", "cod_nodo GIS", "Estación Maestro", "Motivo Operativo"],
        [
            ["(15001) Tibanica - Primavera", "5011", "Tibanica - Primavera", "Extensión Av. Cali (TZ022)"],
            ["(15002) Los Laureles", "5010", "Los Laureles", "Extensión Av. Cali (TZ022)"],
            ["(15003) Islandia", "5009", "Islandia", "Extensión Av. Cali (TZ022)"],
            ["(57503) Ampliacion San Mateo", "7503", "San Mateo - CC Unisur", "Plataforma anexa al nodo"],
            ["(04004) La Granja", "4100", "Granja - Kr 77", "Unificación vagones GIS"],
            ["(50003) Corral Molinos", "9001", "Molinos", "Corral operativo anexo"],
        ],
        col_widths=[2.35, 1.05, 2.05, 2.40],
        font_size=8.3,
    )

    # Abajo derecha: Reglas de alcance y calendario Colombia 2026
    add_rect(s6, 8.60, 4.74, 4.18, 2.20, C_BG_CARD, border_color=C_BORDER, rounded=True)
    add_textbox(s6, 8.76, 4.84, 3.88, 0.28, "REGLAS DE ALCANCE Y CALENDARIO", font_size=9.2, bold=True, color=C_DARK)
    add_bullets(
        s6,
        8.76,
        5.14,
        3.88,
        1.74,
        [
            ("Anti-duplicación:", "6 columnas de fecha desbordada eliminadas antes de melt()."),
            ("Alcance Troncal:", "Excluidos TransMiCable (7,43M) y Dual en calle (43,27M). Incluidos Dual dentro de estación (109,61M)."),
            ("14 Festivos Ley Emiliani:", "Ene 1, Ene 12, Mar 23, Abr 2–3, May 1/18, Jun 8/15/29, Jul 20, Ago 7/17, Oct 12."),
        ],
        font_size=8.9,
        space_after=4,
    )

    # ==========================================================================
    # SLIDE 7 — 04 · FASE 2: FORMULACIÓN MATEMÁTICA DEL ETL Y ESCALAMIENTO
    # ==========================================================================
    s7 = setup_content_slide(prs, "04 · Fase 2 — Formulación Matemática de Transformaciones", "Imputación Jerárquica, Feature Engineering Cíclico y StandardScaler", 7)

    # Izquierda: Fórmulas de Imputación, Agregación, Cíclica y Z-Score
    add_rect(s7, 0.55, 1.18, 6.45, 2.48, C_BG_CARD, border_color=C_BLUE, border_pt=1.0, rounded=True)
    add_textbox(s7, 0.72, 1.26, 6.1, 0.28, "1. IMPUTACIÓN DE CEROS ESTRUCTURALES Y AGREGACIÓN HORA–ESTACIÓN", font_size=9.0, bold=True, color=C_BLUE)
    add_picture_fit(s7, formulas["etl_imp"], 0.68, 1.56, 6.18, 1.48)
    add_textbox(
        s7,
        0.72,
        3.12,
        6.1,
        0.48,
        "Ceros imputados → 0 nulos: ancho_est (17), area_est (17), cap_biart (5), cap_art (3), num_vag (1). Granularidad final: 787.951 filas × 26 columnas.",
        font_size=8.8,
        color=C_SLATE,
    )

    add_rect(s7, 0.55, 3.82, 6.45, 3.05, C_BG_CARD, border_color=C_RED, border_pt=1.0, rounded=True)
    add_textbox(s7, 0.72, 3.92, 6.1, 0.28, "2. CODIFICACIÓN CÍCLICA, ESCALAMIENTO Z-SCORE E ÍNDICE DE SATURACIÓN", font_size=9.0, bold=True, color=C_RED)
    add_picture_fit(s7, formulas["etl_feat"], 0.68, 4.24, 6.18, 1.85)
    add_textbox(
        s7,
        0.72,
        6.18,
        6.1,
        0.62,
        "StandardScaler ajustado sobre 17 variables numéricas de perfil de estación (models/scaler.joblib). Tendencia causal R_(e,t) y saturación S_(e,h) usan únicamente ventana pasada de 28 días.",
        font_size=8.8,
        color=C_SLATE,
    )

    # Derecha: Boxplot Antes vs Después + Esquema de Vista Minable
    add_rect(s7, 7.18, 1.18, 5.60, 3.45, C_WHITE, border_color=C_BORDER, rounded=True)
    add_picture_fit(s7, charts["scaler"], 7.28, 1.24, 5.40, 3.32)

    add_textbox(s7, 7.18, 4.74, 5.60, 0.26, "ESQUEMA CONSOLIDADO DE VISTA_MINABLE.PARQUET (2,02 MB)", font_size=9.0, bold=True, color=C_DARK)
    add_styled_table(
        s7,
        7.18,
        5.02,
        5.60,
        1.85,
        ["Grupo de Variables", "Columnas Clave", "Rol en ML"],
        [
            ["Tiempo / Calendario", "Fecha, Hora_Bloque, Dia_Semana, Mes, Tipo_Dia, Franja", "Índice + Estacionalidad"],
            ["Identidad Espacial", "ID_Estacion (cod_nodo), Nombre_Estacion, Troncal", "Llave JOIN 100% exacta"],
            ["Infraestructura Limpia", "tipo_esta, longitud, latitud, area_est, num_vag, cap_art/biart", "Capacidad y Geometría"],
            ["Variable Objetivo (Y)", "Total_Validaciones (int64, 0 nulos, 400,84M total)", "Target de Regresión"],
        ],
        col_widths=[1.45, 2.70, 1.45],
        font_size=8.2,
    )

    # ==========================================================================
    # SLIDE 8 — 05 · EVIDENCIA HISTÓRICA DE MINERÍA DE DATOS (ENE–OCT 2026)
    # ==========================================================================
    s8 = setup_content_slide(prs, "05 · Evidencia Histórica en Datos Abiertos (Ene–Oct 2026)", "Dinámica Temporal y Mapa de Calor Horario de 400,84M Validaciones", 8)

    add_rect(s8, 0.55, 1.18, 6.08, 4.18, C_WHITE, border_color=C_BORDER, rounded=True)
    add_picture_fit(s8, charts["hist_series"], 0.64, 1.26, 5.90, 4.02)

    add_rect(s8, 6.78, 1.18, 6.00, 4.18, C_WHITE, border_color=C_BORDER, rounded=True)
    add_picture_fit(s8, charts["heatmap"], 6.86, 1.26, 5.84, 4.02)

    # 4 tarjetas de hallazgos reales abajo
    hist_cards = [
        ("1.318.542 / día", "Promedio Global (304 días)", "Día Hábil: 1.614.991 · Sábado: 1.041.270 · Dom/Fest: 476.802.", C_DARK),
        ("60,8% en Hora Pico", "Doble Joroba Hábil", "Pico AM (05–08h) y Pico PM (16–19h) dominan la congestión; Valle aporta 39,2%.", C_RED),
        ("Efecto Estacional", "Vacaciones vs. Semestre", "Caídas nítidas en Enero (35,2M) y Junio (32,3M); máximos en Sep (45,1M) y Oct (44,9M).", C_BLUE),
        ("Obras Metro L1", "Transición en Caracas", "10 estaciones cerradas todo el periodo y 3 cerradas en Oct detectadas automáticamente.", C_AMBER),
    ]
    for idx, (hv, ht, hd, hc) in enumerate(hist_cards):
        hx = 0.55 + idx * 3.10
        add_rect(s8, hx, 5.52, 2.94, 1.38, C_BG_CARD, border_color=C_BORDER, rounded=True)
        add_rect(s8, hx, 5.52, 2.94, 0.07, hc)
        add_textbox(s8, hx + 0.12, 5.64, 2.70, 0.34, hv, font_size=14.5, bold=True, color=hc)
        add_textbox(s8, hx + 0.12, 5.98, 2.70, 0.24, ht, font_size=9.5, bold=True, color=C_DARK)
        add_textbox(s8, hx + 0.12, 6.24, 2.70, 0.62, hd, font_size=8.4, color=C_SLATE)

    # ==========================================================================
    # SLIDE 9 — 06 · FASE 3: CLUSTERING K-MEANS — FÓRMULAS Y SELECCIÓN ÓPTIMA
    # ==========================================================================
    s9 = setup_content_slide(prs, "06 · Fase 3 — Aprendizaje No Supervisado (K-Means + PCA)", "Formulación Matemática del Clustering y Justificación de k = 4", 9)

    add_rect(s9, 0.55, 1.18, 6.35, 3.65, C_BG_CARD, border_color=C_RED, border_pt=1.0, rounded=True)
    add_textbox(s9, 0.74, 1.28, 6.0, 0.28, "ECUACIONES DE AGRUPAMIENTO, VALIDACIÓN Y REDUCCIÓN DIMENSIONAL", font_size=9.2, bold=True, color=C_RED)
    add_picture_fit(s9, formulas["kmeans"], 0.68, 1.62, 6.10, 3.12)

    add_rect(s9, 0.55, 4.98, 6.35, 1.92, C_GREEN_SOFT, border_color=C_GREEN, border_pt=1.0, rounded=True)
    add_textbox(s9, 0.74, 5.08, 6.0, 0.28, "¿POR QUÉ k = 4 Y NO k = 2? (CRITERIO ESTADÍSTICO + NEGOCIO)", font_size=9.2, bold=True, color=C_GREEN)
    add_bullets(
        s9,
        0.74,
        5.38,
        6.0,
        1.45,
        [
            ("k = 2 descartado (Sil = 0,4718):", "Solo aísla los 11 Mega-Portales frente a 131 estaciones heterogéneas (partición binaria trivial sin utilidad táctica)."),
            ("k = 4 óptimo multi-segmento:", "Máximo local de Silhouette entre k=3..8 (0,2252), caída en Davies-Bouldin (1,54 → 1,41), codo de Inercia y PCA 2D explica 49,44% de varianza."),
        ],
        font_size=9.5,
        bullet_color=C_GREEN,
        space_after=4,
    )

    # Derecha: Gráfica real Elbow + Silhouette + tabla compacta k=2..6
    add_rect(s9, 7.08, 1.18, 5.70, 3.78, C_WHITE, border_color=C_BORDER, rounded=True)
    add_picture_fit(s9, charts["elbow_sil"], 7.16, 1.24, 5.54, 3.65)

    add_styled_table(
        s9,
        7.08,
        5.08,
        5.70,
        1.82,
        ["k", "Inercia", "Silhouette", "Calinski-H", "Davies-B"],
        [
            ["2", "1.826,87", "0,4718", "44,99", "0,9352"],
            ["3", "1.506,44", "0,2150", "41,87", "1,5411"],
            ["4 (Óptimo)", "1.290,52", "0,2252", "40,05", "1,4131"],
            ["5", "1.164,30", "0,1796", "36,76", "1,4403"],
            ["6..8", "1.066→942", "0,18→0,19", "34,4→29,9", "1,44→1,40"],
        ],
        col_widths=[1.22, 1.15, 1.12, 1.12, 1.09],
        font_size=8.6,
        highlight_row_idx=2,
    )

    # ==========================================================================
    # SLIDE 10 — 06 · MAPA GEOESPACIAL DE BOGOTÁ Y PERFIL RADAR DE CLÚSTERES
    # ==========================================================================
    s10 = setup_content_slide(prs, "06 · Descubrimiento de Patrones Espaciales y Operativos", "Mapa de las 142 Estaciones en Bogotá y Asimetría Origen–Destino (Radar)", 10)

    # Izquierda: Mapa real Bogotá
    add_rect(s10, 0.55, 1.18, 5.75, 5.72, C_WHITE, border_color=C_BORDER, rounded=True)
    add_picture_fit(s10, charts["bogota_map"], 0.65, 1.24, 5.55, 5.60)

    # Derecha superior: Radar
    add_rect(s10, 6.48, 1.18, 6.30, 3.48, C_WHITE, border_color=C_BORDER, rounded=True)
    add_picture_fit(s10, charts["radar"], 6.62, 1.22, 6.02, 3.38)

    # Derecha inferior: Tabla de los 4 segmentos descubiertos
    add_styled_table(
        s10,
        6.48,
        4.78,
        6.30,
        2.12,
        ["Clúster", "Est.", "Prom. Hábil", "% AM / PM", "Sat. Pico", "Rol Urbano"],
        [
            ["C0 Mega-Portales", "10", "52.366", "51% / 12%", "49,51", "Origen masivo AM"],
            ["C1 Nodos Fuertes", "21", "15.958", "23% / 37%", "10,99", "Atracción laboral PM"],
            ["C2 Intermedias", "72", "7.562", "20% / 38%", "9,03", "Servicios centrales PM"],
            ["C3 Periféricas", "39", "6.804", "50% / 13%", "7,32", "Barrios origen AM"],
        ],
        col_widths=[1.48, 0.50, 0.96, 1.00, 0.78, 1.58],
        font_size=8.4,
        highlight_row_idx=0,
    )

    # ==========================================================================
    # SLIDE 11 — 07 · FASE 3: MODELO PREDICTIVO XGBOOST Y PARTICIÒN CRONOLÓGICA
    # ==========================================================================
    s11 = setup_content_slide(prs, "07 · Fase 3 — Aprendizaje Supervisado (XGBoost Regressor)", "Formulación Matemática, Partición Cronológica y Prevención de Data Leakage", 11)

    # Diagrama temporal de partición cronológica (arriba)
    add_rect(s11, 0.55, 1.18, 12.23, 1.28, C_BG_CARD, border_color=C_BORDER, rounded=True)
    add_textbox(s11, 0.75, 1.24, 11.8, 0.26, "PARTICIÓN ESTRICTAMENTE CRONOLÓGICA (SIN SHUFFLE · CERO FUGA DE DATOS DEL FUTURO)", font_size=9.2, bold=True, color=C_DARK)

    add_rect(s11, 0.75, 1.58, 7.15, 0.72, C_DARK, rounded=True)
    add_textbox(s11, 0.85, 1.66, 6.95, 0.55, "TRAIN (8 Meses: 2026-01-01 → 2026-08-31) · 618.494 filas\nMes 1 semilla causal; cada mes m solo usa historia < m (early stopping en Ago)", font_size=9.2, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    add_rect(s11, 8.05, 1.58, 2.75, 0.72, C_RED, rounded=True)
    add_textbox(s11, 8.12, 1.66, 2.60, 0.55, "TEST (Sep–Oct 2026)\n169.457 filas · Corte ≤ Ago 31", font_size=9.2, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    add_rect(s11, 10.95, 1.58, 1.68, 0.72, C_GREEN, rounded=True)
    add_textbox(s11, 11.00, 1.66, 1.58, 0.55, "PROYECCIÓN\nNov 2026 (30d)", font_size=9.2, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    # Bloque de fórmulas XGBoost + Ridge + Métricas
    add_rect(s11, 0.55, 2.62, 12.23, 2.68, C_BG_CARD, border_color=C_RED, border_pt=1.0, rounded=True)
    add_textbox(s11, 0.75, 2.70, 11.8, 0.28, "ECUACIONES DEL MODELO SUPERVISADO, REGULARIZACIÓN L1/L2 Y MÉTRICAS DE VALIDACIÓN", font_size=9.2, bold=True, color=C_RED)
    add_picture_fit(s11, formulas["xgb"], 0.72, 3.02, 11.85, 2.18)

    # Hiperparámetros y arquitectura de 33 features abajo
    hp_cards = [
        ("33 Features Causales", "Codificación cíclica (sin/cos), quincena, Tipo_Dia, Franja, infraestructura física, promedio histórico y ventana reciente de 28 días.", C_BLUE),
        ("Hiperparámetros XGBoost", "tree_method='hist', lr=0,05, max_depth=8, min_child_weight=10, subsample=0,85, L1 α=1,0, L2 λ=5,0, best_iter=77, random_state=42.", C_DARK),
        ("Regla de Cierres Metro L1", "Estaciones con actividad cero en los últimos 28 días previos al corte (obras Caracas) se fijan en 0 sin predecir demanda fantasma.", C_AMBER),
    ]
    for idx, (ht, hd, hc) in enumerate(hp_cards):
        cx = 0.55 + idx * 4.15
        add_rect(s11, cx, 5.46, 3.93, 1.44, C_WHITE, border_color=hc, border_pt=1.0, rounded=True)
        add_textbox(s11, cx + 0.14, 5.56, 3.65, 0.28, ht, font_size=10.5, bold=True, color=hc)
        add_textbox(s11, cx + 0.14, 5.88, 3.65, 0.94, hd, font_size=9.2, color=C_SLATE)

    # ==========================================================================
    # SLIDE 12 — 07 · RESULTADOS DE VALIDACIÓN EN TEST (SEP–OCT 2026)
    # ==========================================================================
    s12 = setup_content_slide(prs, "07 · Validación Out-of-Sample (Test: Septiembre – Octubre 2026)", "XGBoost Supera a los Baselines: R² = 0,9834 y -61,2% de Error Diario", 12)

    # Arriba: Tabla formal comparativa
    add_styled_table(
        s12,
        0.55,
        1.16,
        12.23,
        2.05,
        ["Granularidad de Evaluación (Test Sep–Oct)", "Modelo Evaluado", "MAE", "RMSE", "R²", "WAPE (%)", "MAPE y≥10 (%)"],
        [
            ["Hora–Estación (169.457 obs.)", "Baseline Promedio Histórico", "80,69", "185,62", "0,9660", "15,19%", "34,49%"],
            ["Hora–Estación (169.457 obs.)", "Baseline Ridge Lineal (α=10)", "94,62", "193,86", "0,9629", "17,81%", "57,80%"],
            ["Hora–Estación (169.457 obs.)", "XGBRegressor (hist) — Ganador", "68,98", "167,34", "0,9724", "12,99%", "29,80%"],
            ["Diario–Sistema (61 días)", "Baseline Promedio Histórico", "98.164", "108.236", "0,9405", "6,65%", "6,87%"],
            ["Diario–Sistema (61 días)", "Baseline Ridge Lineal (α=10)", "43.141", "63.850", "0,9793", "2,92%", "3,32%"],
            ["Diario–Sistema (61 días)", "XGBRegressor (hist) — Ganador", "38.074", "57.218", "0,9834", "2,58%", "2,76%"],
        ],
        col_widths=[3.05, 2.95, 1.15, 1.25, 1.05, 1.28, 1.50],
        font_size=8.8,
        highlight_row_idx=5,
    )

    # Abajo: 2 gráficos reales lado a lado (Backtest + Feature Importance)
    add_rect(s12, 0.55, 3.36, 6.22, 3.55, C_WHITE, border_color=C_BORDER, rounded=True)
    add_picture_fit(s12, charts["backtest"], 0.62, 3.44, 6.08, 3.40)

    add_rect(s12, 6.95, 3.36, 5.83, 3.55, C_WHITE, border_color=C_BORDER, rounded=True)
    add_picture_fit(s12, charts["feat_imp"], 7.03, 3.44, 5.67, 3.40)

    # ==========================================================================
    # SLIDE 13 — 08 · PROYECCIONES NOVIEMBRE 2026 EN 3 ESCENARIOS (Q10 / L2 / Q90)
    # ==========================================================================
    s13 = setup_content_slide(prs, "08 · Simulador de Proyecciones a 30 Días (Noviembre 2026)", "Regresión Cuantil Residual (Q10 / Base L2 / Q90) y Drivers Operativos", 13)

    # Izquierda: Fórmulas de escenarios + tarjetas numéricas
    add_rect(s13, 0.55, 1.18, 5.95, 2.85, C_BG_CARD, border_color=C_BLUE, border_pt=1.0, rounded=True)
    add_textbox(s13, 0.72, 1.26, 5.65, 0.28, "FORMULACIÓN CUANTIL RESIDUAL Y COHERENCIA MONOTÓNICA", font_size=8.8, bold=True, color=C_BLUE)
    add_picture_fit(s13, formulas["scenarios"], 0.66, 1.56, 5.75, 2.38)

    # 3 Tarjetas de escenarios Nov 2026
    sc_cards = [
        ("PESIMISTA (Q10 + Obras)", "37,99 M", "1.266.415 val/día (-9,69% vs Base) · ICP 93,5% · Evasión 19,0%", C_AMBER, C_AMBER_SOFT),
        ("BASE (Central L2)", "42,07 M", "1.402.299 val/día (19 Hábiles, 4 Sáb, 7 Dom/Fest) · ICP 96,4%", C_RED, C_RED_SOFT),
        ("OPTIMISTA (Q90 + Evasión)", "44,70 M", "1.490.009 val/día (+6,25% vs Base) · ICP 97,8% · Evasión 12,5%", C_GREEN, C_GREEN_SOFT),
    ]
    for idx, (st_lbl, st_val, st_desc, sc_c, sc_bg) in enumerate(sc_cards):
        sy = 4.18 + idx * 0.92
        add_rect(s13, 0.55, sy, 5.95, 0.82, sc_bg, border_color=sc_c, border_pt=1.0, rounded=True)
        add_textbox(s13, 0.72, sy + 0.06, 2.45, 0.26, st_lbl, font_size=8.6, bold=True, color=sc_c)
        add_textbox(s13, 0.72, sy + 0.34, 5.65, 0.40, st_desc, font_size=8.8, color=C_SLATE)
        add_textbox(s13, 4.45, sy + 0.04, 1.90, 0.36, st_val, font_size=17.5, bold=True, color=sc_c, align=PP_ALIGN.RIGHT)

    # Derecha: Gráfico real 3 Escenarios Nov 2026
    add_rect(s13, 6.68, 1.18, 6.10, 4.45, C_WHITE, border_color=C_BORDER, rounded=True)
    add_picture_fit(s13, charts["scenarios_nov"], 6.76, 1.26, 5.94, 4.30)

    add_rect(s13, 6.68, 5.76, 6.10, 1.16, C_DARK, rounded=True)
    add_textbox(
        s13,
        6.86,
        5.86,
        5.75,
        0.96,
        "Calendario Noviembre 2026 (reentrenado con 100% Ene–Oct):\n• 19 días hábiles · 4 sábados · 7 domingos/festivos (incluye 2 Nov Todos los Santos y 16 Nov Independencia de Cartagena).\n• Garantía estricta verificada: Optimista ≥ Base ≥ Pesimista en el 100% de días y estaciones.",
        font_size=8.9,
        color=C_WHITE,
    )

    # ==========================================================================
    # SLIDE 14 — 09 · LOS 5 KPIS CORPORATIVOS: FÓRMULAS Y MATRIZ POR ESCENARIO
    # ==========================================================================
    s14 = setup_content_slide(prs, "09 · Consolidado de Indicadores de Gestión (KPIs Corporativos)", "Fórmulas y Matriz Sensibilizada de los 5 KPIs por Escenario", 14)

    add_rect(s14, 0.55, 1.16, 12.23, 1.85, C_BG_CARD, border_color=C_RED, border_pt=1.0, rounded=True)
    add_textbox(s14, 0.74, 1.24, 11.8, 0.26, "ECUACIONES OFICIALES Y SENSIBILIZACIÓN FÍSICA DE LOS 5 KPIS", font_size=9.0, bold=True, color=C_RED)
    add_picture_fit(s14, formulas["kpis"], 0.72, 1.54, 11.85, 1.38)

    add_styled_table(
        s14,
        0.55,
        3.18,
        12.23,
        2.78,
        ["KPI Corporativo", "Fuente del Dato", "Histórico Ene–Oct", "Pesimista Nov", "Base Nov", "Optimista Nov"],
        [
            ["1. Demanda Diaria Validada", "Observado Datos Abiertos + ML", "1.318.542 / día", "1.266.415 / día", "1.402.299 / día", "1.490.009 / día"],
            ["2. Cumplimiento Program. (ICP)", "Paramétrico Oficial Sensib.", "96,40%", "93,50%", "96,40%", "97,80%"],
            ["3. Tiempo Medio Espera (Pond.)", "Paramétrico Oficial Sensib.", "4,95 min\n(3,50P / 7,20V)", "5,73 min\n(4,19P / 8,11V)", "4,96 min\n(3,52P / 7,20V)", "4,63 min\n(3,26P / 6,76V)"],
            ["4. Tasa Evasión Focalizada", "Paramétrico Oficial Sensib.", "15,32%\n(hist. 28,51%)", "19,00%", "15,32%", "12,50%"],
            ["5. Km Comerciales Troncal", "Paramétrico Oficial Sensib.", "11,300 M km", "10,960 M km", "11,300 M km", "11,464 M km"],
        ],
        col_widths=[2.85, 2.58, 1.75, 1.65, 1.65, 1.75],
        font_size=9.0,
        highlight_row_idx=0,
    )

    add_rect(s14, 0.55, 6.12, 12.23, 0.78, C_GREEN_SOFT, border_color=C_GREEN, border_pt=0.9, rounded=True)
    add_textbox(
        s14,
        0.74,
        6.22,
        11.85,
        0.60,
        "Transparencia Metodológica: Se separa explícitamente el KPI 1 (calculado 100% sobre 400,84M de registros observados de Datos Abiertos y XGBoost) de los KPIs 2–5 (parámetros oficiales de gestión de TransMilenio S.A. sensibilizados con ecuaciones físicas de cumplimiento de flota y saturación).",
        font_size=9.5,
        color=C_DARK,
    )

    # ==========================================================================
    # SLIDE 15 — 10 · FASE 4: DASHBOARD WEB INTERACTIVO (STREAMLIT + PLOTLY)
    # ==========================================================================
    s15 = setup_content_slide(prs, "10 · Fase 4 — Producto Analítico Web (dashboard/app.py)", "Plataforma Interactiva en Streamlit + Plotly Lista para Despliegue Cloud", 15)

    tabs_desc = [
        ("Pestaña 1", "Contexto y Arquitectura", "Modelo APP, Misión/Visión, 3 objetivos y trazabilidad volumétrica real de 4 fases.", C_DARK),
        ("Pestaña 2", "Auditoría ETL", "Balance de depuración, 6 homologaciones, 14 festivos, boxplots StandardScaler y explorador Parquet.", C_BLUE),
        ("Pestaña 3", "KPIs e Histórico", "5 KPIs con popover de fórmulas, serie diaria, Heatmap, alerta automática de cierres Metro L1 y Top 15.", C_RED),
        ("Pestaña 4", "Clústeres y Mapa", "Mapa geoespacial CARTO de Bogotá (sin token), gráfico Radar 0–1, Codo+Silhouette y PCA 2D.", C_GREEN),
        ("Pestaña 5", "XGBoost y Escenarios", "Backtest Sep–Oct, Feature Importance, simulador de 3 curvas Nov 2026 y ranking proyectado.", C_AMBER),
    ]
    for idx, (tnum, ttit, tdesc, tcol) in enumerate(tabs_desc):
        tx = 0.55 + idx * 2.48
        add_rect(s15, tx, 1.25, 2.32, 3.05, C_BG_CARD, border_color=C_BORDER, rounded=True)
        add_rect(s15, tx, 1.25, 2.32, 0.36, tcol)
        add_textbox(s15, tx + 0.06, 1.30, 2.20, 0.26, tnum.upper(), font_size=9.2, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        add_textbox(s15, tx + 0.10, 1.72, 2.12, 0.44, ttit, font_size=11.5, bold=True, color=C_DARK, align=PP_ALIGN.CENTER)
        add_textbox(s15, tx + 0.12, 2.24, 2.08, 1.90, tdesc, font_size=9.5, color=C_SLATE, align=PP_ALIGN.LEFT)

    # Bloque inferior de Ingeniería Software & Cloud Readiness
    add_rect(s15, 0.55, 4.55, 6.20, 2.32, C_WHITE, border_color=C_DARK, border_pt=1.0, rounded=True)
    add_textbox(s15, 0.75, 4.66, 5.8, 0.30, "ARQUITECTURA DE RENDIMIENTO Y UX", font_size=9.5, bold=True, color=C_RED)
    add_bullets(
        s15,
        0.75,
        5.00,
        5.8,
        1.80,
        [
            ("Carga Ultrarrápida:", "@st.cache_data consume exclusivamente vista_minable.parquet (2,02 MB) y salidas livianas en data/output/ (~125 KB)."),
            ("UX Ejecutiva:", "Pestañas tipo pill, sidebar dividido en 2 bloques con st.pills neutros, botón '↺ Restablecer filtros' y barra de vista activa."),
            ("Calidad Verificada:", "Suite AppTest + py_compile -W error + HTTP /_stcore/health → 0 excepciones y 0 advertencias."),
        ],
        font_size=9.6,
        space_after=5,
    )

    add_rect(s15, 6.95, 4.55, 5.83, 2.32, C_DARK, rounded=True)
    add_textbox(s15, 7.18, 4.68, 5.4, 0.30, "STACK TECNOLÓGICO Y DESPLIEGUE EN LA NUBE", font_size=9.5, bold=True, color=RGBColor(0xF8, 0x71, 0x71))
    add_textbox(
        s15,
        7.18,
        5.06,
        5.4,
        1.65,
        "• Lenguaje y Motor: Python 3.12 · Pandas · NumPy · PyArrow\n"
        "• Machine Learning: Scikit-learn (K-Means, PCA, Ridge, StandardScaler) · XGBoost 3.4 (hist + quantileerror)\n"
        "• Visualización Web: Streamlit 1.67 + Plotly 7.1 (15 gráficas)\n"
        "• Tamaño publicable en GitHub / Streamlit Cloud: 2,38 MB (21 archivos)\n"
        "• Ejecución local:  streamlit run dashboard/app.py",
        font_size=9.8,
        color=C_WHITE,
    )

    # ==========================================================================
    # SLIDE 16 — 11 · HALLAZGOS ESTRATÉGICOS Y RECOMENDACIONES DE NEGOCIO
    # ==========================================================================
    s16 = setup_content_slide(prs, "11 · Conclusiones de Negocio y Recomendaciones Accionables", "Decisiones Estratégicas Respaldadas por la Minería de Datos", 16)

    recs = [
        (
            "01 · FOCALIZACIÓN EN CLÚSTER C0 (10 MEGA-PORTALES)",
            "Priorizar regulación y flota biarticulada en Pico AM (05:30–07:30)",
            "Los 10 nodos de C0 promedian 52.366 val/día hábil con índice de saturación pico de 49,51 (4,5× superior a C1) y 51,3% de demanda concentrada en la mañana. Intervenir solo el 7% de las estaciones impacta el principal cuello de botella de la ciudad.",
            C_RED,
        ),
        (
            "02 · REBALANCEO DINÁMICO POR ASIMETRÍA ORIGEN–DESTINO",
            "Acoplar oferta entre generadoras AM (C0/C3) y atractoras PM (C1/C2)",
            "El análisis Radar demostró dos regímenes opuestos: periferia/portales (~50% AM) frente a nodos laborales/universitarios (~37% PM). El pronóstico horario XGBoost (WAPE 2,58%) permite reprogramar despachos en vacío con anticipación semanal.",
            C_BLUE,
        ),
        (
            "03 · MONITOREO EN VIVO DE OBRAS METRO L1 Y EXPANSIÓN AV. CALI",
            "Integración continua de alertas de cierre y nuevos corredores",
            "El pipeline detectó automáticamente 13 estaciones de Caracas impactadas por obras del Metro y rescató 5,72M validaciones mediante homologación de nodos nuevos (Tibanica, Los Laureles, Islandia), evitando sesgos en la planeación.",
            C_AMBER,
        ),
        (
            "04 · RETORNO FINANCIERO DEL CONTROL DE EVASIÓN FOCALIZADA",
            "+2,63 millones de validaciones/mes en el Escenario Optimista",
            "Pasar de 15,32% a 12,50% de evasión con ICP de 97,8% eleva la proyección de noviembre de 42,07M (Base) a 44,70M (+6,25%), reduciendo el tiempo medio de espera a 4,63 min y fortaleciendo el Fondo de Estabilización Tarifaria (FET).",
            C_GREEN,
        ),
    ]
    for idx, (rtag, rtit, rdesc, rcol) in enumerate(recs):
        row = idx // 2
        col = idx % 2
        rx = 0.55 + col * 6.22
        ry = 1.25 + row * 2.86
        add_rect(s16, rx, ry, 6.00, 2.66, C_BG_CARD, border_color=C_BORDER, rounded=True)
        add_rect(s16, rx, ry, 0.11, 2.66, rcol)
        add_textbox(s16, rx + 0.25, ry + 0.16, 5.55, 0.25, rtag, font_size=9.0, bold=True, color=rcol)
        add_textbox(s16, rx + 0.25, ry + 0.44, 5.55, 0.42, rtit, font_size=13.0, bold=True, color=C_DARK)
        add_textbox(s16, rx + 0.25, ry + 0.94, 5.55, 1.58, rdesc, font_size=10.3, color=C_SLATE)

    # ==========================================================================
    # SLIDE 17 — CIERRE / DEMOSTRACIÓN EN VIVO (ESTILO REFERENCIA UNIMINUTO)
    # ==========================================================================
    s17 = prs.slides.add_slide(prs.slide_layouts[6])
    s17.background.fill.solid()
    s17.background.fill.fore_color.rgb = C_DARK
    add_rect(s17, 0, 0, 13.333, 0.12, C_RED)

    add_textbox(s17, 0.80, 1.05, 8.0, 0.32, "MINERÍA DE DATOS · CLUSTERING K-MEANS · PRONÓSTICO XGBOOST · 3 ESCENARIOS", font_size=10.0, bold=True, color=RGBColor(0xF8, 0x71, 0x71))
    add_textbox(s17, 0.80, 1.50, 7.6, 1.35, "Gracias por su atención\nDemostración en Vivo", font_size=36, bold=True, color=C_WHITE)
    add_rect(s17, 0.80, 3.05, 1.45, 0.08, C_RED)
    add_textbox(
        s17,
        0.80,
        3.35,
        7.2,
        1.25,
        "Pipeline 100% reproducible desde Datos Abiertos de TransMilenio S.A. hasta tablero web interactivo con trazabilidad técnica, cero nulos y precisión diaria del 97,42% (WAPE 2,58%).",
        font_size=13.5,
        color=RGBColor(0xCB, 0xD5, 0xE1),
    )

    add_rect(s17, 0.80, 4.90, 7.0, 1.35, RGBColor(0x1E, 0x29, 0x3B), border_color=RGBColor(0x33, 0x41, 0x55), rounded=True)
    add_textbox(s17, 1.02, 5.04, 6.5, 0.34, "Juan José Morales Benítez  ·  Johan Steven Sarmiento Cruz", font_size=14.0, bold=True, color=C_WHITE)
    add_textbox(s17, 1.02, 5.42, 6.5, 0.30, "Programa de Ingeniería de Sistemas — Corporación Universitaria Minuto de Dios (UNIMINUTO)", font_size=10.5, color=RGBColor(0x94, 0xA3, 0xB8))
    add_textbox(s17, 1.02, 5.78, 6.5, 0.32, "Demo local:  streamlit run dashboard/app.py   |   Bogotá D.C., 2026", font_size=10.5, bold=True, color=RGBColor(0x38, 0xBD, 0xF8))

    if hero_rounded.exists():
        add_picture_fit(s17, hero_rounded, 8.25, 1.55, 4.40, 3.65)
        add_rect(s17, 8.25, 5.25, 4.40, 0.98, C_RED, rounded=True)
        add_textbox(
            s17,
            8.35,
            5.38,
            4.20,
            0.72,
            "400,84M Validaciones Analizadas\n4 Clústeres · R² = 0,9834 · 3 Escenarios",
            font_size=11.5,
            bold=True,
            color=C_WHITE,
            align=PP_ALIGN.CENTER,
        )

    prs.save(str(PPTX_OUT))
    size_mb = PPTX_OUT.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Presentación generada exitosamente en:\n     {PPTX_OUT} ({size_mb:.2f} MB | {len(prs.slides)} diapositivas)")


if __name__ == "__main__":
    build_presentation()
