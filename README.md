# Analítica Avanzada y Minería de Datos — TransMilenio S.A. (2026)

Proyecto universitario end-to-end de **Ingeniería y Minería de Datos** sobre el componente **Troncal de TransMilenio S.A.** en Bogotá D.C., construido a partir de registros oficiales del portal [Datos Abiertos TransMilenio](https://datosabiertos-transmilenio.hub.arcgis.com/) (`400.84 millones` de validaciones limpias entre enero y octubre de 2026 en `142` estaciones activas).

---

## Arquitectura del Proyecto (4 Fases)

```text
Mineria_Datos/
├── .streamlit/
│   └── config.toml                          # Identidad visual TransMilenio (#D31F26 / #1E293B)
├── data/
│   ├── raw/                                 # Fase 1: Data Lake crudo (10 libros .xlsx 15m + maestro GIS)
│   ├── processed/                           # Fase 2: Vista minable (.parquet/.csv), perfiles y reporte_etl.json
│   └── output/                              # Fase 3: Clústeres, backtest, proyecciones Nov 2026 y 5 KPIs
├── models/                                  # Escalador StandardScaler, K-Means, PCA, Ridge y XGBoost (L2/Q90/Q10)
├── src/
│   ├── eda_crudo.py                         # Fase 1: Perfilamiento de calidad, nulos, outliers y llaves de cruce
│   ├── etl.py                               # Fase 2: Limpieza, homologación cod_nodo, imputación y escalamiento
│   └── modelo.py                            # Fase 3: K-Means (k=4), XGBoost cronológico y 3 escenarios Nov 2026
├── dashboard/
│   └── app.py                               # Fase 4: Aplicación interactiva en Streamlit + Plotly (5 pestañas)
└── requirements.txt                         # Dependencias listas para ejecución local y Streamlit Community Cloud
```

---

## Resultados Principales

1. **Fase 2 — ETL Auditable (`src/etl.py`):**
   * Depuración de `785,250` filas crudas (`220,499` filas vacías de plantilla, `10` filas `'Total general'` y `6` columnas de fechas desbordadas en Feb/Abr/Jun/Sep eliminadas antes de `melt()`).
   * **Match del `100.00%`** por llave entera `cod_nodo` (`ID_Estacion`) rescatando `5.72M` validaciones con homologación de 6 códigos y `62` estaciones con cambio de nombre comercial.
   * Consolidación de `vista_minable.parquet` (`787,951 filas × 26 columnas`, `2.02 MB`, **`0` valores nulos**).
2. **Fase 3 — Machine Learning y Escenarios (`src/modelo.py`):**
   * **Clustering K-Means ($k = 4$, `Silhouette = 0.2252`):** identificación de Mega-Portales (`C0`), Nodos Troncales Fuertes (`C1`), Estaciones Intermedias Atractoras PM (`C2`) y Periféricas Generadoras AM (`C3`).
   * **Pronóstico Cronológico `XGBRegressor` (Train Ene–Ago → Test Sep–Oct 2026):** $R^2 = 0.9724$ (`WAPE = 12.99%`) a nivel Hora–Estación y $R^2 = 0.9834$ (`WAPE = 2.58%`) a nivel Diario–Sistema, reduciendo el error diario en `61.2%` frente al promedio histórico.
   * **Proyección Noviembre 2026 (30 días, 3 escenarios):** Pesimista (`37.99M`), Base (`42.07M`) y Optimista (`44.70M`) integrados a los **5 KPIs Corporativos**.

---

## Ejecución Local

1. **Activar entorno virtual e instalar dependencias:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\pip install -r requirements.txt
   ```

2. **Reproducir el pipeline de datos y modelos (opcional si ya existen `data/processed/` y `data/output/`):**
   ```powershell
   .\.venv\Scripts\python src/eda_crudo.py
   .\.venv\Scripts\python src/etl.py
   .\.venv\Scripts\python src/modelo.py
   ```

3. **Iniciar el Dashboard Interactivo en el navegador:**
   ```powershell
   .\.venv\Scripts\streamlit run dashboard/app.py
   ```

---

## Despliegue en Streamlit Community Cloud

1. Subir el repositorio a GitHub (gracias a `.gitignore`, solo se versionan los artefactos livianos `.parquet`, `.json` y `data/output/*.csv` con menos de `2.5 MB` en total).
2. En [share.streamlit.io](https://share.streamlit.io/), seleccionar el repositorio, rama `main` y definir como *Main file path*: `dashboard/app.py`.
