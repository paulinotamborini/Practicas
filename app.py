import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.spatial import ConvexHull
import folium
from streamlit_folium import folium_static
import pyproj
from shapely.geometry import Polygon

st.set_page_config(page_title="Predicción de baches", layout="wide")

st.title("🔧 Predicción de zonas con muchos baches")

# Cargar modelo XGBoost
modelo = xgb.XGBClassifier()
modelo.load_model("modelo_xgb_montecarlo.json")


def crear_poligonos_por_zona(df_coord):
    zonas = df_coord['Zona'].unique()
    poligonos = []

    # Transformador de EPSG:22185 (metros) → EPSG:4326 (lat/lon)
    transformer = pyproj.Transformer.from_crs("EPSG:22185", "EPSG:4326", always_xy=True)

    for z in zonas:
        puntos = df_coord[df_coord['Zona'] == z][['coordenada_x', 'coordenada_y']].values
        if len(puntos) >= 3:
            try:
                hull = ConvexHull(puntos)
                vertices = puntos[hull.vertices]
                # Transformar a lat/lon y reordenar a (lat, lon)
                vertices_latlon = [transformer.transform(x, y)[::-1] for x, y in vertices]
                poligonos.append({'Zona': z, 'vertices': vertices_latlon})
            except:
                continue
    return poligonos



# Cargar umbral 
with open("umbral_optimo.txt", "r") as f:
    umbral = float(f.read())

st.markdown(f"**Umbral de clasificación usado:** `{umbral}`")

# Columnas requeridas
features = ['año', 'mes', 'dia', 'Zona',
            'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
            'precip_lag_1', 'precip_sum_3d', 'precip_sum_7d',
            'temp_max_3d', 'temp_min_3d',
            'baches_lag_1', 'baches_sum_3d',
            'llovio', 'dias_lluvia_3d']

# Botón para descargar plantilla
st.subheader(":arrow_down: Descargá la plantilla de columnas necesarias")
plantilla = pd.DataFrame(columns=features)
st.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False), file_name="plantilla_baches.csv")

# Subida del CSV real
st.subheader(":page_facing_up: Subí tu archivo CSV para hacer predicciones")
archivo_csv = st.file_uploader("Seleccioná un archivo .csv con datos de zonas", type="csv")

if archivo_csv is not None:
    try:
        df = pd.read_csv(archivo_csv)
        st.write("Vista previa de los datos cargados:")
        st.dataframe(df.head())

        columnas_necesarias = set(features)
        columnas_usuario = set(df.columns)

        faltantes = columnas_necesarias - columnas_usuario
        extra = columnas_usuario - columnas_necesarias

        if faltantes:
            st.error(f"Faltan columnas necesarias: {faltantes}")
        else:
            if extra:
                st.warning(f"Estas columnas extra se ignorarán: {extra}")
            df = df[features]  # Reordenamos y filtramos

            if st.button("Predecir zonas con baches"):
                proba = modelo.predict_proba(df)[:, 1]
                pred = (proba > umbral).astype(int)
                df_resultado = df.copy()
                df_resultado["Probabilidad_baches"] = proba
                df_resultado["Prediccion"] = pred
                st.session_state.df_resultado = df_resultado

        if "df_resultado" in st.session_state:
            df_resultado = st.session_state.df_resultado

            # Filtros interactivos
            st.subheader("🔍 Filtros")
            zonas_unicas = sorted(df_resultado["Zona"].unique())
            meses_unicos = sorted(df_resultado["mes"].unique())

            zona_seleccionada = st.multiselect("Filtrar por zona:", zonas_unicas, default=zonas_unicas)
            mes_seleccionado = st.multiselect("Filtrar por mes:", meses_unicos, default=meses_unicos)

            df_filtrado = df_resultado[
                (df_resultado["Zona"].isin(zona_seleccionada)) &
                (df_resultado["mes"].isin(mes_seleccionado))
            ]

            st.dataframe(df_filtrado.head(20))

            # Informe
            st.subheader("Análisis de resultados")
            total = len(df_filtrado)
            positivos = df_filtrado["Prediccion"].sum()
            porcentaje = positivos / total * 100 if total > 0 else 0

            st.markdown(f"- Zonas con riesgo alto de baches: **{positivos}** de **{total}** ({porcentaje:.1f}%)")

            st.markdown("### :triangular_flag_on_post: Top 5 zonas con mayor probabilidad de baches")
            top5 = df_filtrado.sort_values("Probabilidad_baches", ascending=False).head(5)
            st.dataframe(top5[["Zona", "Probabilidad_baches"]])

            # 🔶 Descargar CSV
            st.markdown("### 🔶 Descargar CSV con resultados")
            csv_result = df_filtrado.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar resultados", data=csv_result, file_name="predicciones_baches_filtradas.csv", mime="text/csv")

            # 📈 Gráficos
            st.markdown("### 📈 Visualización de datos")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Distribución de probabilidades")
                fig1, ax1 = plt.subplots(figsize=(4, 3))
                sns.histplot(df_filtrado["Probabilidad_baches"], bins=20, kde=True, ax=ax1)
                ax1.set_xlabel("Probabilidad de muchos baches")
                ax1.set_ylabel("Cantidad de zonas")
                st.pyplot(fig1)

            with col2:
                st.subheader("Conteo por clase predicha")
                fig2, ax2 = plt.subplots(figsize=(4, 3))
                sns.countplot(x="Prediccion", data=df_filtrado, ax=ax2)
                ax2.set_xlabel("Predicción (0 = pocos baches, 1 = muchos baches)")
                ax2.set_ylabel("Cantidad")
                st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
else:
    st.info("Esperando que subas un archivo CSV con los datos de las zonas.")


# ========================
# 🗺️ Mapa de zonas geográficas
# ========================
st.subheader("🗺️ Mapa interactivo de zonas geográficas")

coords = pd.read_csv("DF_COORDENADAS.csv")
zonas_poligonos = crear_poligonos_por_zona(coords)

mapa = folium.Map(location=[-32.95, -60.66], zoom_start=12)

colores = [
    '#005f00', '#006400', '#228B22', '#2E8B57', '#006400',
    '#01796F', '#013220', '#145A32', '#0B3D0B', '#254117'
]


for zona in zonas_poligonos:
    folium.Polygon(
        locations=zona['vertices'],
        color=colores[int(zona['Zona']) % len(colores)],
        fill=True,
        fill_opacity=0.4,
        popup=f"Zona {zona['Zona']}"
    ).add_to(mapa)

folium_static(mapa)
