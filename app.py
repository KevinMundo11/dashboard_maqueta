import streamlit as st
import plotly.express as px
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import plotly.express as px






#######################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

@st.cache_data
def cargar_datos_parquet(path):
    return pd.read_parquet(path)

@st.cache_data
def cargar_shapefile(path):
    return gpd.read_file(path)

@st.cache_data
def procesar_df(df, columnas_categoria, fecha_col, drop_col=None):
    for col in columnas_categoria:
        df[col] = df[col].astype('category')
    df[fecha_col] = pd.to_datetime(df[fecha_col], errors='coerce')
    df['Año'] = df[fecha_col].dt.year
    if drop_col:
        df = df.dropna(subset=[drop_col]).reset_index(drop=True)
    return df


def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5


def generar_mapa(df, col_valores, mexico_map, titulo):
    mapa = folium.Map(location=[23.6345, -102.5528], zoom_start=5)
    max_val = df[col_valores].max()
    min_val = df[col_valores].min()

    for _, row in df.iterrows():
        if not pd.isnull(row[col_valores]):
            porcentaje_normalizado = normalize(row[col_valores], min_val, max_val)
            folium.GeoJson(
                data=row["geometry"].__geo_interface__,
                style_function=lambda feature, porcentaje=porcentaje_normalizado: {
                    "fillColor": mcolors.to_hex(plt.cm.viridis(porcentaje)),
                    "color": "black",
                    "weight": 0.5,
                    "fillOpacity": 0.7,
                },
                tooltip=folium.Tooltip(f"{row['Entidad']}: {row[col_valores]:.2f}%"),
            ).add_to(mapa)
    return mapa

# Función para calcular porcentajes
@st.cache_data
def calcular_porcentajes(df_1, df_2):
    resultados = []
    for año in df_1["Año"].unique():
        datos_filtrados = df_1[df_1["Año"] == año]
        total = len(datos_filtrados)
        if total == 0:
            continue
        promedio_depresion = datos_filtrados["Depresion"].isin(['3', '4']).sum() * 100 / total
        promedio_tristeza = datos_filtrados["Tristeza"].isin(['3', '4']).sum() * 100 / total
        promedio_atentar = datos_filtrados["Atentar_contras_si"].eq('1').sum() * 100 / total
        promeido_emborrachar = datos_filtrados['Frecuencia emborrachar'].isin(['1', '2']).sum() * 100 / total
        resultados += [
            {"Año": año, "Categoría": "Depresión", "Porcentaje": promedio_depresion},
            {"Año": año, "Categoría": "Tristeza", "Porcentaje": promedio_tristeza},
            {"Año": año, "Categoría": "Atentar contra sí", "Porcentaje": promedio_atentar},
            {"Año": año, "Categoría": 'Frecuencia emborrachar', "Porcentaje": promeido_emborrachar}
        ]
    for año in df_2["Año"].unique():
        datos_filtrados = df_2[df_2["Año"] == año]
        total = len(datos_filtrados)
        if total == 0:
            continue
        promedio_no_alfabetismo = datos_filtrados["Alfabetismo"].eq(2).sum() * 100 / total
        resultados.append({"Año": año, "Categoría": "No alfabetismo", "Porcentaje": promedio_no_alfabetismo})
    return pd.DataFrame(resultados)

#######################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
#CARGAR DATOS

path_1 = os.path.join('data', 'ensanut_processed.parquet')
path_2 = os.path.join('data', 'enigh_processed.parquet')
df_1 = cargar_datos_parquet(path_1)
df_2 = cargar_datos_parquet(path_2)

df_1 = procesar_df(
    df_1,
    ['Sexo', 'C_Entidad', 'Generacion', 'Atentar_contras_si'],
    'Fecha',
    drop_col='Edad'
)
df_1['Edad'] = df_1['Edad'].astype('int')

df_2 = procesar_df(
    df_2,
    ['Sexo', 'Alfabetismo', 'C_Entidad', 'Generacion'],
    'Fecha'
)

df_2 = df_2[df_2['Edad'] > 9].reset_index(drop=True)
df_1['Atentar_contras_si'] = df_1['Atentar_contras_si'].cat.remove_unused_categories()


# Cargar el shapefile de México
shapefile_path = os.path.join('data', '2023_1_00_ENT.shp')
mexico_map = cargar_shapefile(shapefile_path)

# Cambiar el nombre de la columna NOMGEO a Entidad en el shapefile para hacer coincidir con el DataFrame
mexico_map = mexico_map.rename(columns={'NOMGEO': 'Entidad'})

# Diccionario de correspondencias entre los nombres de entidades en 'Entidad' del DataFrame y 'Entidad' del shapefile
correspondencias = {
    'AGUASCALIENTES': 'Aguascalientes',
    'BAJA CALIFORNIA': 'Baja California',
    'BAJA CALIFORNIA SUR': 'Baja California Sur',
    'CAMPECHE': 'Campeche',
    'COAHUILA DE ZARAGOZA': 'Coahuila de Zaragoza',
    'COLIMA': 'Colima',
    'CHIAPAS': 'Chiapas',
    'CHIHUAHUA': 'Chihuahua',
    'CIUDAD DE MÉXICO': 'Ciudad de México',
    'DURANGO': 'Durango',
    'GUANAJUATO': 'Guanajuato',
    'GUERRERO': 'Guerrero',
    'HIDALGO': 'Hidalgo',
    'JALISCO': 'Jalisco',
    'MÉXICO': 'México',
    'MICHOACÁN DE OCAMPO': 'Michoacán de Ocampo',
    'MORELOS': 'Morelos',
    'NAYARIT': 'Nayarit',
    'NUEVO LEÓN': 'Nuevo León',
    'OAXACA': 'Oaxaca',
    'PUEBLA': 'Puebla',
    'QUERÉTARO': 'Querétaro',
    'QUINTANA ROO': 'Quintana Roo',
    'SAN LUIS POTOSÍ': 'San Luis Potosí',
    'SINALOA': 'Sinaloa',
    'SONORA': 'Sonora',
    'TABASCO': 'Tabasco',
    'TAMAULIPAS': 'Tamaulipas',
    'TLAXCALA': 'Tlaxcala',
    'VERACRUZ DE IGNACIO DE LA LLAVE': 'Veracruz de Ignacio de la Llave',
    'YUCATÁN': 'Yucatán',
    'ZACATECAS': 'Zacatecas'
}

# Reemplazar los valores en la columna 'Entidad' en el DataFrame con los valores correspondientes del diccionario
df_1['Entidad'] = df_1['Entidad'].replace(correspondencias)
df_2['Entidad'] = df_2['Entidad'].replace(correspondencias)

#######################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
#VARIABLES GLOBALES

category_descriptions = {
    '1': "Alfabeta",
    '2': "No alfabeta",
}
    

##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

















# Título principal
st.title("Dashboard de Análisis")


# Crear un menú en el sidebar
menu = st.sidebar.radio(
    "Navegación",
    ["Alfabetización", "Alfabetización / Hábitos de Alcoholismo", "Salud Mental", "Disclaimer"]
)



#####################################################################################################################################################################################################################################################################################################################################################################################################################################################

# Mostrar contenido basado en la selección
if menu == "Alfabetización":
    st.markdown("""
    Este dashboard presenta un análisis interactivo de tristeza, depresión, alfabetismo y otros factores relacionados.
    """)
    st.title("Distribución Porcentual de Alfabetización por Generación")
    st.markdown("""
    Este análisis muestra cómo se distribuye el nivel de alfabetización entre diferentes generaciones. 
    Se destacan las categorías de personas alfabetas y no alfabetas.
    """)

    # Crear la gráfica de alfabetización
    cross_tab = pd.crosstab(df_2['Generacion'], df_2['Alfabetismo'])
    cross_tab_percentage = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
    viridis_colors = [mcolors.rgb2hex(cm.viridis(i / len(cross_tab_percentage.columns))) for i in range(len(cross_tab_percentage.columns))]

    fig = go.Figure()
    for i, col in enumerate(cross_tab_percentage.columns):
        hover_text = [
            f"Generación: {gen}<br>"
            f"Categoría: {col} ({category_descriptions.get(str(col), 'Descripción no disponible')})<br>"
            f"Porcentaje: {val:.2f}%"
            for gen, val in zip(cross_tab_percentage.index, cross_tab_percentage[col])
        ]
        fig.add_trace(
            go.Bar(
                x=cross_tab_percentage.index,
                y=cross_tab_percentage[col],
                name=f"Categoría {col}",
                marker_color=viridis_colors[i],
                hovertext=hover_text,
                hoverinfo="text"
            )
        )
    
    fig.update_layout(
        title={
            "text": "Distribución Porcentual de Alfabetización por Generación<br><sup>Análisis de Respuestas</sup>",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top"
        },
        xaxis_title="Generación",
        yaxis_title="Porcentaje (%)",
        barmode="group",
        legend_title="Categorías de Alfabetización",
        template="plotly_white",
        xaxis=dict(tickangle=45, showgrid=False),
        yaxis=dict(showgrid=False),
    )
    
    # Mostrar la gráfica
    st.plotly_chart(fig)


#####################################################################################################################################################################################################################################################################################################################################################################################################################################################



elif menu == "Alfabetización / Hábitos de Alcoholismo":
    st.title("Relación entre Alfabetización y Hábitos de Alcoholismo")
    st.markdown("""
    Este análisis compara la distribución de alfabetización con los hábitos de consumo de alcohol en las diferentes entidades de México.
    """)

    # Seleccionar el año desde el usuario
    anio_seleccionado = st.selectbox("Seleccione el año:", sorted(list(set(df_2['Año'].unique()).union(set(df_1['Año'].unique())))))

    # Calcular y mostrar los mapas
    st.subheader(f"Análisis para el Año {anio_seleccionado}")

    # Mapa 1: Analfabetismo
    st.markdown("#### Porcentaje de Analfabetismo por Entidad")
    data_anio = df_2[(df_2['Año'] == anio_seleccionado) & (df_2['Alfabetismo'].notna())]
    totals = data_anio.groupby('Entidad').size()
    percentages_1 = (data_anio[data_anio['Alfabetismo'] == 2].groupby('Entidad').size() / totals) * 100
    percentages_1 = percentages_1.fillna(0).reset_index()
    percentages_1.columns = ['Entidad', 'Porcentaje_1']
    mexico_data_sample = mexico_map.merge(percentages_1, on='Entidad', how='left')
    
    # Crear mapa con folium
    mapa_alfabetismo = generar_mapa(mexico_data_sample, "Porcentaje_1", mexico_map, "Alfabetismo")
    st_folium(mapa_alfabetismo, width=700, height=500)

    # Mapa 2: Porcentaje de Tomadores Frecuentes
    st.markdown("#### Porcentaje de Tomadores al menos Semanales por Entidad")
    data_anio = df_1[(df_1['Año'] == anio_seleccionado) & (df_1['Frecuencia emborrachar'].notna())]
    totals = data_anio.groupby('Entidad').size()
    percentages_2 = (data_anio[(data_anio['Frecuencia emborrachar'] == '1') |
                               (data_anio['Frecuencia emborrachar'] == '2')].groupby('Entidad').size() / totals) * 100
    percentages_2 = percentages_2.fillna(0).reset_index()
    percentages_2.columns = ['Entidad', 'Porcentaje_2']
    mexico_data_sample = mexico_map.merge(percentages_2, on='Entidad', how='left')

    # Crear mapa con folium
    mapa_emborrachar = generar_mapa(mexico_data_sample, "Porcentaje_2", mexico_map, "Habitos de alcoholismo")
    st_folium(mapa_emborrachar, width=700, height=500)



#####################################################################################################################################################################################################################################################################################################################################################################################################################################################



elif menu == "Salud Mental":
    st.title("Análisis de Salud Mental")
    st.markdown("""
    En esta sección, se analizan las respuestas relacionadas con tristeza, depresión y autolesión. Los resultados están organizados por generación para identificar tendencias importantes.
    """)
    variables = ['Tristeza', 'Depresion', 'Atentar_contras_si']
    titles = ['Tristeza', 'Depresión', 'Autolesión']
    category_descriptions_tristeza = {
            '1': "Tristeza un día o menos por semana",
            '2': "Tristeza dos o tres días por semana",
            '3': "Tristeza cuatro o cinco días por semana",
            '4': "Tristeza seis o siete días por semana"
        }
    category_descriptions_depresion = {
            '1': "Depresión un día o menos por semana",
            '2': "Depresión dos o tres días por semana",
            '3': "Depresión cuatro o cinco días por semana",
            '4': "Depresión seis o siete días por semana"
        }
    category_descriptions_autolesion = {
            '1': "Alguna vez intentó acabar con su vida",
            '2': "Nunca ha intentado acabar con su vida",
        }


    category_descriptions = [category_descriptions_tristeza, category_descriptions_depresion, category_descriptions_autolesion]

    # Generar colores bien diferenciados de la paleta Viridis
    viridis_colors = [mcolors.rgb2hex(cm.viridis(i / 4)) for i in range(4)]

    # Crear subplots con tres columnas
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles, shared_yaxes=True)

    # Generar cada gráfica
    for i, var in enumerate(variables):
        # Crear tabla cruzada
        cross_tab = pd.crosstab(df_1['Generacion'], df_1[var])
        cross_tab_percentage = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

        # Agregar barras para cada categoría de la variable
        for j, col in enumerate(cross_tab_percentage.columns):
            hover_text = [
                f"Generación: {gen}<br>"
                f"Categoría: {col} ({category_descriptions[i].get(str(col), 'Descripción no disponible')})<br>"
                f"Porcentaje: {val:.2f}%"
                for gen, val in zip(cross_tab_percentage.index, cross_tab_percentage[col])
            ]
            fig.add_trace(
                go.Bar(
                    x=cross_tab_percentage.index,
                    y=cross_tab_percentage[col],
                    name=f"{var} - {col}",  # Especificar variable y categoría
                    marker_color=viridis_colors[j % len(viridis_colors)],
                    hovertext=hover_text,  # Texto al pasar el cursor
                    hoverinfo="text",  # Mostrar solo el texto proporcionado
                    showlegend=True  # Mostrar leyenda para todas las categorías
                ),
                row=1,
                col=i + 1
            )

        # Configurar el diseño del subplot
        fig.update_xaxes(tickangle=45, row=1, col=i + 1)
        if i == 0:
            fig.update_yaxes(title_text="Porcentaje (%)", row=1, col=i + 1)

    # Configurar diseño general
    fig.update_layout(
        title={
            "text": "Distribución Porcentual por Generación<br><sup>Análisis de Respuestas</sup>",
            "x": 0.5,  # Centrar título
            "xanchor": "center",
            "yanchor": "top"
        },
        yaxis_title="Porcentaje (%)",
        barmode="stack",  # Barras apiladas
        legend_title="Categorías",
        template="plotly_white",  # Tema blanco
        xaxis=dict(tickangle=45, showgrid=False),  # Rotar etiquetas en el eje X
        yaxis=dict(showgrid=False),
    )

    # Mostrar la gráfica en Streamlit
    st.plotly_chart(fig)




#####################################################################################################################################################################################################################################################################################################################################################################################################################################################


elif menu == "Disclaimer":
    st.title("Disclaimer")
    st.markdown("""
    ### Limitaciones de los datos y las visualizaciones:
    - Existen **datos perdidos** en las encuestas originales, lo que puede afectar la representatividad de los resultados.
    - Los valores porcentuales bajos en algunas categorías reflejan **tamaños de muestra pequeños**, lo que limita la generalización.
    - Los resultados presentados son indicativos, pero deben interpretarse con precaución y en el contexto del diseño de la encuesta.
    """)

    # Calcular porcentajes
    df_resultados = calcular_porcentajes(df_1, df_2)

    # Verificar categorías únicas
    categorias_unicas = df_resultados["Categoría"].unique()

    if not df_resultados.empty:
        # Generar colores diferenciados con la paleta Viridis
        viridis_colors = [mcolors.rgb2hex(cm.viridis(i / (len(categorias_unicas) - 1))) for i in range(len(categorias_unicas))]

        # Crear la gráfica de líneas
        fig = px.line(
            df_resultados,
            x="Año",
            y="Porcentaje",
            color="Categoría",
            markers=True,
            color_discrete_sequence=viridis_colors  # Asignar colores generados
        )

        # Personalizar el diseño de la gráfica
        fig.update_traces(line=dict(width=4))  # Ajustar el grosor de las líneas
        fig.update_layout(
            title={
                "text": "Porcentaje por año<br><sup>Análisis de las Respuestas de Interés</sup>",
                "x": 0.5,  # Centrar el título
                "xanchor": "center",
                "yanchor": "top"
            },
            yaxis_title="Porcentaje (%)",
            xaxis_title="Año",
            xaxis=dict(
                tickmode="array",
                tickvals=df_resultados["Año"].unique(),  # Asegurar que se muestren los años únicos
                showgrid=False  # Quitar el grid del eje X
            ),
            yaxis=dict(
                showgrid=False  # Quitar el grid del eje Y
            ),
            template="plotly_white"  # Tema limpio
        )

        # Mostrar la gráfica en Streamlit
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No se encontraron datos para generar la gráfica. Verifica los datos de entrada.")


#####################################################################################################################################################################################################################################################################################################################################################################################################################################################



# Notas generales
st.sidebar.markdown("""
### Notas:
- Este dashboard es interactivo y permite explorar diferentes aspectos como alfabetización, hábitos de alcoholismo y salud mental.
- Navega por las secciones para ver los análisis específicos.
""")















