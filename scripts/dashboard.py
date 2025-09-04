#!/usr/bin/env python3
"""
Dashboard Interactivo - Causas de Muerte EEUU
Dashboard web con Streamlit integrado al pipeline existente
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sqlalchemy import create_engine
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Mortalidad EEUU",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de conexión a la base de datos
@st.cache_resource
def get_db_connection():
    """Crear conexión cacheable a PostgreSQL"""
    try:
        engine = create_engine('postgresql://postgres:postgres123@localhost:5432/deaths_analysis')
        return engine
    except Exception as e:
        st.error(f"Error conectando a base de datos: {e}")
        return None

@st.cache_data
def load_main_data():
    """Cargar datos principales con cache"""
    engine = get_db_connection()
    if not engine:
        return None
    
    query = """
    SELECT 
        fd.year,
        ds.state_name,
        ds.is_national_aggregate,
        dc.cause_name,
        dc.cause_category,
        dc.is_all_causes,
        fd.deaths,
        fd.age_adjusted_death_rate
    FROM clean_data.fact_deaths fd
    JOIN clean_data.dim_states ds ON fd.state_id = ds.state_id
    JOIN clean_data.dim_causes dc ON fd.cause_id = dc.cause_id
    ORDER BY fd.year, ds.state_name, dc.cause_name
    """
    
    return pd.read_sql(query, engine)

@st.cache_data
def load_predictions():
    """Cargar predicciones si existen"""
    try:
        pred_path = 'results/predictive/mortality_predictions.csv'
        if os.path.exists(pred_path):
            return pd.read_csv(pred_path)
        return None
    except Exception:
        return None

@st.cache_data
def load_pipeline_stats():
    """Cargar estadísticas del pipeline"""
    try:
        with open('results/pipeline_stats.json', 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def create_main_header():
    """Crear header principal del dashboard"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">
            📊 Análisis de Mortalidad - Estados Unidos
        </h1>
        <p style="color: #e8f4f8; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
            Pipeline de Big Data | Período 1999-2017 | Predicciones 2018-2022
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_kpi_metrics(df, stats):
    """Crear métricas KPI principales"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(df) if df is not None else 0
        st.metric(
            label="📋 Total Registros",
            value=f"{total_records:,}",
            delta=f"{stats.get('clean_records', 0):,} procesados"
        )
    
    with col2:
        total_states = df['state_name'].nunique() if df is not None else 0
        st.metric(
            label="🗺️ Estados/Jurisdicciones",
            value=str(total_states),
            delta=f"Nacional incluido"
        )
    
    with col3:
        total_causes = df['cause_name'].nunique() if df is not None else 0
        st.metric(
            label="🏥 Causas de Muerte",
            value=str(total_causes),
            delta="Categorizadas"
        )
    
    with col4:
        years_span = f"1999-2017" if df is not None else "N/A"
        st.metric(
            label="📅 Período de Análisis",
            value=years_span,
            delta="19 años"
        )

def create_national_trends_chart(df):
    """Gráfico de tendencias nacionales"""
    if df is None:
        st.error("No se pudieron cargar los datos")
        return
    
    # Filtrar datos nacionales
    national_data = df[df['is_national_aggregate'] == True]
    
    # Separar "All causes" de las demás
    all_causes = national_data[national_data['cause_name'] == 'All causes']
    individual_causes = national_data[national_data['cause_name'] != 'All causes']
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Mortalidad Total Nacional', 'Top 5 Causas Específicas', 
                       'Tasas Ajustadas por Edad', 'Evolución por Categoría'),
        specs=[[{"secondary_y": True}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # 1. Mortalidad total
    fig.add_trace(
        go.Scatter(x=all_causes['year'], y=all_causes['deaths'],
                  mode='lines+markers', name='Muertes Totales',
                  line=dict(color='#1f77b4', width=3)),
        row=1, col=1
    )
    
    # 2. Top 5 causas específicas
    top_causes = individual_causes.groupby('cause_name')['deaths'].sum().nlargest(5)
    colors = px.colors.qualitative.Set3
    
    for i, (cause, _) in enumerate(top_causes.items()):
        cause_data = individual_causes[individual_causes['cause_name'] == cause]
        fig.add_trace(
            go.Scatter(x=cause_data['year'], y=cause_data['deaths'],
                      mode='lines+markers', name=cause,
                      line=dict(color=colors[i % len(colors)])),
            row=1, col=2
        )
    
    # 3. Tasas ajustadas
    for i, (cause, _) in enumerate(top_causes.items()):
        cause_data = individual_causes[individual_causes['cause_name'] == cause]
        fig.add_trace(
            go.Scatter(x=cause_data['year'], y=cause_data['age_adjusted_death_rate'],
                      mode='lines+markers', name=f"{cause} (Tasa)",
                      line=dict(color=colors[i % len(colors)], dash='dot'),
                      showlegend=False),
            row=2, col=1
        )
    
    # 4. Evolución por categoría
    category_evolution = individual_causes.groupby(['year', 'cause_category'])['deaths'].sum().reset_index()
    latest_year = category_evolution['year'].max()
    latest_data = category_evolution[category_evolution['year'] == latest_year]
    
    fig.add_trace(
        go.Bar(x=latest_data['cause_category'], y=latest_data['deaths'],
               name='Muertes por Categoría (2017)',
               marker_color=colors[:len(latest_data)]),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Análisis de Tendencias Nacionales (1999-2017)")
    fig.update_xaxes(title_text="Año", row=1, col=1)
    fig.update_xaxes(title_text="Año", row=1, col=2)
    fig.update_xaxes(title_text="Año", row=2, col=1)
    fig.update_xaxes(title_text="Categoría", row=2, col=2)
    
    fig.update_yaxes(title_text="Número de Muertes", row=1, col=1)
    fig.update_yaxes(title_text="Número de Muertes", row=1, col=2)
    fig.update_yaxes(title_text="Tasa Ajustada", row=2, col=1)
    fig.update_yaxes(title_text="Muertes", row=2, col=2)
    
    return fig

def create_geographic_analysis(df):
    """Análisis geográfico por estados"""
    if df is None:
        return None
    
    # Filtrar datos de estados (no nacional)
    state_data = df[df['is_national_aggregate'] == False]
    
    # Calcular promedios por estado para "All causes"
    state_mortality = state_data[state_data['cause_name'] == 'All causes'].groupby('state_name').agg({
        'deaths': 'mean',
        'age_adjusted_death_rate': 'mean'
    }).reset_index()
    
    # Crear mapa coroplético
    fig = px.choropleth(
        state_mortality,
        locations='state_name',
        color='age_adjusted_death_rate',
        locationmode='USA-states',
        color_continuous_scale='Reds',
        scope="usa",
        title="Tasa de Mortalidad Promedio por Estado (1999-2017)",
        labels={'age_adjusted_death_rate': 'Tasa Ajustada'}
    )
    
    fig.update_layout(height=500)
    return fig

def create_predictive_analysis_section(predictions_df, df):
    """Sección de análisis predictivo"""
    st.header("🔮 Análisis Predictivo")
    
    if predictions_df is None:
        st.warning("No se encontraron predicciones. Ejecuta primero: `python scripts/predictive_analysis.py`")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Predicciones de Mortalidad 2018-2022")
        
        # Seleccionar causa para visualización
        available_causes = predictions_df['cause_name'].unique()
        selected_cause = st.selectbox(
            "Seleccionar causa de muerte:",
            available_causes,
            index=0
        )
        
        # Crear gráfico de predicción
        if df is not None:
            # Datos históricos
            hist_data = df[
                (df['is_national_aggregate'] == True) & 
                (df['cause_name'] == selected_cause)
            ]
            
            # Datos predichos
            pred_data = predictions_df[predictions_df['cause_name'] == selected_cause]
            
            fig = go.Figure()
            
            # Línea histórica
            fig.add_trace(go.Scatter(
                x=hist_data['year'],
                y=hist_data['deaths'],
                mode='lines+markers',
                name='Datos Históricos',
                line=dict(color='blue', width=3),
                marker=dict(size=6)
            ))
            
            # Línea predictiva
            fig.add_trace(go.Scatter(
                x=pred_data['year'],
                y=pred_data['predicted_deaths'],
                mode='lines+markers',
                name='Predicciones',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Línea de conexión
            if not hist_data.empty and not pred_data.empty:
                last_hist = hist_data.iloc[-1]
                first_pred = pred_data.iloc[0]
                fig.add_trace(go.Scatter(
                    x=[last_hist['year'], first_pred['year']],
                    y=[last_hist['deaths'], first_pred['predicted_deaths']],
                    mode='lines',
                    line=dict(color='gray', width=2, dash='dot'),
                    showlegend=False
                ))
            
            fig.update_layout(
                title=f"Tendencia y Predicción: {selected_cause}",
                xaxis_title="Año",
                yaxis_title="Número de Muertes",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Evaluación de Riesgos")
        
        # Calcular cambios predichos
        risk_data = []
        for cause in predictions_df['cause_name'].unique():
            cause_pred = predictions_df[predictions_df['cause_name'] == cause]
            if len(cause_pred) >= 2:
                change_pct = ((cause_pred['predicted_deaths'].iloc[-1] - 
                              cause_pred['predicted_deaths'].iloc[0]) / 
                              cause_pred['predicted_deaths'].iloc[0] * 100)
                
                if change_pct > 5:
                    risk_level = "🔴 Alto"
                elif change_pct > 0:
                    risk_level = "🟡 Medio"
                else:
                    risk_level = "🟢 Bajo"
                
                risk_data.append({
                    'Causa': cause,
                    'Cambio (%)': f"{change_pct:+.1f}%",
                    'Riesgo': risk_level
                })
        
        if risk_data:
            risk_df = pd.DataFrame(risk_data).sort_values('Cambio (%)', ascending=False)
            st.dataframe(risk_df, use_container_width=True)

def create_data_quality_section(df, stats):
    """Sección de calidad de datos"""
    st.header("🔍 Calidad de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Métricas de Calidad")
        
        if df is not None:
            # Completitud
            total_cells = len(df) * len(df.columns)
            null_cells = df.isnull().sum().sum()
            completeness = (1 - null_cells / total_cells) * 100
            
            # Consistencia
            duplicates = df.duplicated().sum()
            consistency = (1 - duplicates / len(df)) * 100
            
            # Crear gráfico de métricas
            metrics_data = pd.DataFrame({
                'Métrica': ['Completitud', 'Consistencia', 'Validez'],
                'Porcentaje': [completeness, consistency, 98.5]  # Validez estimada
            })
            
            fig = px.bar(
                metrics_data, 
                x='Métrica', 
                y='Porcentaje',
                title="Métricas de Calidad de Datos",
                color='Porcentaje',
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Distribución de Datos")
        
        if df is not None:
            # Distribución por año
            year_dist = df['year'].value_counts().sort_index()
            
            fig = px.line(
                x=year_dist.index,
                y=year_dist.values,
                title="Registros por Año",
                labels={'x': 'Año', 'y': 'Número de Registros'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def create_export_section(df, predictions_df):
    """Sección de exportación de datos"""
    st.header("💾 Exportar Datos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if df is not None and st.button("Exportar Datos Históricos"):
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Descargar CSV",
                data=csv_data,
                file_name=f"mortalidad_historica_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if predictions_df is not None and st.button("Exportar Predicciones"):
            csv_pred = predictions_df.to_csv(index=False)
            st.download_button(
                label="Descargar Predicciones CSV",
                data=csv_pred,
                file_name=f"predicciones_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col3:
        # Crear reporte resumen
        if st.button("Generar Reporte Ejecutivo"):
            summary = {
                'fecha_reporte': datetime.now().isoformat(),
                'total_registros': len(df) if df is not None else 0,
                'periodo_analisis': '1999-2017',
                'predicciones_disponibles': predictions_df is not None,
                'causas_principales': df[df['is_national_aggregate'] == True]['cause_name'].unique().tolist() if df is not None else []
            }
            
            json_data = json.dumps(summary, indent=2, default=str)
            st.download_button(
                label="Descargar Reporte JSON",
                data=json_data,
                file_name=f"reporte_ejecutivo_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

def main():
    """Función principal del dashboard"""
    # Header principal
    create_main_header()
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        df = load_main_data()
        predictions_df = load_predictions()
        stats = load_pipeline_stats()
    
    # Sidebar con controles
    st.sidebar.header("🔧 Controles del Dashboard")
    
    # Selector de vista
    view_options = [
        "📊 Resumen Ejecutivo",
        "📈 Análisis de Tendencias",
        "🗺️ Análisis Geográfico", 
        "🔮 Análisis Predictivo",
        "🔍 Calidad de Datos",
        "💾 Exportar Datos"
    ]
    
    selected_view = st.sidebar.selectbox("Seleccionar Vista:", view_options)
    
    # Filtros adicionales
    if df is not None:
        st.sidebar.subheader("Filtros")
        
        # Filtro de años
        year_range = st.sidebar.slider(
            "Rango de Años:",
            min_value=int(df['year'].min()),
            max_value=int(df['year'].max()),
            value=(int(df['year'].min()), int(df['year'].max()))
        )
        
        # Filtro de causas
        available_causes = df['cause_name'].unique().tolist()
        selected_causes = st.sidebar.multiselect(
            "Causas de Muerte:",
            available_causes,
            default=available_causes[:5]
        )
        
        # Aplicar filtros
        df_filtered = df[
            (df['year'] >= year_range[0]) & 
            (df['year'] <= year_range[1]) &
            (df['cause_name'].isin(selected_causes) if selected_causes else True)
        ]
    else:
        df_filtered = df
    
    # Mostrar vista seleccionada
    if selected_view == "📊 Resumen Ejecutivo":
        create_kpi_metrics(df_filtered, stats)
        
        if df_filtered is not None:
            st.header("📈 Tendencias Principales")
            fig = create_national_trends_chart(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif selected_view == "📈 Análisis de Tendencias":
        st.header("📈 Análisis Detallado de Tendencias")
        if df_filtered is not None:
            fig = create_national_trends_chart(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif selected_view == "🗺️ Análisis Geográfico":
        st.header("🗺️ Análisis Geográfico")
        fig_map = create_geographic_analysis(df_filtered)
        if fig_map:
            st.plotly_chart(fig_map, use_container_width=True)
    
    elif selected_view == "🔮 Análisis Predictivo":
        create_predictive_analysis_section(predictions_df, df)
    
    elif selected_view == "🔍 Calidad de Datos":
        create_data_quality_section(df_filtered, stats)
    
    elif selected_view == "💾 Exportar Datos":
        create_export_section(df_filtered, predictions_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Dashboard de Análisis de Mortalidad EEUU | Pipeline de Big Data | Universidad de la Sabana</p>
        <p>Datos: NCHS Leading Causes of Death (1999-2017) | Actualizado: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()