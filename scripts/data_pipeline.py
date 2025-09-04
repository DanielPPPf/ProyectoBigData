#!/usr/bin/env python3
"""
Pipeline de Datos Final - Causas de Muerte EEUU
Version corregida con IDs simples
"""

import sys
import os
import logging
from datetime import datetime
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import json

# Crear directorio results
os.makedirs('results', exist_ok=True)

# Configurar logging simple
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'results/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Crear conexión a PostgreSQL"""
    try:
        engine = create_engine('postgresql://postgres:postgres123@localhost:5432/deaths_analysis')
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Conexión a base de datos establecida")
        return engine
    except Exception as e:
        logger.error(f"Error conectando a base de datos: {e}")
        raise

def load_csv_data():
    """Cargar datos del CSV"""
    try:
        csv_path = 'data/NCHS__Leading_Causes_of_Death__United_States.csv'
        logger.info(f"Cargando datos de: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Datos cargados: {len(df)} registros")
        return df
    except Exception as e:
        logger.error(f"Error cargando CSV: {e}")
        raise

def clean_data(df):
    """Limpiar y transformar datos"""
    logger.info("Limpiando datos...")
    
    # Renombrar columnas
    df_clean = df.rename(columns={
        '113 Cause Name': 'cause_name_113',
        'Cause Name': 'cause_name',
        'State': 'state',
        'Year': 'year',
        'Deaths': 'deaths',
        'Age-adjusted Death Rate': 'age_adjusted_death_rate'
    })
    
    # Limpiar tipos de datos
    df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce')
    df_clean['deaths'] = pd.to_numeric(df_clean['deaths'], errors='coerce')
    df_clean['age_adjusted_death_rate'] = pd.to_numeric(df_clean['age_adjusted_death_rate'], errors='coerce')
    
    # Limpiar strings
    for col in ['state', 'cause_name', 'cause_name_113']:
        df_clean[col] = df_clean[col].str.strip()
    
    # Remover registros problemáticos
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['year', 'state', 'cause_name', 'deaths'])
    df_clean = df_clean[df_clean['deaths'] >= 0]
    df_clean = df_clean.drop_duplicates()
    
    final_count = len(df_clean)
    logger.info(f"Datos limpios: {final_count} registros ({initial_count - final_count} removidos)")
    
    return df_clean

def create_tables_and_load_data():
    """Crear tablas y cargar datos - VERSION FINAL CORREGIDA"""
    logger.info("Iniciando pipeline...")
    
    # 1. Conectar a DB
    engine = get_db_connection()
    
    # 2. Cargar CSV
    df = load_csv_data()
    
    # 3. Limpiar datos
    df_clean = clean_data(df)
    
    # 4. Cargar datos raw
    logger.info("Cargando datos en raw_data.deaths_raw...")
    df_clean.to_sql('deaths_raw', engine, schema='raw_data', 
                    if_exists='replace', index=False, method='multi')
    
    # 5. Crear dimensión de estados CON IDs
    logger.info("Creando dimensión de estados...")
    states_dim = pd.DataFrame({
        'state_name': df_clean['state'].unique()
    }).sort_values('state_name').reset_index(drop=True)
    states_dim['is_national_aggregate'] = states_dim['state_name'] == 'United States'
    
    # Agregar IDs simples
    states_dim['state_id'] = range(1, len(states_dim) + 1)
    
    # 6. Crear dimensión de causas CON IDs
    logger.info("Creando dimensión de causas...")
    causes_grouped = df_clean.groupby('cause_name')['cause_name_113'].first().reset_index()
    causes_dim = pd.DataFrame({
        'cause_name': causes_grouped['cause_name'],
        'cause_name_113': causes_grouped['cause_name_113']
    })
    causes_dim['is_all_causes'] = causes_dim['cause_name'] == 'All causes'
    
    # Categorizar causas
    cause_categories = {
        'All causes': 'Total',
        'Heart disease': 'Cardiovascular',
        'Cancer': 'Cancer',
        'Stroke': 'Cardiovascular',
        'CLRD': 'Respiratory',
        'Unintentional injuries': 'External',
        'Diabetes': 'Endocrine',
        'Influenza and pneumonia': 'Infectious',
        "Alzheimer's disease": 'Neurological',
        'Kidney disease': 'Renal',
        'Suicide': 'External'
    }
    causes_dim['cause_category'] = causes_dim['cause_name'].map(cause_categories)
    
    # Agregar IDs simples
    causes_dim['cause_id'] = range(1, len(causes_dim) + 1)
    
    # 7. Cargar dimensiones
    logger.info("Cargando dimensiones...")
    states_dim.to_sql('dim_states', engine, schema='clean_data', 
                     if_exists='replace', index=False, method='multi')
    causes_dim.to_sql('dim_causes', engine, schema='clean_data', 
                     if_exists='replace', index=False, method='multi')
    
    # 8. Crear tabla de hechos USANDO LOS IDs
    logger.info("Creando tabla de hechos...")
    
    # Usar los DataFrames con IDs directamente
    fact_table = df_clean.merge(
        states_dim[['state_id', 'state_name']], 
        left_on='state', 
        right_on='state_name', 
        how='left'
    ).merge(
        causes_dim[['cause_id', 'cause_name']], 
        left_on='cause_name', 
        right_on='cause_name', 
        how='left'
    )
    
    fact_deaths = fact_table[[
        'year', 'state_id', 'cause_id', 'deaths', 'age_adjusted_death_rate'
    ]].dropna()
    
    fact_deaths.to_sql('fact_deaths', engine, schema='clean_data', 
                      if_exists='replace', index=False, method='multi')
    
    # 9. Generar reportes
    logger.info("Generando reportes...")
    
    # Reporte nacional
    with engine.connect() as conn:
        national_query = """
        SELECT 
            fd.year,
            dc.cause_name,
            fd.deaths,
            fd.age_adjusted_death_rate
        FROM clean_data.fact_deaths fd
        JOIN clean_data.dim_states ds ON fd.state_id = ds.state_id
        JOIN clean_data.dim_causes dc ON fd.cause_id = dc.cause_id
        WHERE ds.is_national_aggregate = true
        ORDER BY fd.year, fd.deaths DESC
        """
        national_df = pd.read_sql(national_query, conn)
        national_df.to_csv('results/national_trends.csv', index=False)
    
    # Estadísticas finales
    stats = {
        'timestamp': datetime.now().isoformat(),
        'raw_records': len(df),
        'clean_records': len(df_clean),
        'states': len(states_dim),
        'causes': len(causes_dim),
        'fact_records': len(fact_deaths)
    }
    
    with open('results/pipeline_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("=== PIPELINE COMPLETADO ===")
    logger.info(f"Registros procesados: {stats['clean_records']}")
    logger.info(f"Estados: {stats['states']}, Causas: {stats['causes']}")
    logger.info("Reportes generados en results/")

if __name__ == '__main__':
    try:
        create_tables_and_load_data()
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        sys.exit(1)