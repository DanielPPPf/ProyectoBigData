# Pipeline de Big Data: Análisis Predictivo de Mortalidad EEUU

Pipeline integral de Big Data para el análisis de las principales causas de muerte en Estados Unidos (1999-2017), incluyendo capacidades predictivas, visualización interactiva y evaluación de calidad de datos.

## Índice

- [Descripción General](#descripción-general)
- [Características Principales](#características-principales)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Uso del Sistema](#uso-del-sistema)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Componentes del Pipeline](#componentes-del-pipeline)
- [Visualización de Resultados](#visualización-de-resultados)
- [Solución de Problemas](#solución-de-problemas)
- [Extensiones y Mejoras](#extensiones-y-mejoras)

## Descripción General

Este proyecto implementa un pipeline completo de Big Data que procesa datos históricos de mortalidad del NCHS (National Center for Health Statistics), transformándolos en insights accionables para la toma de decisiones en salud pública.

**Dataset procesado:** NCHS Leading Causes of Death: United States  
**Período analizado:** 1999-2017 (19 años)  
**Registros procesados:** 10,868  
**Cobertura geográfica:** 50 estados + DC + agregado nacional  
**Causas de muerte:** 11 principales categorías

## Características Principales

### Pipeline ETL Robusto
- Extracción automática desde CSV
- Limpieza y validación de datos con logging detallado
- Modelo dimensional (esquema estrella) en PostgreSQL
- Garantías de calidad de datos (completitud 100%)

### Análisis Predictivo
- Modelos de machine learning (regresión lineal + Random Forest)
- Predicciones para período 2018-2022
- Métricas de rendimiento (R² >0.85 para principales causas)
- Evaluación de riesgos por causa de muerte

### Dashboard Interactivo
- Interface web con Streamlit
- 6 vistas especializadas
- Filtros dinámicos y visualizaciones interactivas
- Mapas coropléticos de Estados Unidos
- Capacidades de exportación

### Evaluación de Calidad Big Data
- Análisis completo de las 5 V's (Volume, Velocity, Variety, Veracity, Value)
- Métricas automatizadas de calidad de datos
- Benchmarks de rendimiento

## Arquitectura del Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   CSV Data      │───▶│  ETL Pipeline    │───▶│   PostgreSQL DB     │
│ (NCHS Dataset)  │    │  (data_pipeline) │    │  (Raw + Clean Data) │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                           │
┌─────────────────┐    ┌──────────────────┐               │
│ Interactive     │◀───│  Predictive      │◀──────────────┘
│ Dashboard       │    │  Analysis        │
│ (Streamlit)     │    │  (ML Models)     │
└─────────────────┘    └──────────────────┘
```

## Requisitos del Sistema

### Software Base
- **Docker** >= 20.10.0 y Docker Compose >= 1.29.0
- **Python** >= 3.8 (recomendado: Conda)
- **Navegador web** moderno (Chrome, Firefox, Safari)

### Puertos Requeridos
- **5432:** PostgreSQL
- **8501:** Dashboard Streamlit

## Instalación

### 1. Preparación del Entorno

```bash
# Clonar repositorio (o descargar archivos)
mkdir mortality-analysis-pipeline
cd mortality-analysis-pipeline

# Crear estructura de directorios
make create-dirs
# O manualmente:
mkdir -p data scripts init-db results config
```

### 2. Instalar Dependencias Python

```bash
# Con conda (recomendado)
conda create -n mortality-analysis python=3.11 
conda activate mortality-analysis

# Instalar paquetes principales
conda install pandas sqlalchemy psycopg2 scikit-learn
conda install matplotlib seaborn plotly streamlit

# Con pip (alternativo)
pip install pandas sqlalchemy psycopg2-binary scikit-learn
pip install matplotlib seaborn plotly streamlit
```

### 3. Obtener Dataset

Descargar el archivo CSV o descargarlo desde el repositorio:
- **URL:** https://data.cdc.gov/NCHS/NCHS-Leading-Causes-of-Death-United-States/bi63-dtpu
- **Archivo:** `NCHS__Leading_Causes_of_Death__United_States.csv`
- **Ubicación:** Colocar en carpeta `data/`

### 4. Configurar Docker

```yaml
# docker-compose.yml (crear si no existe)
version: '3.8'
services:
  postgres:
    image: postgres:13
    container_name: deaths_postgres
    environment:
      POSTGRES_DB: deaths_analysis
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## Uso del Sistema

### Flujo de Trabajo Completo

```bash
# 1. Verificar instalación
make help

# 2. Iniciar base de datos
make up

# 3. Esperar a que PostgreSQL esté listo
make wait-for-db

# 4. Verificar conexión y dependencias
make test-connection
make check-python
make check-data

# 5. Ejecutar pipeline ETL
python scripts/data_pipeline.py

# 6. Ejecutar análisis predictivo
python scripts/predictive_analysis.py

# 7. Lanzar dashboard interactivo
streamlit run scripts/dashboard.py
# Acceder via: http://localhost:8501

# 8. Análisis de calidad Big Data (opcional)
python scripts/bigdata_5vs_analysis.py
```

### Comandos Makefile Disponibles

```bash
make help              # Mostrar ayuda completa
make up                # Iniciar PostgreSQL
make down              # Detener servicios
make wait-for-db       # Esperar conexión BD
make test-connection   # Probar conectividad
make check-data        # Verificar archivo CSV
make check-python      # Verificar dependencias
make db-connect        # Conectar a PostgreSQL CLI
make logs              # Ver logs de PostgreSQL
make clean             # Limpiar contenedores
make status            # Estado de servicios
```

### Scripts Individuales

#### Pipeline ETL Principal
```bash
python scripts/data_pipeline.py
```
**Salidas generadas:**
- Base de datos poblada (raw_data + clean_data schemas)
- `results/national_trends.csv`
- `results/pipeline_stats.json`
- Logs detallados

#### Análisis Predictivo
```bash
python scripts/predictive_analysis.py
```
**Salidas generadas:**
- `results/predictive/mortality_predictions.csv`
- `results/predictive/predictive_analysis_report.json`
- `results/plots/predictive_analysis.png`

#### Análisis 5 V's Big Data
```bash
python scripts/bigdata_5vs_analysis.py
```
**Salidas generadas:**
- `results/bigdata_5vs_report.json`
- Métricas detalladas de Volume, Velocity, Variety, Veracity, Value

#### Dashboard Interactivo
```bash
streamlit run scripts/dashboard.py
```
**Funcionalidades:**
- 6 vistas especializadas (Resumen, Tendencias, Geografía, etc.)
- Filtros interactivos por año y causa
- Exportación de datos y reportes
- Visualizaciones con Plotly

## Estructura del Proyecto

```
mortality-analysis-pipeline/
├── README.md
├── Makefile
├── docker-compose.yml
├── data/
│   └── NCHS__Leading_Causes_of_Death__United_States.csv
├── scripts/
│   ├── data_pipeline.py           # Pipeline ETL principal
│   ├── predictive_analysis.py     # Análisis predictivo ML
│   ├── bigdata_5vs_analysis.py    # Evaluación 5 V's
│   └── dashboard.py               # Dashboard Streamlit
├── results/
│   ├── predictive/                # Resultados de predicciones
│   ├── plots/                     # Gráficos generados
│   ├── national_trends.csv        # Tendencias nacionales
│   ├── pipeline_stats.json        # Estadísticas del pipeline
│   └── bigdata_5vs_report.json    # Reporte 5 V's
└── config/
    └── [archivos de configuración]
```

## Componentes del Pipeline

### 1. Extracción de Datos (Extract)
- Lectura automática del CSV con validaciones
- Verificación de estructura y tipos esperados
- Logging de métricas de carga

### 2. Transformación (Transform)
- Normalización de nombres de columnas
- Limpieza de datos (nulos, duplicados, tipos)
- Creación de dimensiones con IDs únicos
- Categorización médica de causas de muerte

### 3. Carga (Load)
- Schema `raw_data`: Datos originales sin procesar
- Schema `clean_data`: Modelo dimensional optimizado
  - `dim_states`: Dimensión de estados/jurisdicciones
  - `dim_causes`: Dimensión de causas categorizadas
  - `fact_deaths`: Tabla de hechos con métricas

### 4. Análisis Predictivo
- **Preparación de features:** Lags, tendencias, promedios móviles
- **Modelos implementados:**
  - Regresión lineal para series temporales simples
  - Random Forest para análisis multivariado
- **Validación:** División temporal, métricas R², MAE, RMSE
- **Predicciones:** Horizonte 2018-2022 para 10 causas principales

### 5. Evaluación de Calidad
- **Volume:** 10,868 registros, 4.476 MB procesados
- **Velocity:** >5,000 registros/segundo throughput
- **Variety:** Multi-dimensional (temporal, geográfico, categórico)
- **Veracity:** 100% completitud, consistencia verificada
- **Value:** 15+ insights accionables identificados

## Visualización de Resultados

### Dashboard Web (Puerto 8501)

#### Vista Resumen Ejecutivo
- KPIs principales del dataset
- Métricas de procesamiento
- Gráficos de tendencias principales

#### Vista Análisis de Tendencias
- Evolución temporal por causa de muerte
- Comparativas entre principales causas
- Análisis de tasas ajustadas por edad

#### Vista Análisis Geográfico
- Mapa coroplético de Estados Unidos
- Disparidades por estado
- Rankings de mortalidad

#### Vista Análisis Predictivo
- Predicciones futuras interactivas
- Evaluación de riesgos por causa
- Métricas de confianza del modelo

#### Vista Calidad de Datos
- Métricas de completitud y consistencia
- Distribuciones de datos
- Validaciones aplicadas

#### Vista Exportar Datos
- Descarga de datasets históricos
- Reportes de predicciones
- Summaries ejecutivos en JSON

### Archivos de Salida

**CSV Reports:**
- `national_trends.csv`: Tendencias históricas nacionales
- `mortality_predictions.csv`: Predicciones 2018-2022
- Exports personalizados desde dashboard

**JSON Reports:**
- `pipeline_stats.json`: Estadísticas de procesamiento
- `predictive_analysis_report.json`: Métricas de modelos ML
- `bigdata_5vs_report.json`: Evaluación completa de calidad

**Visualizaciones PNG:**
- `predictive_analysis.png`: Gráficos de predicciones
- Exports automáticos desde dashboard

## Solución de Problemas

### Error: "Connection refused" PostgreSQL

```bash
# Verificar estado de Docker
docker ps

# Reiniciar PostgreSQL
make down
make up
make wait-for-db

# Verificar puertos
netstat -an | grep 5432
```

### Error: "Module not found"

```bash
# Verificar entorno conda activo
conda env list

# Reinstalar dependencias
make check-python
conda install [paquete-faltante]
```

### Error: "File not found" CSV

```bash
# Verificar archivo de datos
make check-data

# Descargar dataset si falta
# URL: https://data.cdc.gov/NCHS/NCHS-Leading-Causes-of-Death-United-States/bi63-dtpu
```

### Dashboard no carga completamente

```bash
# Verificar que el pipeline haya corrido
ls -la results/

# Ejecutar pipeline completo
python scripts/data_pipeline.py
python scripts/predictive_analysis.py

# Relanzar dashboard
streamlit run scripts/dashboard.py --server.port 8501
```

### Problemas de rendimiento

```bash
# Aumentar memoria Docker
# Docker Desktop -> Settings -> Resources -> Memory: 8GB+

# Verificar espacio en disco
df -h

# Limpiar contenedores no utilizados
make clean
docker system prune
```

### Logs para debugging

```bash
# Ver logs PostgreSQL
make logs

# Logs detallados del pipeline
python scripts/data_pipeline.py 2>&1 | tee debug.log

# Verificar conexión directa BD
make db-connect
```

## Extensiones y Mejoras

### Escalabilidad
- **Datos adicionales:** Integrar fuentes socioeconómicas, ambientales
- **Resolución geográfica:** Análisis a nivel condado/código postal  
- **Tiempo real:** Streaming analytics para datos actualizados
- **Cloud deployment:** AWS/Azure/GCP para mayor escala

### Modelos Avanzados
- **Deep Learning:** Redes neuronales para patrones no lineales
- **Análisis multivariado:** Interacciones entre causas de muerte
- **Análisis de supervivencia:** Predicciones individualizadas
- **Detección de anomalías:** Alertas automáticas por cambios inusuales

### Integraciones
- **APIs REST:** Endpoints para sistemas externos
- **Alertas automáticas:** Notificaciones por email/SMS
- **Export avanzado:** Formatos Tableau, Power BI
- **Programación:** Ejecución automática periódica

### Mejoras de Interfaz
- **Mobile responsive:** Optimización para dispositivos móviles
- **Autenticación:** Control de acceso por roles
- **Personalización:** Dashboards configurables por usuario
- **Colaboración:** Compartir análisis y reportes

---
