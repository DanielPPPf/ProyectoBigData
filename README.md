# Pipeline de Datos - Causas Principales de Muerte en EE.UU.
## Versión Simplificada (PostgreSQL + Python)

Un pipeline ETL eficiente y minimalista para análisis de las principales causas de muerte en Estados Unidos utilizando datos del NCHS (National Center for Health Statistics).

## 🎯 Objetivo del Proyecto

Este proyecto implementa un pipeline ETL simplificado que:
- Procesa datos históricos de mortalidad (1999-2017)
- Limpia y valida la calidad de los datos
- Crea un modelo de datos dimensional en PostgreSQL
- Genera reportes CSV automáticos
- Mantiene una infraestructura mínima y eficiente

## 🏗️ Arquitectura Simplificada

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Datos CSV     │───▶│   PostgreSQL    │───▶│   Reportes CSV  │
│   (NCHS)        │    │   (3 Esquemas)  │    │   & Análisis    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                       ┌─────────────────┐
                       │  Pipeline Python│
                       │   (Scripts)     │
                       └─────────────────┘
```

### Componentes:

- **PostgreSQL 15**: Base de datos principal con 3 esquemas:
  - `raw_data`: Datos sin procesar
  - `clean_data`: Modelo dimensional (fact + dimensions)  
  - `analytics`: Métricas calculadas
- **Pipeline Python**: Scripts automatizados de procesamiento
- **Reportes CSV**: Exportación automática de resultados

## 🚀 Inicio Rápido

### Prerrequisitos
- Docker y Docker Compose
- Make (opcional pero recomendado)
- Al menos 2GB de RAM disponible

### 1. Clonar y configurar
```bash
# Crear estructura del proyecto
mkdir deaths-analysis && cd deaths-analysis

# Crear directorios necesarios
make create-dirs
# O manualmente:
mkdir -p data scripts init-db results config
```

### 2. Obtener los datos
```bash
# Colocar tu archivo CSV en la carpeta data/
cp tu_archivo.csv data/NCHS__Leading_Causes_of_Death__United_States.csv
```

### 3. Levantar la infraestructura
```bash
# Opción 1: Usando Make (recomendado)
make setup

# Opción 2: Docker Compose directo
docker-compose build
docker-compose up -d postgres
```

### 4. Ejecutar el pipeline
```bash
# Ejecutar pipeline completo
make pipeline

# Ver resultados
make view-results
```

## 📊 Servicio Disponible

| Servicio | URL | Credenciales |
|----------|-----|-------------|
| **PostgreSQL** | localhost:5432 | postgres / postgres123 |

## 🗂️ Estructura del Proyecto

```
deaths-analysis/
├── docker-compose.yml          # Configuración simplificada
├── Dockerfile.python           # Imagen Python para pipeline
├── requirements.txt            # Dependencias Python
├── Makefile                    # Automatización de tareas
├── README.md                   # Esta documentación
├── .gitignore                  # Archivos a ignorar
│
├── data/                       # Datos fuente
│   └── NCHS__Leading_Causes_of_Death__United_States.csv
│
├── init-db/                    # Scripts de inicialización DB
│   └── 01-create-schema.sql
│
├── scripts/                    # Scripts Python
│   └── data_pipeline.py        # Pipeline principal
│
├── config/                     # Configuraciones opcionales
│   └── dev_config.json
│
└── results/                    # Resultados y reportes
    ├── national_trends.csv
    ├── state_analysis.csv
    ├── cause_trends.csv
    └── pipeline_stats.json
```

## 🔄 Flujo del Pipeline

### 1. **Extracción (Extract)**
- Carga automática del archivo CSV
- Validación de estructura
- Logging de estadísticas iniciales

### 2. **Transformación (Transform)**
- Limpieza de nombres de columnas
- Validación de tipos de datos
- Detección y corrección de problemas:
  - Valores nulos o vacíos
  - Duplicados
  - Valores fuera de rango
  - Outliers extremos

### 3. **Carga (Load)**
- **Raw Data**: `raw_data.deaths_raw`
- **Dimensional Model**:
  - `clean_data.dim_states`: Dimensión de estados
  - `clean_data.dim_causes`: Dimensión de causas
  - `clean_data.fact_deaths`: Tabla de hechos principal
- **Analytics**: `analytics.yearly_summary`

### 4. **Reportes**
- Exportación automática a CSV
- Estadísticas del pipeline
- Logs detallados de ejecución

## 📈 Reportes Generados

### Archivos CSV Automáticos
1. **national_trends.csv**: Tendencias nacionales por año y causa
2. **state_analysis.csv**: Análisis comparativo por estado  
3. **cause_trends.csv**: Evolución temporal de cada causa
4. **pipeline_stats.json**: Estadísticas de calidad y ejecución

## 🛠️ Comandos Principales

### Gestión de Servicios
```bash
make up              # Levantar PostgreSQL
make down            # Detener servicios  
make status          # Ver estado
make logs            # Ver logs
```

### Pipeline de Datos
```bash
make pipeline        # Ejecutar pipeline completo
make check-data      # Verificar archivo CSV
make view-results    # Mostrar resumen de resultados
make quick-analysis  # Análisis rápido desde BD
```

### Base de Datos
```bash
make db-connect      # Conectar a PostgreSQL
make db-query        # Sesión interactiva SQL
make db-backup       # Crear backup
make db-reset        # Resetear BD
```

### Análisis y Exportación
```bash
make export-data     # Exportar dataset completo
make quick-analysis  # Estadísticas rápidas
```

## 📊 Estructura de la Base de Datos

### Esquema Raw Data
```sql
raw_data.deaths_raw (
    id SERIAL PRIMARY KEY,
    year INTEGER,
    cause_name_113 TEXT,
    cause_name TEXT, 
    state TEXT,
    deaths INTEGER,
    age_adjusted_death_rate DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### Esquema Clean Data (Modelo Dimensional)
```sql
-- Tabla de Hechos Principal
clean_data.fact_deaths (
    death_id SERIAL PRIMARY KEY,
    year INTEGER NOT NULL,
    state_id INTEGER REFERENCES dim_states(state_id),
    cause_id INTEGER REFERENCES dim_causes(cause_id), 
    deaths INTEGER NOT NULL,
    age_adjusted_death_rate DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- Dimensión Estados
clean_data.dim_states (
    state_id SERIAL PRIMARY KEY,
    state_name VARCHAR(100) NOT NULL UNIQUE,
    is_national_aggregate BOOLEAN DEFAULT FALSE,
    state_code CHAR(2),
    region VARCHAR(50)
)

-- Dimensión Causas
clean_data.dim_causes (
    cause_id SERIAL PRIMARY KEY,
    cause_name VARCHAR(200) NOT NULL UNIQUE,
    cause_name_113 TEXT,
    cause_category VARCHAR(100),
    is_all_causes BOOLEAN DEFAULT FALSE
)
```

### Esquema Analytics
```sql
analytics.yearly_summary (
    summary_id SERIAL PRIMARY KEY,
    year INTEGER NOT NULL,
    total_deaths INTEGER,
    avg_death_rate DECIMAL(10,2),
    leading_cause VARCHAR(200),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

## 🔍 Validaciones de Calidad

### Controles Automáticos
- ✅ Tipos de datos correctos
- ✅ Campos críticos no nulos  
- ✅ Rangos válidos de años (1990-2030)
- ✅ Valores no negativos en muertes
- ✅ Detección de outliers (percentil 99)
- ✅ Eliminación de duplicados exactos
- ✅ Integridad referencial

### Métricas Reportadas
- Registros procesados vs. removidos
- Completitud por año
- Estados y causas identificados
- Tiempo de ejecución del pipeline

## 🚨 Solución de Problemas

### PostgreSQL No Inicia
```bash
make logs            # Ver logs de error
make db-reset        # Resetear si es necesario
docker system prune  # Limpiar recursos
```

### Archivo de Datos No Encontrado
```bash
# Verificar ubicación exacta
ls -la data/
# Debe existir: data/NCHS__Leading_Causes_of_Death__United_States.csv
```

### Pipeline Falla  
```bash
make logs-pipeline   # Ver logs detallados
make test-connection # Probar conexión DB
make pipeline-dev    # Ejecutar con configuración de desarrollo
```

### Ver Estado General
```bash
make status          # Estado de servicios
make monitor         # Monitoreo en tiempo real
make quick-analysis  # Verificar datos en BD
```

## 🔧 Personalización

### Configuración Personalizada
```bash
# Crear configuración personalizada
cp config/dev_config.json config/mi_config.json
# Editar según necesidades

# Usar configuración personalizada
docker-compose run --rm python_pipeline python scripts/data_pipeline.py --config /app/config/mi_config.json
```

### Variables de Entorno
```bash
# En docker-compose.yml, sección python_pipeline:
environment:
  - DB_HOST=postgres
  - DB_NAME=deaths_analysis  
  - DB_USER=postgres
  - DB_PASSWORD=tu_password_personalizada
```

### Añadir Nuevos Análisis
```bash
# Conectar y ejecutar SQL personalizado
make db-query

# Exportar resultados personalizados
make db-connect
# Luego: \copy (SELECT ...) TO '/results/mi_reporte.csv' WITH CSV HEADER;
```

## 📚 Comandos SQL Útiles

### Exploratorios Básicos
```sql
-- Conectar con: make db-connect

-- Ver estructura de tablas
\dt clean_data.*

-- Resumen rápido  
SELECT COUNT(*) FROM clean_data.fact_deaths;
SELECT COUNT(*) FROM clean_data.dim_states;  
SELECT COUNT(*) FROM clean_data.dim_causes;

-- Top causas de muerte
SELECT dc.cause_name, SUM(fd.deaths) as total_deaths
FROM clean_data.fact_deaths fd
JOIN clean_data.dim_causes dc ON fd.cause_id = dc.cause_id
WHERE dc.is_all_causes = false
GROUP BY dc.cause_name
ORDER BY total_deaths DESC;
```

## ⚡ Rendimiento

### Optimizaciones Incluidas
- Índices automáticos en campos clave
- Carga por lotes (método `multi`)
- Conexiones pooled con SQLAlchemy
- Validación eficiente de datos

### Para Datasets Grandes
```bash
# Usar configuración optimizada
echo '{"processing": {"validate_data": false}}' > config/fast_config.json
docker-compose run --rm python_pipeline python scripts/data_pipeline.py --config /app/config/fast_config.json
```

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama: `git checkout -b feature/mi-mejora`
3. Commit: `git commit -m 'Agregar nueva funcionalidad'`  
4. Push: `git push origin feature/mi-mejora`
5. Abre un Pull Request

## 📄 Licencia

Proyecto bajo Licencia MIT - ver [LICENSE](LICENSE) para detalles.

## 🆘 Soporte

Para problemas o preguntas:

1. Revisar [Solución de Problemas](#-solución-de-problemas)
2. Ejecutar `make logs` para diagnósticos
3. Usar `make help` para ver todos los comandos
4. Abrir issue en el repositorio

---

**Pipeline Simplificado y Eficiente! 🚀📊**