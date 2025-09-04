# Makefile para el Pipeline de Datos - Causas de Muerte EEUU
# Version simplificada para conda

DOCKER_COMPOSE = docker-compose
DATA_FILE = NCHS__Leading_Causes_of_Death__United_States.csv

# Colores
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m

.PHONY: help up down logs clean pipeline

help:
	@echo "Pipeline de Datos - Causas Principales de Muerte en EE.UU."
	@echo "Comandos disponibles:"
	@echo "  help              - Mostrar ayuda"
	@echo "  up               - Levantar PostgreSQL"
	@echo "  down             - Detener servicios"
	@echo "  wait-for-db      - Esperar a PostgreSQL"
	@echo "  test-connection  - Probar conexión"
	@echo "  check-data       - Verificar archivo CSV"
	@echo "  check-python     - Verificar dependencias Python"
	@echo "  simple-analysis  - Ejecutar análisis simple"
	@echo "  db-connect       - Conectar a PostgreSQL"
	@echo "  status           - Ver estado de servicios"

up:
	@echo "Levantando PostgreSQL..."
	$(DOCKER_COMPOSE) up -d postgres
	@echo "PostgreSQL iniciado en localhost:5432"

down:
	@echo "Deteniendo servicios..."
	$(DOCKER_COMPOSE) down

wait-for-db:
	@echo "Esperando a que PostgreSQL esté listo..."
	@echo -n "Verificando conexión"
	@for i in $$(seq 1 30); do \
		if docker exec deaths_postgres pg_isready -U postgres -d deaths_analysis >/dev/null 2>&1; then \
			echo " ✓"; \
			echo "PostgreSQL está listo"; \
			exit 0; \
		fi; \
		echo -n "."; \
		sleep 2; \
	done; \
	echo " ✗"; \
	echo "Timeout esperando PostgreSQL"; \
	exit 1

check-data:
	@if [ ! -f "data/$(DATA_FILE)" ]; then \
		echo "✗ Archivo no encontrado: data/$(DATA_FILE)"; \
		echo "Coloca el archivo CSV en la carpeta 'data/'"; \
		exit 1; \
	else \
		echo "✓ Archivo de datos encontrado"; \
	fi

check-python:
	@echo "Verificando dependencias de Python..."
	@python -c "import pandas, psycopg2, sqlalchemy" 2>/dev/null || \
		(echo "✗ Faltan dependencias"; \
		 echo "Instala con: conda install pandas psycopg2 sqlalchemy"; \
		 exit 1)
	@echo "✓ Dependencias de Python OK"

test-connection:
	@echo "Probando conexión a PostgreSQL..."
	@python -c "\
import psycopg2; \
conn = psycopg2.connect( \
	host='localhost', \
	database='deaths_analysis', \
	user='postgres', \
	password='postgres123', \
	port=5432 \
); \
print('✓ Conexión exitosa'); \
conn.close()"

simple-analysis:
	@echo "Ejecutando análisis simple..."
	python scripts/simple_analysis.py

db-connect:
	$(DOCKER_COMPOSE) exec postgres psql -U postgres -d deaths_analysis

status:
	@echo "Estado de los servicios:"
	$(DOCKER_COMPOSE) ps

logs:
	$(DOCKER_COMPOSE) logs -f postgres

clean:
	@echo "Limpiando proyecto..."
	$(DOCKER_COMPOSE) down -v --remove-orphans

create-dirs:
	@echo "Creando directorios..."
	mkdir -p data scripts init-db results config
	@echo "✓ Directorios creados"
predictive-analysis:
	@echo "Ejecutando análisis predictivo..."
	python scripts/predictive_analysis.py
dashboard:
	@echo "Iniciando dashboard interactivo..."
	streamlit run scripts/dashboard.py --server.port 8501