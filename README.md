# Optimización de Combos Estratégicos - Tostao
**Rol:** Lead Data Scientist  
**Caso de Uso:** Análisis de Canasta de Compras y Segmentación Clientes 

# Objetivo del proyecto

Implementar un algoritmo para identificar los Top 5 "Combos" (conjuntos de productos) con mayor potencial de venta (proponer el precio de cada combo) para diferentes clusters de tiendas, cuantificando el "lift" esperado.

---

El pipeline de datos está estructurado en las siguientes fases cronológicas:

1. **Preprocesamiento y Split Temporal:** Se cargaron y limpiaron las transacciones. Para evitar el sesgo de entrenar y validar con datos mezclados, se aplicó una división basada en el tiempo (Time-Based Split). El primer 80% de la data histórica se utilizó para construir los modelos y el 20% más reciente se reservó para pruebas.
   
2. **Segmentación de Clientes (K-Means + PCA):**
   * Se agruparon los tickets según el volumen de compra, precio unitario y ventas totales utilizando K-Means.
   * Se aplicó Análisis de Componentes Principales (PCA) para reducir las dimensiones y poder visualizar la separación de los grupos en un plano bidimensional.
   * Se determinó que la configuración óptima era de 4 clústeres mediante el análisis conjunto del Método del Codo (Inercia) y el Coeficiente de Silueta.

3. **Minería de Patrones de Consumo (Algoritmo Apriori):**
   Se transformó el detalle de los tickets a una matriz binaria (One-Hot Encoding) para evaluar la co-ocurrencia de productos por cada clúster. Con esto, se extrajeron reglas de asociación filtrando por soporte, confianza y un Lift mayor a 1.0.

4. **Contextualización Comercial y Estrategia de Pricing:**
   Las reglas de asociación resultantes se cruzaron con las variables de `id_tienda` y `fecha` (calculando el día de la semana) para determinar las sucursales y días de mayor impacto. Adicionalmente, el pipeline integra un **módulo de analítica comercial** que extrae los precios unitarios de un maestro de datos y aplica una **heurística de descuento fijo del 15%** (seleccionada tras una evaluación multiescenario de sensibilidad al 10%, 15% y 20% en la fase de experimentación), logrando el equilibrio óptimo entre el estímulo psicológico del consumidor y la protección del margen del retail.

5. **Gobernanza del Modelo con MLOps (MLflow):**
   Cada ejecución del algoritmo, los hiperparámetros utilizados (como el `min_support`), las métricas de afinidad (Lift/Confianza), los precios sugeridos calculados y los gráficos de dispersión de los clústeres quedaron registrados y versionados localmente en el servidor de MLflow.

6. **Generación de Entregables de Negocio:**
   Al finalizar el entrenamiento, el pipeline compila y exporta de manera automatizada un reporte físico estructurado en `data/reporte_combos_sugeridos.csv`. Este archivo actúa como el puente de comunicación directa entre el equipo de Ciencia de Datos y los *stakeholders* de las áreas de Operaciones, Inventario y Pricing de Tostao'.

--- 
## Estructura del Proyecto

```text
Combos_Tostao/
├── data/                          # Depósito local de los datasets y reportes generados
│   ├── datos_tostao_limpios.csv   # Dataset histórico unificado y limpio
│   ├── reporte_combos_sugeridos.csv # Reporte comercial ejecutivo (Combos únicos filtrados)
│   └── reporte_tecnico_todas_las_reglas.csv # Reporte técnico completo (Con todas las reglas espejo)
├── evidencia_mlruns/              # último run, dando evidencia de la utilización de MLflow
├── env_tostao/                    # Entorno virtual con las dependencias del proyecto (Ignorado en Git)
├── mlruns/                        # Métricas locales y tracking automatizado de MLflow (Ignorado en Git)
├── notebooks/                     # Jupyter Notebooks con el ciclo de experimentos preliminares
├── .gitignore                     # Archivo de exclusión de datos pesados y temporales
├── preprocess.py                  # Script modular para la transformación binaria (One-Hot Encoding)
├── train.py                       # Pipeline principal de producción (Clustering + Apriori + Pricing + MLflow)
├── README.md                      # Documentación escrita y guías de ejecución del proyecto
└── requirements.txt               # Listado oficial de librerías para reproducción del entorno

```

### Instrucciones de Instalación y Ejecución
Para replicar el entorno de desarrollo y ejecutar este pipeline, se deben seguir estos pasos en la terminal:


### 1. Clonar el repositorio y entrar a la carpeta

```bash
git clone https://github.com/nasandoval/Combos_Tostao.git
cd Combos_Tostao

```
### 2. Crear y activar el entorno virtual local
Crear el entorno virtual en la raíz:

```bash
python -m venv env_tostao

```

Activar el entorno virtual (En Windows - PowerShell):

```bash
.\env_tostao\Scripts\Activate.ps1

```

### 3. Instalar las dependencias del proyecto
Con el entorno activo, ejecute el instalador automatizado para descargar las versiones exactas de las librerías utilizadas:

```bash
pip install -r requirements.txt

```

### 4. Ejecutar e interactuar con la interfaz de MLflow
Para auditar la trazabilidad del modelo, revisar los hiperparámetros del clustering, las métricas de afinidad (Lift) y el reporte del Top 5 de combos estratégicos de cada clúster, levante el servidor local con:

```bash
mlflow ui

```

Una vez encendido el servidor, abra su navegador web (Chrome/Edge) e ingrese a la siguiente dirección de hosting local:

[http://127.0.0.1:5000](http://127.0.0.1:5000)