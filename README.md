# Optimización de Combos Estratégicos - Tostao' 
**Rol:** Lead Data Scientist
**Caso de Uso:** Análisis de Canasta de Compras y Segmentación Clientes (End-to-End)

# Objetivo del proyecto

Implementar un algoritmo para identificar los Top 5 "Combos" (conjuntos de productos) con mayor potencial de venta (proponer el precio de cada combo) para diferentes clusters de tiendas, cuantificando el "lift" esperado.

Para lograr esto, la solución no se limita a un análisis estadístico general, sino que segmenta el comportamiento de compra mediante Clustering y luego extrae las reglas de asociación específicas para cada perfil de cliente, gobernando todo el ciclo de experimentos con MLflow.

---

## Arquitectura de la Solución (Flujo End-to-End)

El pipeline de datos está estructurado en las siguientes fases cronológicas:

1. **Preprocesamiento y Split Temporal:** Se cargaron y limpiaron las transacciones. Para evitar el sesgo de entrenar y validar con datos mezclados, se aplicó una división basada en el tiempo (Time-Based Split). El primer 80% de la data histórica se utilizó para construir los modelos y el 20% más reciente se reservó para pruebas.
   
2. **Segmentación de Clientes (K-Means + PCA):**
   * Se agruparon los tickets según el volumen de compra, precio unitario y ventas totales utilizando K-Means.
   * Se aplicó Análisis de Componentes Principales (PCA) para reducir las dimensiones y poder visualizar la separación de los grupos en un plano bidimensional.
   * Se determinó que la configuración óptima era de 4 clústeres mediante el análisis conjunto del Método del Codo (Inercia) y el Coeficiente de Silueta.

3. **Minería de Patrones de Consumo (Algoritmo Apriori):**
   Se transformó el detalle de los tickets a una matriz binaria (One-Hot Encoding) para evaluar la co-ocurrencia de productos por cada clúster. Con esto, se extrajeron reglas de asociación filtrando por soporte, confianza y un Lift mayor a 1.0.

4. **Contextualización Comercial (Días y Tiendas):**
   Las reglas de asociación resultantes se cruzaron con las variables de `id_tienda` y `fecha` (calculando el día de la semana) para determinar con precisión matemática los 5 combos más fuertes de cada grupo, junto con las 3 sucursales donde más se venden y su día de mayor impacto.

5. **Gobernanza del Modelo con MLOps (MLflow):**
   Cada ejecución del algoritmo, los hiperparámetros utilizados (como el `min_support`), las métricas de afinidad (Lift/Confianza) y los gráficos de dispersión de los clústeres quedaron registrados localmente en el servidor de MLflow.

--- 

##  Estructura del Proyecto

```text
Estructura del Proyecto
Combos_Tostao/
├── data/                 # Deposito local de la data
├── env_tostao/           # Entorno virtual con las dependencias del proyecto (Ignorado en Git)
├── mlruns/               # Métricas locales de MLflow (Ignorado en Git)
├── notebooks/            # Jupyter Notebooks con el ciclo de experimentos
├── .gitignore            # Archivo de exclusión de datos pesados y temporales
├── README.md             # Documentación escrita del proyecto
└── requirements.txt      # Listado oficial de librerías para reproducción del entorno


## Instrucciones de Instalación y Ejecución

Para replicar el entorno de desarrollo y ejecutar este pipeline, se deben seguir estos pasos en la terminal:

### 1. Clonar el repositorio y entrar a la carpeta
```bash
git clone [https://github.com/tu_usuario/Tesis-MLOps-Combos-Tostao.git](https://github.com/tu_usuario/Tesis-MLOps-Combos-Tostao.git)
cd Tesis-MLOps-Combos-Tostao

### 2. Crear y activar el entorno virtual local

# Crear el entorno virtual en la raíz
python -m venv env_tostao

# Activar el entorno virtual (En Windows - PowerShell):
.\env_tostao\Scripts\Activate.ps1

### 3. Instalar las dependencias del proyecto

Con el entorno activo, ejecute el instalador automatizado para descargar las versiones exactas de las librerías utilizadas:

pip install -r requirements.txt

### 4. Ejecutar e interactuar con la interfaz de MLflow

Para auditar la trazabilidad del modelo, revisar los hiperparámetros del clustering, las métricas de afinidad (Lift) y el reporte del Top 5 de combos estratégicos de cada clúster, levante el servidor local con:

mlflow ui

Una vez encendido el servidor, abra su navegador web (Chrome/Edge) e ingrese a la siguiente dirección de hosting local:
http://127.0.0.1:5000

