# ☕ Market Basket Analysis & MLOps - Combos Tostao

Este proyecto implementa un sistema de **Minería de Reglas de Asociación** utilizando el algoritmo **Apriori** para identificar patrones de coocurrencia y afinidad en las transacciones comerciales de Tostao. El ciclo de vida de los experimentos, la sintonización de hiperparámetros y el análisis de sensibilidad están gestionados y registrados mediante **MLflow**.

## 📊 Arquitectura del Experimento (Análisis de Sensibilidad)

Se diseñó un análisis de sensibilidad sobre el hiperparámetro de **Soporte Mínimo (`min_support`)**, manteniendo una **Confianza Mínima (`min_confidence`)** fija del 10% ($0.10$) para mitigar el subajuste debido a la alta dispersión (*sparsity*) del catálogo de consumo masivo.

A través de **MLflow Tracking**, se evaluaron 4 configuraciones clave:

* **Soporte Estricto (1.0% y 0.5%):** Modelos estables que capturan patrones redundantes u obvios (p.ej., Café $\rightarrow$ Buñuelo), limitando la innovación comercial a un máximo de 19 reglas.
* **Soporte Extremo (0.1%):** Provoca una **explosión combinatoria** (257 reglas), introduciendo ruido estadístico y artefactos aleatorios (Lift inestable de 13.77).
* **Punto de Sintonización Óptimo (0.2%):** Configuración seleccionada (**52 reglas descubiertas**). Representa el balance ideal entre frecuencia de negocio y el descubrimiento de "joyas ocultas" de alto valor estratégico, alcanzando un **Lift máximo de 10.14**.

---

## 🛠️ Estructura del Proyecto

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