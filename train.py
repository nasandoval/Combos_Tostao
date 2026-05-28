import pandas as pd
import numpy as np
import mlflow
import warnings
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules

# Se importa la función modular de codificación binaria desde preprocess.py
from preprocess import aplicar_one_hot_encoding

warnings.filterwarnings("ignore")

def ejecutar_pipeline_entrenamiento(ruta_datos_limpios: str):
    """
    Ejecuta el pipeline end-to-end de MLOps consumiendo directamente
    el dataset unificado y ordenado cronológicamente.
    """
    # Se establece el experimento unificado en el servidor de MLflow
    mlflow.set_experiment("Reporte_Final_Combos_Estrategicos")
    
    print("Fase 1: Extracción y partición temporal de datos...")
    
    # 1. Carga del dataset único
    df_completo = pd.read_csv(ruta_datos_limpios)
    df_completo['fecha'] = pd.to_datetime(df_completo['fecha'])
    
    # Asegurar el orden cronológico estricto del histórico
    df_completo = df_completo.sort_values(by='fecha').reset_index(drop=True)
    
    # PASO CLAVE: Crear un maestro de precios reales por producto para la consulta rápida
    maestro_precios = df_completo.drop_duplicates(subset=['nombre']).set_index('nombre')['precio_unitario'].to_dict()
    
    # Lista global para consolidar el reporte físico que irá a GitHub
    registros_reporte_final = []
    
    # 2. Generar un espejo a nivel de ticket único para agrupar variables numéricas
    df_tickets_unicos = df_completo.drop_duplicates(subset=['id_ticket']).copy()
    
    # 3. Aplicar el split temporal (80% entrenamiento) basado en los tickets únicos ordenados
    punto_corte = int(len(df_tickets_unicos) * 0.80)
    df_tickets_train = df_tickets_unicos.iloc[:punto_corte].copy()
    
    # 4. El dataset de entrenamiento final para Apriori conserva todo el detalle de esos tickets
    df_train = df_completo[df_completo['id_ticket'].isin(df_tickets_train['id_ticket'])].copy()
    
    # Selección de variables numéricas claves para realizar la segmentación (K-Means)
    features = ['cantidad', 'precio_unitario', 'venta_total'] 
    X = df_tickets_unicos[features]
    
    print("Fase 2: Escalamiento de variables y Reducción de dimensiones con PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print("Fase 3: Ajuste del modelo de Clustering K-Means (K=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_tickets_unicos['cluster_final'] = kmeans.fit_predict(X_scaled)
    
    # Pasamos el clúster asignado a la tabla de entrenamiento detallada
    mapa_clusters = df_tickets_unicos.set_index('id_ticket')['cluster_final']
    df_train['cluster_final'] = df_train['id_ticket'].map(mapa_clusters)
    
    clusters_analisis = [1, 2, 3]
    nombres_format = {
        1: "Clientes de Canasta Grande y Alto Gasto",
        2: "Clientes de Canasta Surtida Moderada",
        3: "Clientes de Cantidad Focalizada"
    }
    
    dias_espanol = {
        'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
        'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    
    print("Fase 4: Extracción de Combos Estratégicos por Clúster con Apriori...")
    for clus in clusters_analisis:
        with mlflow.start_run(run_name=f"Combos_Contexto_Cluster_{clus}"):
            
            # Filtrar los datos detallados que correspondan al clúster actual
            df_det_clus = df_train[df_train['cluster_final'] == clus].copy()
            
            if df_det_clus.empty:
                continue
                
            # Extraemos y traducimos el día de la semana real
            df_det_clus['nombre_dia'] = df_det_clus['fecha'].dt.day_name().map(dias_espanol)
            
            # Construimos la matriz binaria mapeando el nombre comercial del producto
            matriz = aplicar_one_hot_encoding(df_det_clus)
            
            freq_items = apriori(matriz, min_support=0.01, use_colnames=True)
            reglas = association_rules(freq_items, metric="lift", min_threshold=1.0)
            
            if not reglas.empty:
                reglas = reglas.sort_values(by='lift', ascending=False).reset_index(drop=True)
                top_5 = reglas.head(5)
                
                for idx, row in top_5.iterrows():
                    ant = list(row['antecedents'])
                    cons = list(row['consequents'])
                    combo_completo = list(set(ant + cons))
                    
                    tickets_con_combo = matriz[matriz[combo_completo].all(axis=1)].index
                    df_combo_contexto = df_det_clus[df_det_clus['id_ticket'].isin(tickets_con_combo)]
                    contexto_unico_ticket = df_combo_contexto.drop_duplicates(subset=['id_ticket'])
                    
                    if not contexto_unico_ticket.empty:
                        top_tiendas = contexto_unico_ticket['id_tienda'].value_counts().head(3).index.tolist()
                        top_tiendas_str = ", ".join([str(t) for t in top_tiendas])
                        dia_fuerte = contexto_unico_ticket['nombre_dia'].mode()[0]
                    else:
                        top_tiendas_str, dia_fuerte = "N/A", "N/A"
                    
                    # ESTRATEGIA DE NEGOCIO: Cálculo matemático de Precios con Descuento (15%)
                    porcentaje_desc = 0.15
                    precio_original_combo = sum([maestro_precios.get(prod, 0) for prod in combo_completo])
                    precio_sugerido_combo = int(np.round(precio_original_combo * (1 - porcentaje_desc)))
                    ahorro_cliente = precio_original_combo - precio_sugerido_combo
                    
                    # Log de parámetros y métricas estructuradas en el servidor de MLflow
                    mlflow.log_param(f"combo_{idx+1}_productos", " + ".join(combo_completo))
                    mlflow.log_param(f"combo_{idx+1}_tiendas_top", top_tiendas_str)
                    mlflow.log_param(f"combo_{idx+1}_dia_semana", str(dia_fuerte))
                    mlflow.log_metric(f"combo_{idx+1}_precio_original", float(precio_original_combo))
                    mlflow.log_metric(f"combo_{idx+1}_precio_sugerido", float(precio_sugerido_combo))
                    mlflow.log_metric(f"combo_{idx+1}_lift", float(row['lift']))
                    mlflow.log_metric(f"combo_{idx+1}_confidence", float(row['confidence']))
                    
                    # Añadir fila al reporte de datos físico (Estructura definitiva)
                    registros_reporte_final.append({
                        "Combo_ID": idx + 1,
                        "Cluster_ID": clus,
                        "Nombre_Cluster": nombres_format[clus],
                        "Productos": " + ".join(combo_completo),
                        "Precio_Original_Total": precio_original_combo,
                        "Precio_Sugerido_Tostao": precio_sugerido_combo,
                        "Ahorro_Cliente": ahorro_cliente,
                        "Lift": round(row['lift'], 2),
                        "Confianza": round(row['confidence'], 2),
                        "Top_3_Tiendas": top_tiendas_str,
                        "Dia_Mayor_Venta": dia_fuerte
                    })
                    
                print(f"Clúster {clus} ({nombres_format[clus]}) procesado y enviado a MLflow.")
                
    # FASE 5: Exportación automática (Alineada correctamente dentro de la función)
    if registros_reporte_final:
        df_reporte = pd.DataFrame(registros_reporte_final)
        ruta_reporte = "data/reporte_combos_sugeridos.csv"
        df_reporte.to_csv(ruta_reporte, index=False, encoding='utf-8-sig')
       

if __name__ == "__main__":
    # Invocación directa apuntando al archivo único unificado con el nombre exacto
    ejecutar_pipeline_entrenamiento(ruta_datos_limpios="data/datos_tostao_limpios.csv")