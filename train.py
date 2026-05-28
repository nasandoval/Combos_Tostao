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
    
    # Creamos dos listas globales vacías al inicio de la fase
    registros_tecnicos_completos = []
    registros_comerciales_limpios = []
    
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
                # Ordenamos inicialmente por Lift de mayor a menor (Fuerza de asociación)
                reglas = rules = reglas.sort_values(by='lift', ascending=False).reset_index(drop=True)
                
                # Creamos la columna temporal para identificar las combinaciones comerciales únicas (sin importar el orden)
                reglas['combo_id_limpio'] = reglas.apply(
                    lambda r: " + ".join(sorted(list(set(list(r['antecedents']) + list(r['consequents']))))), 
                    axis=1
                )
                
                # -------------------------------------------------------------
                # ENFOQUE 1: REPORTE TÉCNICO COMPLETO (Conserva todas las reglas)
                # -------------------------------------------------------------
                top_5_tecnico = reglas.head(5)
                
                for idx, row in top_5_tecnico.iterrows():
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
                    
                    porcentaje_desc = 0.15
                    precio_original_combo = sum([maestro_precios.get(prod, 0) for prod in combo_completo])
                    precio_sugerido_combo = int(np.round(precio_original_combo * (1 - porcentaje_desc)))
                    
                    # El log en MLflow registra todo el ecosistema de reglas detalladas
                    mlflow.log_param(f"tech_combo_{idx+1}_productos", " + ".join(combo_completo))
                    mlflow.log_metric(f"tech_combo_{idx+1}_lift", float(row['lift']))
                    mlflow.log_metric(f"tech_combo_{idx+1}_confidence", float(row['confidence']))
                    
                    registros_tecnicos_completos.append({
                        "Combo_ID": idx + 1,
                        "Cluster_ID": clus,
                        "Nombre_Cluster": nombres_format[clus],
                        "Antecedente": " + ".join(ant),
                        "Consecuente": " + ".join(cons),
                        "Productos_Combo": " + ".join(combo_completo),
                        "Precio_Original": precio_original_combo,
                        "Precio_Sugerido_15%": precio_sugerido_combo,
                        "Lift": round(row['lift'], 2),
                        "Confianza": round(row['confidence'], 2),
                        "Top_3_Tiendas": top_tiendas_str,
                        "Dia_Mayor_Venta": dia_fuerte
                    })
                
                # -------------------------------------------------------------
                # ENFOQUE 2: REPORTE COMERCIAL LIMPIO (Filtra duplicados y espejos)
                # -------------------------------------------------------------
                reglas_limpias = reglas.drop_duplicates(subset=['combo_id_limpio'], keep='first').reset_index(drop=True)
                top_5_comercial = reglas_limpias.head(5)
                
                for idx, row in top_5_comercial.iterrows():
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
                    
                    porcentaje_desc = 0.15
                    precio_original_combo = sum([maestro_precios.get(prod, 0) for prod in combo_completo])
                    precio_sugerido_combo = int(np.round(precio_original_combo * (1 - porcentaje_desc)))
                    ahorro_cliente = precio_original_combo - precio_sugerido_combo
                    
                    registros_comerciales_limpios.append({
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

    # FASE 5: Exportación e Impresión de los Dos Informes Diferenciados
    print("\nFase 5: Exportación de entregables y reportes finales...")
    
    # Forzar a Pandas a imprimir el ancho completo de las tablas en consola sin recortar columnas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # 1. Exportar e Imprimir el reporte técnico completo
    if registros_tecnicos_completos:
        df_tech = pd.DataFrame(registros_tecnicos_completos)
        ruta_tech = "data/reporte_tecnico_todas_las_reglas.csv"
        df_tech.to_csv(ruta_tech, index=False, encoding='utf-8-sig')
        print(f"Reporte Técnico Completo (Con reglas espejo) guardado en: {ruta_tech}")
        
        print("\n" + "="*95)
        print("VISTA PREVIA: REPORTE TÉCNICO COMPLETO (TODAS LAS REGLAS DISPONIBLES)")
        print("="*95)
        print(df_tech[['Cluster_ID', 'Combo_ID', 'Antecedente', 'Consecuente', 'Lift', 'Confianza']].to_string(index=False))
        print("="*95 + "\n")
        
    # 2. Exportar e Imprimir el reporte comercial definitivo y limpio
    if registros_comerciales_limpios:
        df_comercial = pd.DataFrame(registros_comerciales_limpios)
        ruta_comercial = "data/reporte_combos_sugeridos.csv"
        df_comercial.to_csv(ruta_comercial, index=False, encoding='utf-8-sig')
        print(f"Reporte Comercial Ejecutivo (Sin duplicados) guardado en: {ruta_comercial}")
        
        print("\n" + "="*95)
        print("VISTA PREVIA: REPORTE COMERCIAL EJECUTIVO (COMBOS ÚNICOS FILTRADOS)")
        print("="*95)
        print(df_comercial[['Cluster_ID', 'Combo_ID', 'Productos', 'Precio_Original_Total', 'Precio_Sugerido_Tostao', 'Ahorro_Cliente', 'Lift', 'Dia_Mayor_Venta']].to_string(index=False))
        print("="*95 + "\n")

if __name__ == "__main__":
    ejecutar_pipeline_entrenamiento(ruta_datos_limpios="data/datos_tostao_limpios.csv")