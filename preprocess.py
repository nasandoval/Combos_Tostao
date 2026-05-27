import pandas as pd
import numpy as np

def aplicar_one_hot_encoding(df_insumo: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma el formato transaccional a una estructura matricial binaria (One-Hot)
    agrupando los productos comprados por cada ticket único.
    """
    # 1. Agrupamos las cantidades por ticket y nombre de producto, pasándolas a columnas
    matriz_pivot = df_insumo.groupby(['id_ticket', 'nombre'])['cantidad'].sum().unstack().reset_index().fillna(0)
    
    # 2. Establecemos el identificador del ticket como el índice principal de la matriz
    matriz_pivot.set_index('id_ticket', inplace=True)
    
    # 3. Convertimos cualquier cantidad mayor a cero en un uno (1) y la ausencia en cero (0)
    matriz_binaria = matriz_pivot.map(lambda x: 1 if x > 0 else 0)
    
    return matriz_binaria