"""
Algoritmos de normalización de datos
Incluye: Min-Max, Z-Score y Logarítmica
"""

import math


def min_max_normalization(columna, min_val=None, max_val=None, index=0):
    """Escalamiento lineal Min-Max: todos los valores se ajustan entre 0 y 1"""
    if min_val is None or max_val is None:
        min_val = min(columna)
        max_val = max(columna)
    if index >= len(columna):
        return []
    valor_norm = (columna[index] - min_val) / (max_val - min_val) if max_val != min_val else 0
    return [valor_norm] + min_max_normalization(columna, min_val, max_val, index + 1)


def z_score_normalization(columna, mean_val=None, std_val=None, index=0):
    """Ajuste Z: cada valor se centra en la media y se divide por desviación estándar"""
    if mean_val is None or std_val is None:
        mean_val = sum(columna) / len(columna)
        std_val = math.sqrt(sum((x - mean_val) ** 2 for x in columna) / len(columna))
    if index >= len(columna):
        return []
    valor_norm = (columna[index] - mean_val) / std_val if std_val != 0 else 0
    return [valor_norm] + z_score_normalization(columna, mean_val, std_val, index + 1)


def log_normalization(columna, index=0):
    """Escalamiento logarítmico: reduce diferencias grandes entre valores"""
    if index >= len(columna):
        return []
    valor_norm = math.log(columna[index] + 1)  # +1 para evitar log(0)
    return [valor_norm] + log_normalization(columna, index + 1)


def normalizar_tabla(tabla, columnas, metodo):
    """
    Normaliza las columnas especificadas de una tabla usando el método indicado
    
    Args:
        tabla: Lista de listas (matriz de datos)
        columnas: Lista de índices de columnas a normalizar
        metodo: 'min-max', 'z-score' o 'log'
    
    Returns:
        Tabla normalizada
    """
    tabla_norm = [fila[:] for fila in tabla]
    
    for col_index in columnas:
        columna = [fila[col_index] for fila in tabla]
        
        if metodo == 'min-max':
            columna_norm = min_max_normalization(columna)
        elif metodo == 'z-score':
            columna_norm = z_score_normalization(columna)
        elif metodo == 'log':
            columna_norm = log_normalization(columna)
        else:
            raise ValueError(f"Método no reconocido: {metodo}")
        
        for i in range(len(tabla)):
            tabla_norm[i][col_index] = columna_norm[i]
    
    return tabla_norm