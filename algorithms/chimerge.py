"""
Algoritmo ChiMerge para discretización de variables continuas
Basado en la prueba chi-cuadrado
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class ChiMerge:
    """
    Implementación del algoritmo ChiMerge para discretización de variables continuas.
    
    El algoritmo ChiMerge fusiona intervalos adyacentes basándose en la prueba
    de chi-cuadrado hasta alcanzar un umbral de significancia.
    """
    
    def __init__(self, max_intervals: int = 10, significance_level: float = 0.05):
        """
        Inicializa el algoritmo ChiMerge.
        
        Args:
            max_intervals: Número máximo de intervalos deseados
            significance_level: Nivel de significancia para la prueba chi-cuadrado
        """
        self.max_intervals = max_intervals
        self.significance_level = significance_level
        self.threshold = chi2.ppf(1 - significance_level, df=1)
        self.intervals_ = {}
        
    def _initialize_intervals(self, X: np.ndarray, y: np.ndarray) -> List[Dict]:
        """Inicializa los intervalos con cada valor único como un intervalo"""
        sorted_indices = np.argsort(X)
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        intervals = []
        unique_values = np.unique(X_sorted)
        
        for val in unique_values:
            mask = X_sorted == val
            y_interval = y_sorted[mask]
            
            classes, counts = np.unique(y_interval, return_counts=True)
            class_freq = dict(zip(classes, counts))
            
            interval = {
                'min': val,
                'max': val,
                'class_freq': class_freq,
                'total': len(y_interval)
            }
            intervals.append(interval)
            
        return intervals
    
    def _calculate_chi_square(self, interval1: Dict, interval2: Dict, 
                             all_classes: List) -> float:
        """Calcula el valor chi-cuadrado entre dos intervalos adyacentes"""
        chi_square = 0.0
        
        R1 = interval1['total']
        R2 = interval2['total']
        N = R1 + R2
        
        if N == 0:
            return float('inf')
        
        for cls in all_classes:
            A_ij = interval1['class_freq'].get(cls, 0)
            A_ij_plus_1 = interval2['class_freq'].get(cls, 0)
            C_j = A_ij + A_ij_plus_1
            
            if C_j == 0:
                continue
            
            E_ij = R1 * C_j / N
            E_ij_plus_1 = R2 * C_j / N
            
            if E_ij > 0:
                chi_square += (A_ij - E_ij) ** 2 / E_ij
            if E_ij_plus_1 > 0:
                chi_square += (A_ij_plus_1 - E_ij_plus_1) ** 2 / E_ij_plus_1
        
        return chi_square
    
    def _merge_intervals(self, interval1: Dict, interval2: Dict) -> Dict:
        """Fusiona dos intervalos adyacentes"""
        merged_class_freq = {}
        
        for cls in set(list(interval1['class_freq'].keys()) + 
                      list(interval2['class_freq'].keys())):
            merged_class_freq[cls] = (interval1['class_freq'].get(cls, 0) + 
                                     interval2['class_freq'].get(cls, 0))
        
        return {
            'min': interval1['min'],
            'max': interval2['max'],
            'class_freq': merged_class_freq,
            'total': interval1['total'] + interval2['total']
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_name: str = 'feature'):
        """Aplica el algoritmo ChiMerge a los datos"""
        intervals = self._initialize_intervals(X, y)
        all_classes = list(np.unique(y))
        
        while len(intervals) > self.max_intervals:
            chi_values = []
            for i in range(len(intervals) - 1):
                chi_val = self._calculate_chi_square(intervals[i], intervals[i + 1], 
                                                     all_classes)
                chi_values.append((chi_val, i))
            
            if not chi_values:
                break
            
            min_chi, min_idx = min(chi_values, key=lambda x: x[0])
            
            if min_chi > self.threshold and len(intervals) <= 2:
                break
            
            merged = self._merge_intervals(intervals[min_idx], intervals[min_idx + 1])
            intervals = intervals[:min_idx] + [merged] + intervals[min_idx + 2:]
        
        cut_points = [interval['max'] for interval in intervals[:-1]]
        self.intervals_[feature_name] = {
            'cut_points': cut_points,
            'intervals': intervals
        }
        
        return self
    
    def transform(self, X: np.ndarray, feature_name: str = 'feature') -> np.ndarray:
        """Transforma los datos usando los intervalos aprendidos"""
        if feature_name not in self.intervals_:
            raise ValueError(f"El atributo '{feature_name}' no ha sido ajustado")
        
        cut_points = self.intervals_[feature_name]['cut_points']
        discretized = np.digitize(X, cut_points)
        
        return discretized
    
    def get_interval_labels(self, feature_name: str = 'feature') -> List[str]:
        """Obtiene etiquetas descriptivas para cada intervalo"""
        if feature_name not in self.intervals_:
            raise ValueError(f"El atributo '{feature_name}' no ha sido ajustado")
        
        intervals = self.intervals_[feature_name]['intervals']
        labels = []
        
        for interval in intervals:
            if interval['min'] == interval['max']:
                label = f"[{interval['min']:.2f}]"
            else:
                label = f"[{interval['min']:.2f}, {interval['max']:.2f}]"
            labels.append(label)
        
        return labels