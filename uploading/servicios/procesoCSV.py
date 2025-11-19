"""
Servicio para procesar archivos CSV
"""

import csv
from uploading.interfaces.interfaz import InterfazCSV


class ProcesoCSV(InterfazCSV):
    """Implementación del servicio de procesamiento CSV"""
    
    def ejecutar(self, archivo_csv):
        """
        Lee y procesa un archivo CSV
        
        Args:
            archivo_csv: Ruta al archivo CSV
            
        Returns:
            Lista de filas del CSV
        """
        resultados = []
        
        try:
            with open(archivo_csv, newline='', encoding='utf-8') as f:
                # Intentar con diferentes delimitadores
                sample = f.read(1024)
                f.seek(0)
                
                # Detectar delimitador
                sniffer = csv.Sniffer()
                try:
                    delimiter = sniffer.sniff(sample).delimiter
                except:
                    delimiter = ','
                
                reader = csv.reader(f, delimiter=delimiter)
                
                for row in reader:
                    resultados.append(row)
        
        except Exception as e:
            raise Exception(f"Error al procesar CSV: {str(e)}")
        
        return resultados