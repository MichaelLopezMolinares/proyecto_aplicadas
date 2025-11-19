"""
Interfaz base para procesamiento de CSV
"""


class InterfazCSV:
    """Interfaz para procesar archivos CSV"""
    
    def ejecutar(self, archivo_csv):
        """
        Ejecuta el procesamiento del archivo CSV
        
        Args:
            archivo_csv: Ruta al archivo CSV
            
        Returns:
            Resultados del procesamiento
        """
        raise NotImplementedError("Debes implementar este método")