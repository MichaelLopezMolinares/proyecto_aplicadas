"""
Controlador para manejar la subida y procesamiento de archivos CSV
"""

from uploading.interfaces.interfaz import InterfazCSV


class ControladorSubida:
    """Controlador para procesar archivos CSV subidos"""
    
    def __init__(self, servicio_csv: InterfazCSV):
        """
        Inicializa el controlador
        
        Args:
            servicio_csv: Servicio que implementa InterfazCSV
        """
        self.servicio = servicio_csv
    
    def procesar(self, ruta_archivo):
        """
        Procesa el archivo CSV usando el servicio configurado
        
        Args:
            ruta_archivo: Ruta al archivo a procesar
            
        Returns:
            Resultado del procesamiento
        """
        return self.servicio.ejecutar(ruta_archivo)