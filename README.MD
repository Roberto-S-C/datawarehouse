# Proyecto de ETL para procesamiento de datos

Este proyecto proporciona una API para procesar archivos de datos a través de un proceso ETL (Extracción, Transformación y Carga). Permite cargar archivos de diferentes formatos y verificar el estado del proceso.

## Instalación de dependencias

Instala las dependencias requeridas utilizando pip:

pip install -r requirements.txt

## Ejecutar el programa

Para iniciar la aplicación, simplemente ejecuta el siguiente comando:

python main.py

## Llamar a los Endpoints

### Endpoint para cargar archivos

Utiliza este endpoint para cargar los archivos a procesar. Puedes subir archivos en formatos .json, .csv y .xlsx.

curl -X POST "http://127.0.0.1:8000/etl/" \
  -F "files=@/ruta/a/tu/archivo/productos.json" \
  -F "files=@/ruta/a/tu/archivo/clientes.csv" \
  -F "files=@/ruta/a/tu/archivo/ventas.xlsx" \
  -F "phase=all"

Asegúrate de reemplazar las rutas de los archivos con las rutas correctas en tu sistema.

### Endpoint para ver el estado del proceso

Este endpoint te permitirá consultar el estado de un proceso ETL en particular. Debes proporcionar un process_id válido.

curl "http://127.0.0.1:8000/etl/status?process_id=ac3965bd-9139-408f-b7be-22a55c3415e9"