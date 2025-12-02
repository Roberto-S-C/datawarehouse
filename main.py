import logging
import os
from typing import Optional, Dict, Any, List
import pandas as pd
import yaml

from typing import Union
from fastapi import FastAPI, UploadFile, File
import yaml
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional
import json
from uuid import uuid4
from datetime import datetime
from collections import defaultdict
from fastapi.responses import HTMLResponse
from ranking_clientes import algoritmos
from fastapi.staticfiles import StaticFiles

# --------------------------- Utilidades ---------------------------

def setup_logging(level: str = "INFO"):
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s")


def read_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------- Extract ---------------------------
class Extractor:
    def __init__(self, conf: Dict[str, Any]):
        self.conf = conf or {}

    def extract_file(self, src: Dict[str, Any]) -> pd.DataFrame:
        stype = src.get("type", "csv").lower()
        path = src.get("path")
        if path is None:
            raise ValueError(f"Falta ruta en fuente {stype}")

        if stype == "csv":
            df = pd.read_csv(path)
        elif stype == "excel":
            sheet = src.get("sheet_name", 0)
            df = pd.read_excel(path, sheet_name=sheet)
        elif stype == "json":
            df = pd.read_json(path)
        else:
            raise NotImplementedError(f"Tipo de fuente no soportado: {stype}")

        logging.info(f"Archivo {path} ({stype}) leído con {len(df)} filas y {len(df.columns)} columnas.")
        return df

    def extract(self) -> pd.DataFrame:
        src_conf = self.conf.get("source", {})
        files: List[Dict[str, Any]] = src_conf.get("files", [])

        if not files:
            raise ValueError("Debe especificar al menos una fuente en source.files")

        dfs = []
        for f in files:
            df = self.extract_file(f)
            dfs.append(df)

        # Combinar los DataFrames automáticamente (merge o concat)
        try:
            combined = pd.concat(dfs, ignore_index=True, sort=False)
            logging.info(f"Archivos combinados: {len(combined)} filas, {len(combined.columns)} columnas.")
            return combined
        except Exception as e:
            logging.error(f"Error combinando archivos: {e}")
            raise


# --------------------------- Transform ---------------------------
class Transformer:
    def __init__(self, conf: Dict[str, Any]):
        self.conf = conf or {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        tconf = self.conf.get("transform", {})
        df = df.copy()

        if "rename" in tconf:
            df = df.rename(columns=tconf["rename"])

        if "drop_columns" in tconf:
            drops = [c for c in tconf["drop_columns"] if c in df.columns]
            df = df.drop(columns=drops)

        if "fillna" in tconf:
            for col, val in tconf["fillna"].items():
                if col in df.columns:
                    df[col] = df[col].fillna(val)

        if tconf.get("deduplicate", False):
            df = df.drop_duplicates()

        logging.info(f"Transformaciones completas: {len(df)} filas, {len(df.columns)} columnas.")
        return df


# --------------------------- Load ---------------------------
class Loader:
    def __init__(self, conf: Dict[str, Any]):
        self.conf = conf or {}

    def load(self, df: pd.DataFrame):
        lconf = self.conf.get("load", {})
        ltype = lconf.get("type", "csv").lower()

        if ltype == "csv":
            path = lconf.get("path")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            df.to_csv(path, index=False)
            logging.info(f"CSV escrito en {path}")
        elif ltype == "excel":
            path = lconf.get("path")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            df.to_excel(path, index=False)
            logging.info(f"Excel escrito en {path}")
        elif ltype == "json":
            path = lconf.get("path")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            df.to_json(path, orient="records", force_ascii=False)
            logging.info(f"JSON escrito en {path}")
        else:
            raise NotImplementedError(f"Tipo de carga no soportado: {ltype}")


# --------------------------- Orquestador / CLI ---------------------------
class ETLPipeline:
    def __init__(self, config_path: str):
        self.config = read_config(config_path)
        setup_logging(self.config.get("logging", {}).get("level", "INFO"))
        self.extractor = Extractor(self.config)
        self.transformer = Transformer(self.config)
        self.loader = Loader(self.config)

    def run_extract(self) -> pd.DataFrame:
        return self.extractor.extract()

    def run_transform(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.run_extract()
        return self.transformer.transform(df)

    def run_load(self, df: Optional[pd.DataFrame] = None):
        if df is None:
            df = self.run_transform()
        self.loader.load(df)


class ETLProcessTracker:
    def __init__(self):
        self.processes: Dict[str, Dict[str, Any]] = {}
        self.statistics = defaultdict(int)
    
    def start_process(self, process_id: str, files: List[str]):
        self.processes[process_id] = {
            "start_time": datetime.now(),
            "status": "running",
            "files": files,
            "current_phase": "starting",
            "errors": [],
            "statistics": {
                "rows_processed": 0,
                "columns_processed": 0,
                "duplicates_removed": 0,
                "null_values_filled": 0
            }
        }
        self.statistics["total_processes"] += 1
    
    def update_process(self, process_id: str, phase: str, stats: dict):
        if process_id in self.processes:
            self.processes[process_id]["current_phase"] = phase
            self.processes[process_id]["statistics"].update(stats)
    
    def complete_process(self, process_id: str, success: bool = True):
        if process_id in self.processes:
            self.processes[process_id].update({
                "end_time": datetime.now(),
                "status": "completed" if success else "failed",
                "duration": str(datetime.now() - self.processes[process_id]["start_time"])
            })
            self.statistics["completed_processes"] += 1
    
    def add_error(self, process_id: str, error: str):
        if process_id in self.processes:
            self.processes[process_id]["errors"].append({
                "time": datetime.now().isoformat(),
                "message": str(error)
            })
            self.statistics["errors"] += 1

# Create a global tracker instance
process_tracker = ETLProcessTracker()

#--------------------------- API ---------------------------
app = FastAPI()
app.mount("/imagenes", StaticFiles(directory="imagenes"), name="imagenes")

class FileTypeDetector:
    @staticmethod
    def detect_file_type(filename: str) -> str:
        """Detect file type from extension"""
        extension = filename.lower().split('.')[-1]
        type_mapping = {
            'csv': 'csv',
        }
        return type_mapping.get(extension, 'unknown')

def convert_numpy_values(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    return obj

@app.get("/", response_class=HTMLResponse)
def form():
    htmlContent = '' 
    with open(Path(Path.cwd(), 'src', 'form.html'), 'r', encoding='UTF-8') as file:
        htmlContent = file.read()
    return htmlContent

@app.post("/etl/", response_class=HTMLResponse)
async def etl_endpoint(
    files: List[UploadFile] = File(...),
    config: Optional[str] = None,
    phase: str = "all"
):
    # Guardar archivos
    os.makedirs(Path(Path.cwd(), 'datasets'), exist_ok=True)
    for file in files:
        contents = await file.read()
        with open(Path(Path.cwd(), 'datasets', file.filename), "wb") as f:
            f.write(contents)
            f.close()

    # Genera ID único para el proceso 
    process_id = str(uuid4())
    
    process_tracker.start_process(
        process_id,
        [file.filename for file in files]
    )
    
    # Directorio para resultados
    results_dir = Path("resultados")
    process_dir = results_dir
    process_dir.mkdir(parents=True, exist_ok=True)
    
    if len(files) == 0:
        return {
            "process_id": process_id,
            "error": "At least one file is required"
        }
    
    algoritmos()

    htmlContent = '' 
    with open(Path(Path.cwd(), 'src', 'results.html'), 'r', encoding='UTF-8') as file:
        htmlContent = file.read()
    return htmlContent

#-----------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
