# Ranking de Clientes - Tres Algoritmos de Machine Learning

## Descripción General

Este proyecto implementa **3 algoritmos simples de Machine Learning** para generar un ranking de clientes basado en su comportamiento de compra y los productos que adquieren.

Los datos provienen de la carpeta `datasets/` que contiene información sobre clientes, productos y ventas.

---

## Estructura de Datos

### Archivos CSV

#### 1. **clientes.csv** (1,000 registros)
- `id_cliente`: ID único del cliente
- `nombre`, `apellido`: Datos personales
- `email`, `telefono`: Contacto
- `direccion`, `ciudad`, `pais`: Ubicación
- `fecha_registro`: Cuándo se registró
- `segmento`: Categoría (Premium, Standard, Básico)

#### 2. **productos.csv** (1,000 registros)
- `id_producto`: ID único
- `nombre`: Nombre del producto
- `categoria`: Categoría (Libros, Electrónicos, Hogar, Deportes, Moda)
- `precio`, `costo`: Precios
- `stock`: Inventario
- `proveedor`: Proveedor
- `fecha_ingreso`: Cuándo se agregó al catálogo
- `activo`: Si está disponible (True/False)
- `rating`: Calificación (0-5 estrellas)

#### 3. **ventas.csv** (1,000 registros)
- `id_venta`: ID único de la transacción
- `id_cliente`: Referencia al cliente
- `id_producto`: Referencia al producto
- `fecha`: Fecha y hora de la venta
- `cantidad`: Unidades vendidas
- `precio_unitario`: Precio por unidad
- `descuento`: Descuento aplicado
- `impuestos`: Impuestos
- `metodo_pago`: Efectivo, Tarjeta, Transferencia, Cripto
- `vendedor`: Nombre del vendedor
- `total`: Monto total

---

## Archivo Principal: `ranking_clientes.py`

### Sección 1: Importaciones (Líneas 1-8)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

**Librerías utilizadas:**
- **pandas**: Manejo de datos en tablas (DataFrames)
- **numpy**: Operaciones matemáticas
- **train_test_split**: Divide datos en 70% entrenamiento y 30% prueba
- **StandardScaler**: Normaliza los datos
- **Tres algoritmos de ML**: Regresión Logística, KNN, Random Forest
- **accuracy_score**: Mide el % de predicciones correctas

---

### Sección 2: Cargar Datos (Líneas 10-17)

```python
clientes = pd.read_csv('datasets/clientes.csv')
productos = pd.read_csv('datasets/productos.csv')
ventas = pd.read_csv('datasets/ventas.csv')
```

Cargamos los 3 archivos CSV necesarios para el análisis.

---

### Sección 3: Preparación de Datos (Líneas 19-32)

#### Paso 1: Agregar datos por cliente

```python
cliente_stats = ventas.groupby('id_cliente').agg({
    'total': ['sum', 'mean', 'count'],
    'cantidad': 'sum'
}).reset_index()
```

Agrupamos todas las compras de cada cliente y calculamos:
- **gasto_total**: Total que gastó el cliente
- **gasto_promedio**: Promedio por compra
- **num_compras**: Cuántas compras realizó
- **cantidad_items**: Cuántos productos compró en total

```python
cliente_stats.columns = ['id_cliente', 'gasto_total', 'gasto_promedio', 'num_compras', 'cantidad_items']
```

#### Paso 2: Calcular diversidad de categorías

```python
ventas_cat = ventas.merge(productos[['id_producto', 'categoria']], on='id_producto')
categorias = ventas_cat.groupby('id_cliente')['categoria'].nunique().reset_index()
```

Calculamos **num_categorias**: Cuántas categorías de productos diferentes compró cada cliente.

#### Paso 3: Combinar datos

```python
cliente_stats = cliente_stats.merge(categorias, on='id_cliente', how='left')
```

Combinamos la información usando `merge` (como un JOIN en SQL).

**Resultado: Tabla con 627 clientes y 5 características cada uno**

---

### Sección 4: Crear Etiqueta (Target) (Líneas 34-40)

```python
mediana_gasto = cliente_stats['gasto_total'].median()
cliente_stats['es_buen_cliente'] = (cliente_stats['gasto_total'] >= mediana_gasto).astype(int)
```

Dividimos los clientes en **2 categorías**:
- **1 = Buen Cliente**: Gasto >= mediana (superior al 50%)
- **0 = Cliente Regular**: Gasto < mediana (inferior al 50%)

**Resultado: 314 buenos clientes | 313 clientes regulares**

```python
X = cliente_stats[['gasto_promedio', 'num_compras', 'cantidad_items', 'num_categorias']]
y = cliente_stats['es_buen_cliente']
```

- **X** = Features (características): Las 4 variables predictoras
- **y** = Target (etiqueta): Lo que queremos predecir (0 ó 1)

---

### Sección 5: Normalizar Datos (Líneas 42-44)

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**¿Por qué es importante?**

Los datos tienen diferentes escalas:
- `gasto_promedio`: 1000-5000
- `num_compras`: 1-10
- `cantidad_items`: 5-50
- `num_categorias`: 1-5

Esto puede sesgar los algoritmos. `StandardScaler` convierte todo a una escala común:
- Media = 0
- Desviación estándar = 1

**Resultado:** Todos los datos en la misma escala, los algoritmos funcionan mejor.

---

### Sección 6: Dividir Datos (Línea 47)

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
```

Dividimos los datos en:
- **70% (X_train, y_train)**: Para entrenar el modelo (que aprenda patrones)
- **30% (X_test, y_test)**: Para probar si el modelo aprendió bien

`random_state=42` asegura que siempre hagamos la misma división (reproducibilidad).

---

### Sección 7: Regresión Logística (Líneas 52-77)

#### Entrenamiento

```python
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
```

**¿Qué hace Regresión Logística?**

Encuentra una "línea de separación" que divide los buenos clientes de los regulares. Es como trazar una frontera en un gráfico.

#### Evaluación

```python
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
```

Hacemos predicciones en datos de prueba y medimos el **accuracy** (% de aciertos).

**Resultado: 96.83% de precisión**

#### Predicción y Ranking

```python
all_probs_lr = lr_model.predict_proba(X_scaled)[:, 1]
```

`predict_proba` devuelve **probabilidades** (0-1) para cada cliente:
- Valor cercano a **1**: Probable que sea "buen cliente"
- Valor cercano a **0**: Probable que sea "cliente regular"

```python
ranking_lr = cliente_stats.sort_values('score_lr', ascending=False)
```

Ordenamos los clientes por puntuación (mejor a peor).

#### Top 5 Clientes (Regresión Logística)

```
ranking  id_cliente  score_lr
       1         344       1.0
       2         244       1.0
       3         647       1.0
       4         404       1.0
       5         419       1.0
```

---

### Sección 8: K-Nearest Neighbors (KNN) (Líneas 79-104)

#### Entrenamiento

```python
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
```

**¿Qué hace KNN?**

Mira a los **5 vecinos más cercanos** de cada cliente. Si la mayoría de sus vecinos son buenos clientes, él también lo será.

**Analogía:** Si todos tus amigos son ricos, probablemente tú también seas rico.

#### Evaluación

```python
knn_accuracy = accuracy_score(y_test, knn_pred)
```

**Resultado: 95.77% de precisión**

#### Top 5 Clientes (KNN)

```
ranking  id_cliente  score_knn
       1           6        1.0
       2        1000        1.0
       3           1        1.0
       4         977        1.0
       5         981        1.0
```

---

### Sección 9: Random Forest (Líneas 106-131)

#### Entrenamiento

```python
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(X_train, y_train)
```

**¿Qué hace Random Forest?**

Crea **10 árboles de decisión** diferentes. Cada árbol "vota" si el cliente es bueno o regular. **La mayoría gana**.

**Analogía:** Como preguntar a 10 expertos y usar la respuesta más común.

#### Evaluación

```python
rf_accuracy = accuracy_score(y_test, rf_pred)
```

**Resultado: 96.83% de precisión**

#### Top 5 Clientes (Random Forest)

```
ranking  id_cliente  score_rf
       1           6       1.0
       2        1000       1.0
       3           1       1.0
       4         977       1.0
       5         981       1.0
```

---

### Sección 10: Comparación y Exportación (Líneas 133-162)

#### Comparación de Algoritmos

```
Accuracy en datos de prueba:
  Regresión Logística: 0.9683 (96.83%)
  KNN:                 0.9577 (95.77%)
  Random Forest:       0.9683 (96.83%)
```

**Análisis:**
- Regresión Logística y Random Forest empatan con **96.83%**
- KNN está muy cerca con **95.77%**
- Los 3 algoritmos funcionan muy bien (>95%)

#### Archivos Generados

```python
ranking_lr.to_csv('ranking_logistic_regression.csv', index=False)
ranking_knn.to_csv('ranking_knn.csv', index=False)
ranking_rf.to_csv('ranking_random_forest.csv', index=False)
```

Se generan **3 archivos CSV** con los rankings:
1. **ranking_logistic_regression.csv**
2. **ranking_knn.csv**
3. **ranking_random_forest.csv**

Cada archivo contiene:
- `id_cliente`: ID del cliente
- `gasto_total`: Total gastado
- `num_compras`: Número de compras
- `score_[algoritmo]`: Probabilidad de ser buen cliente (0-1)
- `ranking`: Posición en el ranking (1 = mejor)

---

## Flujo de Ejecución

```
DATOS CRUDOS (3 archivos CSV)
           ↓
   PREPARAR DATOS
   ├─ Calcular métricas por cliente
   ├─ Contar categorías de productos
   └─ Crear etiqueta (buen cliente = 1, regular = 0)
           ↓
   NORMALIZAR DATOS
   └─ Escalar todas las variables a escala común
           ↓
   DIVIDIR DATOS
   ├─ 70% para entrenamiento
   └─ 30% para prueba
           ↓
   ENTRENAR 3 MODELOS
   ├─→ Regresión Logística
   ├─→ K-Nearest Neighbors
   └─→ Random Forest
           ↓
   EVALUAR EN DATOS DE PRUEBA
   └─ Calcular accuracy de cada modelo
           ↓
   PREDECIR PARA TODOS LOS CLIENTES
   └─ Obtener probabilidades (0-1)
           ↓
   CREAR RANKINGS
   └─ Ordenar clientes por probabilidad
           ↓
   GUARDAR EN CSV
   ├─ ranking_logistic_regression.csv
   ├─ ranking_knn.csv
   └─ ranking_random_forest.csv
```

---

## Cómo Usar

### Requisitos

```bash
pip install pandas scikit-learn numpy
```

### Ejecutar el Script

```bash
python ranking_clientes.py
```

### Salida

El script genera:
1. **Salida en consola** con:
   - Resumen de datos (total de clientes, buenos clientes, regulares)
   - Accuracy de cada modelo
   - Top 10 clientes por cada algoritmo
   - Top 5 clientes con comparación de los 3 algoritmos

2. **3 archivos CSV** con los rankings completos

### Analizar Resultados

Puedes abrir los archivos CSV en:
- Excel
- Google Sheets
- Pandas (Python)
- SQL
- Cualquier herramienta de análisis de datos

---

## Características de los Clientes (Features)

El modelo utiliza **4 características** para predecir si un cliente es "buen cliente":

| Feature | Rango | Significado |
|---------|-------|-------------|
| `gasto_promedio` | 1000-5000 | Cuánto gasta en promedio por compra |
| `num_compras` | 1-10 | Cuántas veces ha comprado |
| `cantidad_items` | 5-50 | Cuántos productos ha comprado en total |
| `num_categorias` | 1-5 | Cuántas categorías diferentes ha comprado |

**Etiqueta (Target):**
- `es_buen_cliente`: 1 si gasto_total >= mediana, 0 si no

---

## Interpretación de los Algoritmos

### Regresión Logística

- **Ventaja:** Simple, rápida, interpretable
- **Desventaja:** Asume relación lineal
- **Caso de uso:** Cuando necesitas explicar por qué un cliente es bueno
- **Accuracy:** 96.83%

### K-Nearest Neighbors (KNN)

- **Ventaja:** No hace suposiciones, flexible
- **Desventaja:** Más lenta, sensible a datos ruidosos
- **Caso de uso:** Cuando tienes datos limpios y bien definidos
- **Accuracy:** 95.77%

### Random Forest

- **Ventaja:** Robusto, maneja relaciones complejas, da importancia de variables
- **Desventaja:** Puede sobreajustar
- **Caso de uso:** Cuando tienes datos complejos
- **Accuracy:** 96.83%
