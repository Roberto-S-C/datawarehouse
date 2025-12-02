import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
from pathlib import Path

def algoritmos():
     # Cargar datos
    datasets_dir = Path(Path.cwd(), 'datasets')
    csv_files = list(datasets_dir.glob('*.csv'))
    
    clientes = pd.read_csv(csv_files[0])
    productos = pd.read_csv(csv_files[1])
    ventas = pd.read_csv(csv_files[2])

    print("=" * 80)
    print("RANKING DE CLIENTES - TRES ALGORITMOS ML")
    print("=" * 80)

    # Preparar datos: calcular métricas por cliente
    cliente_stats = ventas.groupby('id_cliente').agg({
        'total': ['sum', 'mean', 'count'],
        'cantidad': 'sum'
    }).reset_index()

    cliente_stats.columns = ['id_cliente', 'gasto_total', 'gasto_promedio', 'num_compras', 'cantidad_items']

    # Unir con datos de productos para contar categorías diferentes
    ventas_cat = ventas.merge(productos[['id_producto', 'categoria']], on='id_producto')
    categorias = ventas_cat.groupby('id_cliente')['categoria'].nunique().reset_index()
    categorias.columns = ['id_cliente', 'num_categorias']

    cliente_stats = cliente_stats.merge(categorias, on='id_cliente', how='left')

    # Crear etiqueta: "Buen Cliente" si gasto_total está en top 50%
    mediana_gasto = cliente_stats['gasto_total'].median()
    cliente_stats['es_buen_cliente'] = (cliente_stats['gasto_total'] >= mediana_gasto).astype(int)

    # Preparar features y target
    X = cliente_stats[['gasto_promedio', 'num_compras', 'cantidad_items', 'num_categorias']]
    y = cliente_stats['es_buen_cliente']

    # Escalar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir datos: 70% entrenamiento, 30% prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    print(f"\nDatos: {len(cliente_stats)} clientes")
    print(f"Buenos clientes: {(y == 1).sum()} | Clientes regulares: {(y == 0).sum()}")

    # ============================================================================
    # ALGORITMO 1: REGRESIÓN LOGÍSTICA
    # ============================================================================
    print("\n" + "=" * 80)
    print("1. REGRESIÓN LOGÍSTICA")
    print("=" * 80)

    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)

    print(f"\nAccuracy: {lr_accuracy:.4f}")
    print(f"\nProbabilidades (primeros 5 clientes de prueba):")
    lr_probs = lr_model.predict_proba(X_test)[:5]
    for i, prob in enumerate(lr_probs):
        print(f"  Cliente {i+1}: Buen Cliente: {prob[1]:.2%}")

    # Predecir para todos los clientes
    all_probs_lr = lr_model.predict_proba(X_scaled)[:, 1]
    cliente_stats['score_lr'] = all_probs_lr
    ranking_lr = cliente_stats.sort_values('score_lr', ascending=False)[['id_cliente', 'gasto_total', 'num_compras', 'score_lr']].reset_index(drop=True)
    ranking_lr['ranking'] = range(1, len(ranking_lr) + 1)

    print(f"\nTop 10 clientes (Regresión Logística):")
    print(ranking_lr[['ranking', 'id_cliente', 'score_lr']].head(10).to_string(index=False))

    # ============================================================================
    # ALGORITMO 2: K-NEAREST NEIGHBORS (KNN)
    # ============================================================================
    print("\n" + "=" * 80)
    print("2. K-NEAREST NEIGHBORS (KNN)")
    print("=" * 80)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_pred)

    print(f"\nAccuracy: {knn_accuracy:.4f}")
    print(f"\nPredicciones (primeros 5 clientes de prueba):")
    knn_probs = knn_model.predict_proba(X_test)[:5]
    for i, prob in enumerate(knn_probs):
        print(f"  Cliente {i+1}: Buen Cliente: {prob[1]:.2%}")

    # Predecir para todos los clientes
    all_probs_knn = knn_model.predict_proba(X_scaled)[:, 1]
    cliente_stats['score_knn'] = all_probs_knn
    ranking_knn = cliente_stats.sort_values('score_knn', ascending=False)[['id_cliente', 'gasto_total', 'num_compras', 'score_knn']].reset_index(drop=True)
    ranking_knn['ranking'] = range(1, len(ranking_knn) + 1)

    print(f"\nTop 10 clientes (KNN):")
    print(ranking_knn[['ranking', 'id_cliente', 'score_knn']].head(10).to_string(index=False))

    # ============================================================================
    # ALGORITMO 3: RANDOM FOREST
    # ============================================================================
    print("\n" + "=" * 80)
    print("3. RANDOM FOREST")
    print("=" * 80)

    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    print(f"\nAccuracy: {rf_accuracy:.4f}")
    print(f"\nPredicciones (primeros 5 clientes de prueba):")
    rf_probs = rf_model.predict_proba(X_test)[:5]
    for i, prob in enumerate(rf_probs):
        print(f"  Cliente {i+1}: Buen Cliente: {prob[1]:.2%}")

    # Predecir para todos los clientes
    all_probs_rf = rf_model.predict_proba(X_scaled)[:, 1]
    cliente_stats['score_rf'] = all_probs_rf
    ranking_rf = cliente_stats.sort_values('score_rf', ascending=False)[['id_cliente', 'gasto_total', 'num_compras', 'score_rf']].reset_index(drop=True)
    ranking_rf['ranking'] = range(1, len(ranking_rf) + 1)

    print(f"\nTop 10 clientes (Random Forest):")
    print(ranking_rf[['ranking', 'id_cliente', 'score_rf']].head(10).to_string(index=False))

    # ============================================================================
    # RESUMEN COMPARATIVO
    # ============================================================================
    print("\n" + "=" * 80)
    print("COMPARACIÓN DE ALGORITMOS")
    print("=" * 80)

    print(f"\nAccuracy en datos de prueba:")
    print(f"  Regresión Logística: {lr_accuracy:.4f}")
    print(f"  KNN:                 {knn_accuracy:.4f}")
    print(f"  Random Forest:       {rf_accuracy:.4f}")

    print(f"\n\nTop 5 clientes por cada algoritmo:")
    print("\n[REGRESIÓN LOGÍSTICA]:")
    print(ranking_lr[['ranking', 'id_cliente', 'gasto_total', 'num_compras', 'score_lr']].head(5).to_string(index=False))
    print("\n[KNN]:")
    print(ranking_knn[['ranking', 'id_cliente', 'score_knn']].head(5).to_string(index=False))
    print("\n[RANDOM FOREST]:")
    print(ranking_rf[['ranking', 'id_cliente', 'score_rf']].head(5).to_string(index=False))

    # Exportar resultados
    ranking_lr.to_csv(Path(Path.cwd(), 'resultados', 'ranking_logistic_regression.csv'), index=False)
    ranking_knn.to_csv(Path(Path.cwd(), 'resultados', 'ranking_knn.csv'), index=False)
    ranking_rf.to_csv(Path(Path.cwd(), 'resultados', 'ranking_random_forest.csv'), index=False)

    print("\n\n[ARCHIVOS GENERADOS]:")
    print("   - ranking_logistic_regression.csv")
    print("   - ranking_knn.csv")
    print("   - ranking_random_forest.csv")
    print("\n" + "=" * 80)
    graficas(lr_accuracy, knn_accuracy, rf_accuracy, cliente_stats)

# =============================================================================
# GRAFICAS CON MATPLOTLIB
# =============================================================================

# ---------------------------
# Gráfica de accuracy
# ---------------------------
def graficas(lr_accuracy, knn_accuracy, rf_accuracy, cliente_stats):
    algoritmos = ['Regresión Logística', 'KNN', 'Random Forest']
    accuracies = [lr_accuracy, knn_accuracy, rf_accuracy]
    os.makedirs(Path(Path.cwd(), 'imagenes'), exist_ok=True)
    

    plt.figure(figsize=(10, 5))
    plt.bar(algoritmos, accuracies)
    plt.title('Comparación de Accuracy entre Algoritmos')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(Path(Path.cwd(), 'imagenes', 'grafica_accuracy_algoritmos.png'))
    plt.close()

    print("\n[GRÁFICA GENERADA]: grafica_accuracy_algoritmos.png")


    # ---------------------------
    # Gráfica de score promedio en todo el dataset
    # ---------------------------

    promedios_score = [
        cliente_stats['score_lr'].mean(),
        cliente_stats['score_knn'].mean(),
        cliente_stats['score_rf'].mean()
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(algoritmos, promedios_score, color='orange')
    plt.title('Score Promedio por Algoritmo (Dataset Completo)')
    plt.ylabel('Score promedio')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(Path(Path.cwd(), 'imagenes', 'grafica_score_promedio_algoritmos.png'))
    plt.close()

    print("[GRÁFICA GENERADA]: grafica_score_promedio_algoritmos.png")

    # -----------------------------
    # Gráfica: Top 5 compradores
    # -----------------------------
    plt.figure(figsize=(8,5))

    top5 = cliente_stats.sort_values('gasto_total', ascending=False).head(5)

    plt.bar(top5['id_cliente'].astype(str), top5['gasto_total'])
    plt.title("Top 5 Clientes por Gasto Total")
    plt.xlabel("ID Cliente")
    plt.ylabel("Gasto Total")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(Path(Path.cwd(), 'imagenes',"top5_compradores.png"))