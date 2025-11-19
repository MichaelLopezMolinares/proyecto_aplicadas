"""
Sistema Integrado de Minería de Datos
Incluye: K-Medias, K-Modas, Normalización, Discretización, Relleno de Valores Faltantes, Árbol de Decisión
"""

import io
import base64
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Importar los algoritmos personalizados
from algorithms.chimerge import ChiMerge
from algorithms.normalizacion import normalizar_tabla, min_max_normalization, z_score_normalization, log_normalization

app = Flask(__name__, static_folder="static", template_folder="templates")

# Configuraciones globales para K-Means y K-Modes
K_MIN_ELBOW = 2
K_MAX_ELBOW = 10
N_INIT = 10
MAX_ITER = 100
RANDOM_STATE = 42


@app.route("/")
def home():
    return send_from_directory("templates", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se encontró archivo"}), 400

        file = request.files['file']
        if file.filename == "":
            return jsonify({"error": "Nombre de archivo vacío"}), 400

        # Obtener el algoritmo seleccionado
        algoritmo = request.form.get('select', 'arbol')

        # Leer CSV
        try:
            df = pd.read_csv(file, sep=None, engine="python")
        except Exception as e:
            return jsonify({"error": f"Error al leer CSV: {str(e)}"}), 400

        if df.empty:
            return jsonify({"error": "El archivo CSV está vacío"}), 400

        # Ejecutar el algoritmo seleccionado
        if algoritmo == 'arbol':
            return ejecutar_arbol_decision(df)
        elif algoritmo == 'kmedias':
            return ejecutar_kmedias(df)
        elif algoritmo == 'kmodos':
            return ejecutar_kmodos(df)
        elif algoritmo == 'normalizacion':
            return ejecutar_normalizacion(df, request.form)
        elif algoritmo == 'discretizacion':
            return ejecutar_discretizacion(df, request.form)
        else:
            return jsonify({"error": "Algoritmo no reconocido"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def ejecutar_arbol_decision(df):
    """Árbol de Decisión - Predice valores faltantes marcados con '?'"""
    target_col = df.columns[-1]
    
    # Identificar filas con valores faltantes en el target
    mask_missing = (df[target_col].astype(str) == '?') | df[target_col].isna()
    df_train = df[~mask_missing].copy()
    df_missing = df[mask_missing].copy()

    if df_train.empty:
        return jsonify({"error": "No hay datos de entrenamiento (todas las filas tienen target faltante)"}), 400

    # Preparar datos de entrenamiento
    X_train = df_train.iloc[:, :-1].copy()
    y_train = df_train.iloc[:, -1].copy()

    # Limpiar y codificar
    for col in X_train.columns:
        if X_train[col].dtype == object:
            X_train[col] = X_train[col].fillna('MISSING').astype(str)
        else:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(X_train[col].median())

    X_train_encoded = pd.get_dummies(X_train, drop_first=False)
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train.astype(str))

    # Entrenar modelo
    clf = DecisionTreeClassifier(random_state=42, max_depth=10)
    clf.fit(X_train_encoded, y_train_encoded)

    # Generar imagen del árbol
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(clf, feature_names=X_train_encoded.columns, 
              class_names=[str(c) for c in le.classes_], 
              filled=True, max_depth=4, ax=ax)
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=120)
    plt.close(fig)
    img_buf.seek(0)
    img_b64 = base64.b64encode(img_buf.read()).decode('ascii')
    data_uri = "data:image/png;base64," + img_b64

    # Predecir valores faltantes
    predictions = []
    if not df_missing.empty:
        X_missing = df_missing.iloc[:, :-1].copy()
        
        for col in X_missing.columns:
            if X_missing[col].dtype == object:
                X_missing[col] = X_missing[col].fillna('MISSING').astype(str)
            else:
                X_missing[col] = pd.to_numeric(X_missing[col], errors='coerce').fillna(X_train[col].median())
        
        X_missing_encoded = pd.get_dummies(X_missing, drop_first=False)
        X_missing_encoded = X_missing_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
        
        y_pred = clf.predict(X_missing_encoded)
        y_pred_original = le.inverse_transform(y_pred)
        
        for idx, pred_val in zip(df_missing.index, y_pred_original):
            row_dict = df.loc[idx, X_missing.columns].to_dict()
            predictions.append({
                "index": int(idx),
                "row": row_dict,
                "predicted": str(pred_val)
            })

    # Crear DataFrame con predicciones
    df_filled = df.copy()
    for p in predictions:
        df_filled.at[p['index'], target_col] = p['predicted']

    preview = df_filled.head(50).to_dict(orient="records")

    return jsonify({
        "message": "Árbol de Decisión completado",
        "target_column": target_col,
        "predictions": predictions,
        "preview": preview,
        "tree_image_datauri": data_uri
    })


def ejecutar_kmedias(df):
    """K-Medias - Clustering de datos numéricos"""
    df_original = df.copy()
    df = df.replace("?", np.nan)
    # convertir a numérico solo las columnas con datos numéricos válidos
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().any():
                df[col] = converted
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    reference_col = None
    reference_categories = None
    for col in cat_cols:
        unique_vals = pd.Series(df_original[col]).dropna().unique()
        if len(unique_vals) > 1 and len(unique_vals) <= K_MAX_ELBOW:
            reference_col = col
            reference_categories = unique_vals
            break
        
    X_all = df[numeric_cols].copy()
    X_all = X_all.dropna(axis=1, how='all')
    
    numeric_cols = X_all.columns.tolist()
    missing_mask = X_all.isna()
    row_missing = missing_mask.any(axis=1)
    
    X_filled = X_all.fillna(X_all.median())
    df[numeric_cols] = X_filled
    
    X_train = X_filled[~row_missing].copy()
    if X_train.empty:
        return jsonify({"error": "No hay filas completas para entrenar K-Medias"}), 400
   
    # Determinar mejor K usando método del codo con filas completas
    if reference_col:
        target_k = min(len(reference_categories), len(X_train))
        k_min = k_max = max(2, target_k)
    else:
        k_min = max(1, min(K_MIN_ELBOW, len(X_train)))
        k_max = max(k_min, min(K_MAX_ELBOW, len(X_train)))
    
    inertias = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        kmeans.fit(X_train)
        inertias.append((k, kmeans.inertia_))
    
    # Método del codo simplificado
    best_k = elegir_k_por_codo(inertias)
    
    # Entrenar con mejor K usando filas completas
    kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=N_INIT)
    kmeans_final.fit(X_train)
    labels_train = kmeans_final.labels_
    cluster_centers_df = pd.DataFrame(kmeans_final.cluster_centers_, columns=numeric_cols)
    
    label_series = pd.Series(index=df.index, dtype=int)
    label_series.loc[X_train.index] = labels_train
    
    if row_missing.any():
        predictions = kmeans_final.predict(X_filled.loc[row_missing])
        label_series.loc[row_missing[row_missing].index] = predictions
    
    category_cluster_map = {}
    if cat_cols:
        for col in cat_cols:
            value_map = {}
            col_values = df_original[col]
            for cluster_idx in range(best_k):
                cluster_rows = X_train.index[labels_train == cluster_idx]
                if len(cluster_rows) == 0:
                    continue
                values = col_values.loc[cluster_rows].dropna()
                if values.empty:
                    continue
                majority_val = values.mode().iloc[0]
                value_map[majority_val] = cluster_idx
            if value_map:
                category_cluster_map[col] = value_map
    
    # Generar visualización
    cluster_plot_uri = None
    if X_filled.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_filled.iloc[:, 0], X_filled.iloc[:, 1],
                             c=label_series.values, cmap='viridis', s=50, alpha=0.6)
        ax.scatter(kmeans_final.cluster_centers_[:, 0],
                   kmeans_final.cluster_centers_[:, 1],
                   marker='X', s=200, c='red', edgecolors='black', label='Centroides')
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        ax.set_title(f'K-Medias Clustering (K={best_k})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        img_buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_buf, format='png', dpi=120)
        plt.close(fig)
        img_buf.seek(0)
        img_b64 = base64.b64encode(img_buf.read()).decode('ascii')
        cluster_plot_uri = "data:image/png;base64," + img_b64
    
    # Agregar columna de clusters al DataFrame
    df_result = df.copy()
    cat_priority = cat_cols
    if reference_col:
        cat_priority = [reference_col] + [c for c in cat_cols if c != reference_col]
    
    if missing_mask.values.any():
        for idx in row_missing[row_missing].index:
            assigned_cluster = label_series.at[idx]
            if category_cluster_map:
                for col in cat_priority:
                    mapping = category_cluster_map.get(col)
                    if not mapping:
                        continue
                    value = df_original.at[idx, col]
                    if pd.isna(value):
                        continue
                    cluster_candidate = mapping.get(value)
                    if cluster_candidate is not None:
                        assigned_cluster = cluster_candidate
                        break
            missing_cols = missing_mask.loc[idx]
            for col in missing_cols[missing_cols].index:
                df_result.at[idx, col] = cluster_centers_df.at[assigned_cluster, col]
            label_series.at[idx] = assigned_cluster
    
    df_result['Cluster'] = label_series.values
    
    df_preview = df_result.head(50).copy()
    df_preview = df_preview.replace({np.nan: "?"})
    preview = df_preview.to_dict(orient="records")
    
    
    return jsonify({
        "message": f"K-Medias completado (K={best_k})",
        "best_k": best_k,
        "inertias": inertias,
        "preview": preview,
        "tree_image_datauri": cluster_plot_uri,
        "predictions": []
    })

def ejecutar_kmodos(df):
    """K-Modas - Clustering de datos categóricos"""
    df_work = df.replace("?", np.nan).copy()
    initial_cat_cols = df_work.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_cols = []
    for col in initial_cat_cols:
        series_numeric = pd.to_numeric(df_work[col], errors="coerce")
        if series_numeric.notna().any():
            continue
        cat_cols.append(col)
    
    if not cat_cols:
        # intentar encontrar la primera columna no numérica en el CSV original (sin reemplazar) para usarla como base
        raw_cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if raw_cat_cols:
            cat_cols = [raw_cat_cols[0]]
            df_work[raw_cat_cols[0]] = df[raw_cat_cols[0]].copy()
        else:
            return jsonify({"error": "No hay columnas categóricas para K-Modas"}), 400
    
    # Determinar columna base
    base_col = None
    missing_counts = {col: df_work[col].isna().sum() for col in cat_cols}
    cols_with_missing = [col for col, cnt in missing_counts.items() if cnt > 0]
    if cols_with_missing:
        base_col = max(cols_with_missing, key=lambda c: missing_counts[c])
    elif "Clase" in cat_cols and df_work["Clase"].notna().any():
        base_col = "Clase"
    else:
        base_col = df_work[cat_cols].notna().sum().idxmax()
    
    feat_cols = [col for col in cat_cols if col != base_col]
    if not feat_cols:
        feat_cols = []
    
    encoded_blocks = []
    cat_missing_mask = np.zeros((len(df_work), len(feat_cols)), dtype=bool) if feat_cols else np.zeros((len(df_work), 0), dtype=bool)
    if feat_cols:
        df_feat = df_work[feat_cols].copy()
        cat_missing_mask = df_feat.isna().to_numpy()
        df_feat = df_feat.fillna("MISSING").astype(str)
        encoded_blocks.append(df_feat.apply(lambda col: pd.Categorical(col).codes).to_numpy(dtype=np.uint16))
    
    aux_cols = []
    for col in df.columns:
        if col == base_col or col in feat_cols:
            continue
        series_numeric = pd.to_numeric(df_work[col], errors="coerce")
        if series_numeric.notna().sum() == 0:
            continue
        aux_cols.append(col)
        aux_series = series_numeric.apply(lambda v: f"NUM_{v}" if pd.notna(v) else "MISSING")
        encoded_blocks.append(pd.Categorical(aux_series).codes.astype(np.uint16).reshape(-1, 1))
    
    X_encoded = np.hstack(encoded_blocks) if encoded_blocks else None
    if X_encoded is None:
        return jsonify({"error": "No hay atributos válidos para K-Modas"}), 400
    
    mask_train = ~np.any(cat_missing_mask, axis=1) if feat_cols else np.ones(len(df_work), dtype=bool)
    if not np.any(mask_train):
        return jsonify({"error": "No hay filas completas para entrenar K-Modas"}), 400
    
    X_train = X_encoded[mask_train]
    k_min = max(1, min(K_MIN_ELBOW, len(X_train)))
    k_max = max(k_min, min(K_MAX_ELBOW, len(X_train)))
    
    costs = []
    for k in range(k_min, k_max + 1):
        modelo = KModes(
            n_clusters=k,
            init="Huang",
            n_init=N_INIT,
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
            verbose=0,
        ).fit(X_train)
        costs.append((k, modelo.cost_))
    
    best_k = elegir_k_por_codo(costs)
    
    modelo = KModes(
        n_clusters=best_k,
        init="Huang",
        n_init=N_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    labels_train = modelo.fit_predict(X_train)
    
    labels_all = np.full(len(df_work), -1, dtype=int)
    train_idx = np.where(mask_train)[0]
    labels_all[train_idx] = labels_train
    
    centroids = modelo.cluster_centroids_
    incomplete_idx = np.where(~mask_train)[0]
    for idx in incomplete_idx:
        row = X_encoded[idx]
        dists = np.sum(row != centroids, axis=1)
        labels_all[idx] = int(np.argmin(dists))
    
    base_series = df_work[base_col]
    known_base_mask = base_series.notna()
    cluster_to_base = {}
    base_to_cluster = {}
    for cluster_id in range(best_k):
        cluster_rows = (labels_all == cluster_id) & known_base_mask.to_numpy()
        valores = base_series[cluster_rows]
        if valores.empty:
            continue
        modo = valores.mode().iloc[0]
        cluster_to_base[cluster_id] = modo
        if modo not in base_to_cluster:
            base_to_cluster[modo] = cluster_id
    
    for idx in range(len(df_work)):
        if known_base_mask.iloc[idx]:
            val = base_series.iloc[idx]
            mapped = base_to_cluster.get(val)
            if mapped is not None:
                labels_all[idx] = mapped
        else:
            val = cluster_to_base.get(labels_all[idx])
            if val is not None:
                df_work.at[idx, base_col] = val
    
    df_result = df_work.copy()
    for col_idx, col in enumerate(feat_cols):
        col_series = df_work[col]
        modas = {}
        for cluster_id in range(best_k):
            valores = col_series[labels_all == cluster_id].dropna()
            if valores.empty:
                continue
            modas[cluster_id] = valores.mode().iloc[0]
        for row_idx, cluster_id in enumerate(labels_all):
            if cat_missing_mask[row_idx, col_idx] and cluster_id in modas:
                df_result.at[row_idx, col] = modas[cluster_id]
    
    df_result[base_col] = df_work[base_col]
    df_result["Cluster"] = labels_all
    df_preview = df_result.head(50).copy()
    df_preview = df_preview.replace({np.nan: "?"})
    preview = df_preview.to_dict(orient="records")
    
    return jsonify({
        "message": f"K-Modas completado (K={best_k})",
        "best_k": best_k,
        "kmodes_costs": costs,
        "preview": preview,
        "tree_image_datauri": None,
        "predictions": []
    })

def ejecutar_normalizacion(df, form_data):
    """Normalización de datos numéricos"""
    metodo = form_data.get('metodo_norm', 'min-max')
    
    # Seleccionar columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return jsonify({"error": "No hay columnas numéricas para normalizar"}), 400
    
    # Convertir a lista de listas
    tabla = df[numeric_cols].values.tolist()
    columnas_indices = list(range(len(numeric_cols)))
    
    # Aplicar normalización
    tabla_normalizada = normalizar_tabla(tabla, columnas_indices, metodo)
    
    # Crear DataFrame normalizado
    df_norm = df.copy()
    for i, col in enumerate(numeric_cols):
        df_norm[col] = [fila[i] for fila in tabla_normalizada]
    
    preview = df_norm.head(50).to_dict(orient="records")
    
    return jsonify({
        "message": f"Normalización {metodo} completada",
        "metodo": metodo,
        "columnas_normalizadas": numeric_cols,
        "preview": preview,
        "predictions": [],
        "tree_image_datauri": None
    })
    
def ejecutar_discretizacion(df, form_data):
    """Discretización usando ChiMerge"""
    class_column = form_data.get('columna_clase', df.columns[-1])
    max_intervals = int(form_data.get('max_intervals', 5))
    
    if class_column not in df.columns:
        return jsonify({"error": f"Columna de clase '{class_column}' no encontrada"}), 400
    
    y = df[class_column].values
    feature_columns = [col for col in df.columns if col != class_column]
    
    df_discretized = pd.DataFrame()
    chimerge = ChiMerge(max_intervals=max_intervals, significance_level=0.05)
    
    for column in feature_columns:
        X = df[column].values
        
        if not np.issubdtype(X.dtype, np.number):
            df_discretized[column] = df[column]
            continue
        
        try:
            chimerge.fit(X, y, feature_name=column)
            X_discretized = chimerge.transform(X, feature_name=column)
            labels = chimerge.get_interval_labels(column)
            X_labeled = [labels[val] for val in X_discretized]
            df_discretized[column] = X_labeled
        except Exception as e:
            df_discretized[column] = df[column]
    
    df_discretized[class_column] = df[class_column]
    
    preview = df_discretized.head(50).to_dict(orient="records")
    
    return jsonify({
        "message": "Discretización ChiMerge completada",
        "columna_clase": class_column,
        "max_intervals": max_intervals,
        "preview": preview,
        "predictions": [],
        "tree_image_datauri": None
    })


def elegir_k_por_codo(valores):
    """Método del codo para seleccionar K óptimo"""
    if len(valores) == 1:
        return valores[0][0]
    
    ks = np.array([k for k, _ in valores], dtype=float)
    vals = np.array([v for _, v in valores], dtype=float)
    
    x1, y1 = ks[0], vals[0]
    x2, y2 = ks[-1], vals[-1]
    
    dx, dy = x2 - x1, y2 - y1
    denom = np.hypot(dx, dy)
    
    if denom == 0:
        return int(ks[0])
    
    dist = np.abs(dy * ks - dx * vals + x2 * y1 - y2 * x1) / denom
    return int(ks[np.argmax(dist)])


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
