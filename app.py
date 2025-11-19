"""
Sistema Integrado de Minería de Datos
Incluye: K-Medias, K-Modas, Normalización, Discretización, Árbol de Decisión
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
    df = df.replace("?", np.nan)

    # convertir a numerico lo que se pueda
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return jsonify({"error": "No hay columnas numericas para K-Medias"}), 400

    # forzar float para no perder decimales al imputar
    df[numeric_cols] = df[numeric_cols].astype(float)

    # matriz numerica y mascara de faltantes
    X = df[numeric_cols].to_numpy(dtype=float, copy=True)
    missing_mask = np.isnan(X)
    mask_train = ~np.any(missing_mask, axis=1)
    if not np.any(mask_train):
        return jsonify({"error": "No hay filas completamente numericas para entrenar K-Medias"}), 400

    X_train = X[mask_train]
    n_train = X_train.shape[0]

    # detectar columna de clase (opcional)
    non_num = df.select_dtypes(exclude=[np.number]).columns.tolist()
    class_col = None
    if non_num:
        completas = [c for c in non_num if not df[c].isna().any()]
        if completas:
            class_col = completas[0]
        elif "Clase" in non_num:
            class_col = "Clase"
        else:
            class_col = non_num[-1]

    # rango de K acotado por filas de entrenamiento (permitir K=1 como en la versión base)
    k_min = 1
    k_max = min(K_MAX_ELBOW, n_train)
    if k_min > k_max:
        return jsonify({"error": f"Rango K invalido: [{k_min}, {k_max}]"}), 400

    inertias = []
    for k in range(k_min, k_max + 1):
        modelo = KMeans(
            n_clusters=k,
            init="random",
            random_state=RANDOM_STATE,
            n_init=N_INIT,
            max_iter=MAX_ITER,
        ).fit(X_train)
        inertias.append((k, float(modelo.inertia_)))

    best_k = elegir_k_por_codo(inertias)

    modelo_final = KMeans(
        n_clusters=best_k,
        init="random",
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        max_iter=MAX_ITER,
    ).fit(X_train)
    labels_train = modelo_final.labels_
    centroids = modelo_final.cluster_centers_

    # mapeo cluster -> clase (mayoria) y clase -> cluster
    cluster_to_class, class_to_cluster = {}, {}
    if class_col is not None and class_col in df.columns:
        df_train_class = df.loc[mask_train, class_col]
        known_mask = df_train_class.notna()
        if known_mask.any():
            for cid in np.unique(labels_train):
                vals = df_train_class[(labels_train == cid) & known_mask]
                if len(vals) == 0:
                    continue
                cls = vals.mode().iloc[0]
                cluster_to_class[int(cid)] = cls
                if cls not in class_to_cluster:
                    class_to_cluster[cls] = int(cid)

    # asignar clusters a todas las filas
    labels_all = np.full(len(df), -1, dtype=int)
    labels_all[mask_train] = labels_train

    for idx in np.where(~mask_train)[0]:
        # asignacion por clase si hay mapeo
        if class_col is not None and class_col in df.columns:
            cls_val = df.at[idx, class_col]
            if pd.notna(cls_val) and cls_val in class_to_cluster:
                labels_all[idx] = class_to_cluster[cls_val]
                continue

        mask_feat = ~missing_mask[idx]
        if not np.any(mask_feat):
            labels_all[idx] = 0  # fila sin info numerica
            continue
        x_row = X[idx, mask_feat].reshape(1, -1)
        cent_sub = centroids[:, mask_feat]
        dists = np.linalg.norm(cent_sub - x_row, axis=1)
        labels_all[idx] = int(np.argmin(dists))

    # imputar numericos faltantes con centroides (redondeo a 2 decimales)
    for i in range(len(df)):
        c = labels_all[i] if labels_all[i] >= 0 else 0
        for j, col in enumerate(numeric_cols):
            if missing_mask[i, j]:
                df.at[i, col] = float(np.round(centroids[c, j], 2))

    # aplicar mapeo de clase (si existe)
    if class_col is not None and cluster_to_class:
        df[class_col] = df[class_col].astype("object")
        for i, c in enumerate(labels_all):
            if c in cluster_to_class:
                df.at[i, class_col] = cluster_to_class[c]

    df["Cluster"] = labels_all + 1  # 1-based para lectura

    # visualizacion 2D con dos primeras columnas numericas
    cluster_plot_uri = None
    if len(numeric_cols) >= 2:
        X_plot = df[numeric_cols].to_numpy(dtype=float, copy=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels_all, cmap="viridis", s=50, alpha=0.6)
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="X",
            s=200,
            c="red",
            edgecolors="black",
            label="Centroides",
        )
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        ax.set_title(f"K-Medias Clustering (K={best_k})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Cluster")

        img_buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_buf, format="png", dpi=120)
        plt.close(fig)
        img_buf.seek(0)
        img_b64 = base64.b64encode(img_buf.read()).decode("ascii")
        cluster_plot_uri = "data:image/png;base64," + img_b64

    df_preview = df.head(50).copy()
    df_preview = df_preview.replace({np.nan: "?"})
    preview = df_preview.to_dict(orient="records")

    return jsonify({
        "message": f"K-Medias completado (K={best_k})",
        "best_k": int(best_k),
        "inertias": inertias,
        "preview": preview,
        "tree_image_datauri": cluster_plot_uri,
        "predictions": []
    })

def ejecutar_kmodos(df):
    """K-Modas - alineado con la l?gica de KModas base con detecci?n num?rica robusta."""
    df_work = df.replace("?", np.nan).copy()

    # limpiar strings y detectar columnas num?ricas a partir de la proporci?n de valores num?ricos
    num_threshold = 0.8
    for col in df_work.columns:
        if df_work[col].dtype == object:
            serie = df_work[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
            serie = serie.replace({"": np.nan})
            coerced = pd.to_numeric(serie, errors="coerce")
            non_null = serie.notna()
            if non_null.any() and (coerced.notna().sum() / non_null.sum()) >= num_threshold:
                df_work[col] = coerced
            else:
                df_work[col] = serie

    # columnas categoricas (no num?ricas)
    cat_cols = df_work.select_dtypes(exclude=[np.number]).columns.tolist()
    if not cat_cols:
        return jsonify({"error": "No hay columnas categ?ricas para K-Modas"}), 400

    # columna base
    completas = [c for c in cat_cols if not df_work[c].isna().any()]
    if completas:
        base_col = completas[0]
    elif "Clase" in cat_cols and df_work["Clase"].notna().any():
        base_col = "Clase"
    else:
        base_col = df_work[cat_cols].notna().sum().idxmax()

    feat_cols = [c for c in cat_cols if c != base_col]
    if not feat_cols:
        return jsonify({"error": "No hay columnas categ?ricas distintas de la base para K-Modas"}), 400

    # codificaci?n categ?rica para KModes
    df_feat = df_work[feat_cols].fillna("MISSING").astype(str)
    cat_missing_mask = df_work[feat_cols].isna().to_numpy()
    X_encoded = df_feat.apply(lambda col: pd.Categorical(col).codes).to_numpy(dtype=np.uint16)

    mask_train = ~np.any(cat_missing_mask, axis=1)
    if not np.any(mask_train):
        return jsonify({"error": "No hay filas completas para entrenar K-Modas"}), 400

    X_train = X_encoded[mask_train]

    k_min = 1
    k_max = min(K_MAX_ELBOW, len(X_train))
    if k_min > k_max:
        return jsonify({"error": f"Rango K inv?lido: [{k_min}, {k_max}]"}), 400

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
    labels_all[mask_train] = labels_train

    # asignar clusters a filas incompletas por distancia simb?lica
    centroids = modelo.cluster_centroids_
    for idx in np.where(~mask_train)[0]:
        row = X_encoded[idx]
        dists = np.sum(row != centroids, axis=1)
        labels_all[idx] = int(np.argmin(dists))

    # mapeo cluster <-> valor base usando s?lo filas completas de entrenamiento
    base_series_train = df_work.loc[mask_train, base_col]
    cluster_to_base = {}
    base_to_cluster = {}
    for cid in range(best_k):
        valores = base_series_train[labels_train == cid].dropna()
        if valores.empty:
            continue
        modo = valores.mode().iloc[0]
        cluster_to_base[cid] = modo
        if modo not in base_to_cluster:
            base_to_cluster[modo] = cid

    # usar valor base conocido para re-asignar (solo filas incompletas)
    for idx in np.where(~mask_train)[0]:
        val = df_work.at[idx, base_col]
        if pd.notna(val) and val in base_to_cluster:
            labels_all[idx] = base_to_cluster[val]

    # imputar otras categ?ricas por moda del cluster
    df_result = df_work.copy()
    for col_idx, col in enumerate(feat_cols):
        col_series = df_work[col]
        modas = {}
        for cid in range(best_k):
            vals = col_series[labels_all == cid].dropna()
            if vals.empty:
                continue
            modas[cid] = vals.mode().iloc[0]
        for row_idx, cid in enumerate(labels_all):
            if cat_missing_mask[row_idx, col_idx] and cid in modas:
                df_result.at[row_idx, col] = modas[cid]

    # imputar base faltante con modo del cluster resultante
    for idx in range(len(df_result)):
        if pd.isna(df_result.at[idx, base_col]):
            mapped = cluster_to_base.get(labels_all[idx])
            if mapped is not None:
                df_result.at[idx, base_col] = mapped

    df_result["Cluster"] = labels_all + 1  # 1-based para consistencia

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
