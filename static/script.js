// Referencias a elementos DOM
const fileInput = document.getElementById('file-input');
const algorithmSelect = document.getElementById('algorithm-select');
const processBtn = document.getElementById('process-btn');
const statusDiv = document.getElementById('status');
const resultsSection = document.getElementById('results');
const infoBox = document.getElementById('info-box');
const infoText = document.getElementById('info-text');
const visualization = document.getElementById('visualization');
const resultImg = document.getElementById('result-img');
const predictionsArea = document.getElementById('predictions-area');
const predictionsContainer = document.getElementById('predictions-container');
const previewArea = document.getElementById('preview-area');
const previewContainer = document.getElementById('preview-container');

// Paneles de opciones
const optionsNormalizacion = document.getElementById('options-normalizacion');
const optionsDiscretizacion = document.getElementById('options-discretizacion');

// Efecto suave al cargar elementos
function fadeIn(element) {
    element.classList.add("fade-in");
    element.style.display = "block";
}

function fadeOut(element) {
    element.classList.remove("fade-in");
    element.style.display = "none";
}

// Habilitar botón cuando se seleccione archivo y algoritmo
fileInput.addEventListener('change', checkInputs);
algorithmSelect.addEventListener('change', function() {
    checkInputs();
    showOptionsPanel();
});

function checkInputs() {
    if (fileInput.files.length > 0 && algorithmSelect.value !== '') {
        processBtn.disabled = false;
        processBtn.classList.add("btn-enabled");
    } else {
        processBtn.disabled = true;
        processBtn.classList.remove("btn-enabled");
    }
}

function showOptionsPanel() {
    optionsNormalizacion.classList.remove('active');
    optionsDiscretizacion.classList.remove('active');

    const algorithm = algorithmSelect.value;
    if (algorithm === 'normalizacion') {
        optionsNormalizacion.classList.add('active');
        fadeIn(optionsNormalizacion);
    } else if (algorithm === 'discretizacion') {
        optionsDiscretizacion.classList.add('active');
        fadeIn(optionsDiscretizacion);
    }
}

// Procesar datos
processBtn.addEventListener('click', async function() {
    const file = fileInput.files[0];
    const algorithm = algorithmSelect.value;

    if (!file || !algorithm) {
        alert('Por favor selecciona un archivo y un algoritmo');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('select', algorithm);

    if (algorithm === 'normalizacion') {
        formData.append('metodo_norm', document.getElementById('metodo-norm').value);
    } else if (algorithm === 'discretizacion') {
        const columnaClase = document.getElementById('columna-clase').value;
        if (columnaClase) {
            formData.append('columna_clase', columnaClase);
        }
        formData.append('max_intervals', document.getElementById('max-intervals').value);
    }

    showStatus('loading', 'Procesando datos... Por favor espera');
    processBtn.disabled = true;
    hideResults();

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Error en el servidor');
        }

        showStatus('success', data.message || '¡Proceso completado exitosamente!');
        displayResults(data, algorithm);

    } catch (error) {
        showStatus('error', `Error: ${error.message}`);
        console.error(error);
    } finally {
        processBtn.disabled = false;
    }
});

function showStatus(type, message) {
    statusDiv.className = `status show ${type}`;
    statusDiv.textContent = message;

    // Efecto vibración sutil en errores
    if (type === "error") {
        statusDiv.classList.add("shake");
        setTimeout(() => statusDiv.classList.remove("shake"), 500);
    }
}

function hideResults() {
    fadeOut(infoBox);
    fadeOut(visualization);
    fadeOut(predictionsArea);
    fadeOut(previewArea);
}

function displayResults(data, algorithm) {
    resultsSection.classList.add('show');

    if (data.best_k !== undefined) {
        infoBox.style.display = 'block';
        infoText.textContent = `Mejor K seleccionado: ${data.best_k}`;
        fadeIn(infoBox);
    } else if (data.metodo) {
        infoBox.style.display = 'block';
        infoText.textContent = `Método aplicado: ${data.metodo}`;
        fadeIn(infoBox);
    } else if (data.columna_clase) {
        infoBox.style.display = 'block';
        infoText.textContent = `Columna de clase: ${data.columna_clase}, Intervalos: ${data.max_intervals}`;
        fadeIn(infoBox);
    } else if (data.target_column) {
        infoBox.style.display = 'block';
        infoText.textContent = `Columna objetivo: ${data.target_column}`;
        fadeIn(infoBox);
    }

    if (data.tree_image_datauri) {
        resultImg.src = data.tree_image_datauri;
        fadeIn(visualization);
    }

    if (data.predictions && data.predictions.length > 0) {
        buildPredictionsTable(data.predictions);
        fadeIn(predictionsArea);
    }

    if (data.preview && data.preview.length > 0) {
        buildPreviewTable(data.preview);
        fadeIn(previewArea);
    }
}

function buildPredictionsTable(predictions) {
    if (!predictions || predictions.length === 0) {
        predictionsContainer.innerHTML = '<p style="text-align:center; color:#666;">No hay predicciones</p>';
        return;
    }

    const firstRow = predictions[0].row;
    const columns = Object.keys(firstRow);

    let html = '<table><thead><tr>';
    html += '<th>Índice</th>';
    columns.forEach(col => {
        html += `<th>${escapeHtml(col)}</th>`;
    });
    html += '<th>Predicción</th>';
    html += '</tr></thead><tbody>';

    predictions.forEach(pred => {
        html += `<tr class="fade-row"><td><strong>${pred.index}</strong></td>`;
        columns.forEach(col => {
            const value = pred.row[col] !== undefined ? pred.row[col] : '';
            html += `<td>${escapeHtml(String(value))}</td>`;
        });
        html += `<td style="background:#d4edda; font-weight:bold;">${escapeHtml(String(pred.predicted))}</td>`;
        html += '</tr>';
    });

    html += '</tbody></table>';
    predictionsContainer.innerHTML = html;
}

function buildPreviewTable(rows) {
    if (!rows || rows.length === 0) {
        previewContainer.innerHTML = '<p style="text-align:center; color:#666;">No hay vista previa</p>';
        return;
    }

    const columns = Object.keys(rows[0]);

    let html = '<table><thead><tr>';
    columns.forEach(col => {
        html += `<th>${escapeHtml(col)}</th>`;
    });
    html += '</tr></thead><tbody>';

    rows.forEach(row => {
        html += '<tr class="fade-row">';
        columns.forEach(col => {
            const value = row[col] !== undefined ? row[col] : '';
            html += `<td>${escapeHtml(String(value))}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table>';
    previewContainer.innerHTML = html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
