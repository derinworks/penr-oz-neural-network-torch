function getQueryParam(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

function updateQueryParam(name, value) {
    const url = new URL(window.location);
    url.searchParams.set(name, value);
    window.history.replaceState({}, '', url); // no page reload
}

function renderHistogramChart({ container, histograms, labelFn, title, formatX = null, formatY = null }) {
    const canvas = document.createElement('canvas');
    canvas.width = 800;
    canvas.height = 400;
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');

    const allX = histograms.flatMap(hist => hist?.x ?? []);
    const minX = Math.min(...allX);
    const maxX = Math.max(...allX);

    const datasets = histograms.map((hist, index) => {
        if (!hist) return null;

        const dataPoints = hist.x.map((xVal, i) => ({
            x: xVal,
            y: hist.y[i]
        }));

        return {
            label: labelFn(index),
            data: dataPoints,
            borderWidth: 2,
            fill: true,
            tension: 0.3
        };
    }).filter(d => d != null);

    new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            stacked: true,
            parsing: false,
            plugins: {
                title: title
                    ? {
                        display: true,
                        text: title,
                        font: { size: 18, weight: 'bold' },
                        padding: { top: 10, bottom: 20 }
                    }
                    : false,
                tooltip: {
                    callbacks: {
                        label: context => {
                            const label = context.dataset.label || '';
                            const x = context.raw.x;
                            const y = context.raw.y;
                            return `${label}: (${formatX ? formatX(x) : x}, ${formatY ? formatY(y) : y})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    min: minX,
                    max: maxX,
                    ticks: {
                        count: 10,
                        callback: val => formatX ? formatX(val) : parseFloat(val).toFixed(2)
                    },
                    title: {
                        display: true,
                        text: 'X Axis'
                    }
                },
                y: {
                    beginAtZero: true,
                    stacked: true,
                    ticks: {
                        callback: val => formatY ? formatY(val) : val
                    },
                    title: {
                        display: true,
                        text: 'Y Axis'
                    }
                }
            }
        }
    });
}

async function loadStats() {
    // Take inputs and store
    const modelId = document.getElementById('model-id').value.trim();
    if (!modelId) {
        alert("Please enter a model ID.");
        return;
    }
    updateQueryParam('model_id', modelId);

    const algoFilter = document.getElementById('algo-filter').value.trim().toLowerCase();
    updateQueryParam('algo', algoFilter);

    // Fetch stats
    const response = await fetch(`/stats?model_id=${encodeURIComponent(modelId)}`);
    if (!response.ok) {
        alert("Failed to fetch stats. Check model ID.");
        return;
    }

    const data = await response.json();
    if (!data) {
        alert(`No stats recorded yet for model ${modelId}`);
        return;
    }
    
    // Layer to index lookup
    const layerToIndex = new Map(data.layers.map((layer, i) => [layer, i]));

    // Apply optional algo filter
    const algoFilters = algoFilter.split(',').map(f => f.trim()).filter(f => f.length > 0);
    const layers = data.layers.filter(layer => {
        const algo = layer.algo?.toLowerCase() || '';
        return algoFilters.length === 0 || algoFilters.some(f => algo.includes(f));
    });

    // Clear previous stats
    const container = document.getElementById('stats-container');
    container.innerHTML = '';

    // Render activation stats
    const headerActivations = document.createElement('h2');
    headerActivations.textContent = `Activations for model ${modelId}`;
    container.appendChild(headerActivations);
    layers.forEach(layer => {
        const activation = layer.activation;
        const mean = activation.mean.toFixed(2);
        const std = activation.std.toFixed(2);
        const saturated = (activation.saturated * 100).toFixed(1) + '%';
        
        const header = document.createElement('h3');
        header.textContent = `Layer ${layerToIndex.get(layer)} (${layer.algo}): mean ${mean} std ${std} saturated: ${saturated}`;
        container.appendChild(header);
    });
    renderHistogramChart({
        container,
        histograms: layers.map(layer => layer.activation.histogram),
        labelFn: (i) => `Layer ${layerToIndex.get(layers[i])} (${layers[i].algo})`,
        title: 'Activation Distribution',
    });

    // Render gradient stats
    const headerGradients = document.createElement('h2');
    headerGradients.textContent = `Gradients for model ${modelId}`;
    container.appendChild(headerGradients);
    layers.forEach(layer => {
        const gradient = layer.gradient;
        if (gradient) {
            const mean = gradient.mean.toExponential(6);
            const std = gradient.std.toExponential(6);
            
            const header = document.createElement('h3');
            header.textContent = `Layer ${layerToIndex.get(layer)} (${layer.algo}): mean ${mean} std ${std}`;
            container.appendChild(header);
        }
    });
    renderHistogramChart({
        container,
        histograms: layers.map(layer => layer.gradient?.histogram),
        labelFn: i => `Layer ${layerToIndex.get(layers[i])} (${layers[i].algo})`,
        title: 'Gradient Distribution',
        formatX: x => x.toFixed(6),
    });


    // Render weight stats
    const headerWeights = document.createElement('h2');
    headerWeights.textContent = `Weights for model ${modelId}`;
    container.appendChild(headerWeights);
    data.weights.forEach((w, index) => {
        const mean = w.gradient.mean.toExponential(6);
        const std = w.gradient.std.toExponential(6);
        const ratio = (std / w.data.std).toExponential(6);
        
        const header = document.createElement('h3');
        header.textContent = `Weights ${w.shape}: mean ${mean} std ${std}  grad:data ratio ${ratio}`;
        container.appendChild(header);
    });
    renderHistogramChart({
        container,
        histograms: data.weights.map(w => w.gradient.histogram),
        labelFn: i => `Weights ${data.weights[i].shape}`,
        title: 'Weight Gradient Distribution',
        formatX: x => x.toFixed(3),
    });
}

// On page load: auto-fill inputs
window.onload = () => {
    const modelId = getQueryParam('model_id');
    if (modelId) {
        document.getElementById('model-id').value = modelId;
    }
    const algoFilter = getQueryParam('algo');
    if (algoFilter) {
        document.getElementById('algo-filter').value = algoFilter
    }
};
