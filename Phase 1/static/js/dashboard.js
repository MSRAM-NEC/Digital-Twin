/* ═══════════════════════════════════════════════════════════════
   Drug–Heart Digital Twin — Dashboard v2.0
   Complete rewrite with clean architecture
   ═══════════════════════════════════════════════════════════════ */

/* ── Chart.js Defaults ── */
Chart.defaults.color = '#5a6f8a';
Chart.defaults.borderColor = 'rgba(30, 60, 100, 0.08)';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 8;
Chart.defaults.plugins.legend.labels.padding = 14;
Chart.defaults.animation.duration = 700;

const C = {}; // Chart instances
const PAL = {
    blue: '#2563eb', blueA: 'rgba(37,99,235,.10)',
    purple: '#7c3aed', purpleA: 'rgba(124,58,237,.10)',
    cyan: '#0891b2', cyanA: 'rgba(8,145,178,.10)',
    pink: '#db2777', pinkA: 'rgba(219,39,119,.10)',
    green: '#059669', greenA: 'rgba(5,150,105,.10)',
    amber: '#d97706', amberA: 'rgba(217,119,6,.10)',
    red: '#dc2626', redA: 'rgba(220,38,38,.10)',
};

const tooltipStyle = {
    backgroundColor: 'rgba(255,255,255,.97)',
    titleColor: '#1a2639',
    bodyColor: '#2e3f5c',
    borderColor: 'rgba(30,60,100,.12)',
    borderWidth: 1,
    padding: 12,
    cornerRadius: 8,
    boxShadow: '0 4px 16px rgba(30,60,100,.10)',
};

/* ══════════════════════════════════
   INIT
   ══════════════════════════════════ */
document.addEventListener('DOMContentLoaded', init);

function init() {
    loadDrugs();
}

/* ── Tab Switching ── */
function switchTab(name) {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
    document.getElementById('sec-' + name).classList.add('active');

    // Lazy-load heart iframe
    if (name === 'heart') {
        const iframe = document.getElementById('heartIframe');
        if (!iframe.src || iframe.src === '' || iframe.src === window.location.href) {
            iframe.src = '/heart?mode=drug';
        }
    }
}

/* ══════════════════════════════════
   LOAD DRUG LIST
   ══════════════════════════════════ */
async function loadDrugs() {
    try {
        const res = await fetch('/api/drugs');
        const list = await res.json();
        const sel = document.getElementById('drugSelect');
        list.forEach(d => {
            const o = document.createElement('option');
            o.value = d.name;
            o.textContent = `${d.name} — ${d.category}`;
            o.dataset.dose = d.typical_dose;
            o.dataset.maxDose = d.max_dose;
            sel.appendChild(o);
        });
        sel.addEventListener('change', () => {
            const opt = sel.options[sel.selectedIndex];
            if (opt.dataset.dose) {
                const doseInput = document.getElementById('doseInput');
                doseInput.value = opt.dataset.dose;
                if (opt.dataset.maxDose) doseInput.max = opt.dataset.maxDose;
            }
        });
        if (sel.options.length) {
            const doseInput = document.getElementById('doseInput');
            doseInput.value = sel.options[0].dataset.dose || 50;
            if (sel.options[0].dataset.maxDose) doseInput.max = sel.options[0].dataset.maxDose;
        }
    } catch (e) {
        console.error('Drug list error:', e);
    }
}

/* ══════════════════════════════════
   SIMULATION
   ══════════════════════════════════ */
async function runSimulation() {
    const btn = document.getElementById('btnSimulate');
    const overlay = document.getElementById('loadingOverlay');

    const doseInput = document.getElementById('doseInput');

    btn.disabled = true;
    btn.textContent = '⏳ Simulating…';
    overlay.classList.add('on');
    animPipeline();

    const body = {
        drug_name: document.getElementById('drugSelect').value,
        dose_mg: +doseInput.value,
        age: +document.getElementById('ageInput').value,
        weight_kg: +document.getElementById('weightInput').value,
        sex: document.getElementById('sexSelect').value,
        t_max_hours: +document.getElementById('timeInput').value,
        heart_disease: document.getElementById('heartDiseaseCheck').checked,
        hypertension: document.getElementById('hypertensionCheck').checked,
        diabetes: document.getElementById('diabetesCheck').checked,
        renal_impairment: document.getElementById('renalCheck').checked,
    };

    try {
        const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const json = await res.json();
        if (json.status === 'success') {
            const maxDose = parseFloat(doseInput.max);
            const inputDose = parseFloat(doseInput.value);

            // If dose substantially exceeds the clinical limit, warn in the dashboard instead of blocking
            if (maxDose && inputDose > maxDose) {
                const excessRatio = (inputDose - maxDose) / maxDose;
                // Overdose penalty: gradually penalizes score. E.g. at 10% overdose (ratio 0.1) penalty is 1.5 
                const penalty = Math.min(10, excessRatio * 15);

                json.data.layer_5_cardiovascular.warnings.unshift(
                    `⚠️ SUPRAMAXIMAL DOSE: Administered dose (${inputDose}mg) clinically exceeds maximum safety limits (${maxDose}mg). Severe toxicity risk.`
                );

                // Apply gradual penalty dynamically to output UI
                let safeScore = json.data.layer_6_ml_predictions.drug_safety.safety_score;
                safeScore = Math.max(0, safeScore - penalty);

                json.data.layer_6_ml_predictions.drug_safety.safety_score = parseFloat(safeScore.toFixed(1));
                json.data.layer_6_ml_predictions.drug_safety.demerits_breakdown['Overdose Penalty'] = parseFloat(penalty.toFixed(1));

                // Recalculate color and string classifiers gracefully
                if (safeScore >= 8) {
                    json.data.layer_6_ml_predictions.drug_safety.classification = "Safe";
                    json.data.layer_6_ml_predictions.drug_safety.color = "green";
                } else if (safeScore >= 6) {
                    json.data.layer_6_ml_predictions.drug_safety.classification = "Caution";
                    json.data.layer_6_ml_predictions.drug_safety.color = "yellow";
                } else if (safeScore >= 4) {
                    json.data.layer_6_ml_predictions.drug_safety.classification = "Warning";
                    json.data.layer_6_ml_predictions.drug_safety.color = "orange";
                } else if (safeScore > 0) {
                    json.data.layer_6_ml_predictions.drug_safety.classification = "Dangerous";
                    json.data.layer_6_ml_predictions.drug_safety.color = "red";
                } else {
                    json.data.layer_6_ml_predictions.drug_safety.classification = "Lethal Toxicity";
                    json.data.layer_6_ml_predictions.drug_safety.color = "red";
                }
            }

            render(json.data);
            const pill = document.getElementById('statusPill');
            pill.innerHTML = '<div class="status-dot"></div>Simulation Done';
        } else {
            alert('Error: ' + (json.message || 'Unknown'));
        }
    } catch (e) {
        alert('Connection error: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '🧬 Run Simulation';
        overlay.classList.remove('on');
        donePipeline();
    }
}

/* ── Pipeline Animation ── */
function animPipeline() {
    for (let i = 1; i <= 7; i++) {
        document.getElementById('p' + i).classList.remove('active', 'done');
        const a = document.getElementById('a' + i);
        if (a) a.classList.remove('active', 'done');
    }
    const labels = [
        'Processing Drug Representation…',
        'Predicting Target Interactions…',
        'Computing Pharmacokinetics…',
        'Modeling Pharmacodynamics…',
        'Simulating Cardiovascular System…',
        'Running ML Predictions…',
        'Generating Digital Twin Output…',
    ];
    labels.forEach((txt, i) => {
        setTimeout(() => {
            if (i > 0) {
                document.getElementById('p' + i).classList.replace('active', 'done');
                const pa = document.getElementById('a' + i);
                if (pa) pa.classList.replace('active', 'done');
            }
            document.getElementById('p' + (i + 1)).classList.add('active');
            const na = document.getElementById('a' + (i + 1));
            if (na) na.classList.add('active');
            document.getElementById('loadingSub').textContent = txt;
        }, i * 250);
    });
}

function donePipeline() {
    for (let i = 1; i <= 7; i++) {
        const n = document.getElementById('p' + i);
        n.classList.remove('active'); n.classList.add('done');
        const a = document.getElementById('a' + i);
        if (a) { a.classList.remove('active'); a.classList.add('done'); }
    }
}

/* ══════════════════════════════════
   RENDER ALL
   ══════════════════════════════════ */
function render(d) {
    const box = document.getElementById('resultsBox');
    box.classList.add('show');

    renderStats(d);
    renderDrug(d.layer_1_drug_representation);
    renderTargets(d.layer_2_drug_target);
    renderPK(d.layer_3_pharmacokinetics);
    renderADME(d.layer_3_pharmacokinetics.adme);
    renderPD(d.layer_4_pharmacodynamics);
    renderECG(d.layer_5_cardiovascular.ecg);
    renderPressure(d.layer_5_cardiovascular.windkessel);
    renderPerfusion(d.layer_5_cardiovascular.cardiac_output);
    renderWarnings(d.layer_5_cardiovascular.warnings);
    renderPreds(d.layer_6_ml_predictions);
    renderSafety(d.layer_6_ml_predictions.drug_safety);
    renderTimes(d.simulation_info);
    syncHeart(d);

    box.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── Summary Strip ── */
function renderStats(d) {
    const cv = d.layer_5_cardiovascular.cardiac_parameters;
    const sf = d.layer_6_ml_predictions.drug_safety;
    const ar = d.layer_6_ml_predictions.arrhythmia_risk;
    const pk = d.layer_3_pharmacokinetics;
    const co = d.layer_5_cardiovascular.cardiac_output;

    const items = [
        { i: '💓', v: cv.heart_rate_bpm, l: 'Heart Rate', s: 'bpm', c: 'c-pink' },
        { i: '🩸', v: `${cv.systolic_bp_mmHg}/${cv.diastolic_bp_mmHg}`, l: 'Blood Pressure', s: 'mmHg', c: 'c-red' },
        { i: '📊', v: cv.qt_interval_ms.toFixed(0), l: 'QT Interval', s: 'ms', c: 'c-purple' },
        { i: '💪', v: cv.contractility_pct.toFixed(0) + '%', l: 'Contractility', s: 'of normal', c: 'c-blue' },
        { i: '🫁', v: co.cardiac_output_Lmin, l: 'Cardiac Output', s: 'L/min', c: 'c-cyan' },
        { i: '📈', v: pk.cmax.toFixed(2), l: 'Cmax', s: 'μg/mL', c: 'c-amber' },
        { i: '⚡', v: ar.risk_level, l: 'Arrhythmia', s: (ar.probability * 100).toFixed(1) + '%', c: ar.risk_level === 'HIGH' ? 'c-red' : ar.risk_level === 'MODERATE' ? 'c-amber' : 'c-green' },
        { i: '🛡️', v: sf.safety_score.toFixed(1) + '/10', l: 'Safety Score', s: sf.classification, c: sf.color === 'green' ? 'c-green' : sf.color === 'red' ? 'c-red' : 'c-amber' },
    ];

    document.getElementById('summaryCards').innerHTML = items.map(it => `
        <div class="stat-card ${it.c}">
            <div class="stat-icon">${it.i}</div>
            <div class="stat-val">${it.v}</div>
            <div class="stat-label">${it.l}</div>
            <div class="stat-sub">${it.s}</div>
        </div>
    `).join('');
}

/* ── L1: Drug ── */
function renderDrug(l) {
    document.getElementById('smilesDisplay').textContent = l.smiles;
    const p = l.properties;
    const data = [
        ['Mol. Weight', p.mw], ['LogP', p.logp], ['TPSA', p.tpsa],
        ['H-Donors', p.hbd], ['H-Acceptors', p.hba],
        ['Category', p.category.replace(/_/g, ' ')],
        ['Half-life', p.half_life + 'h'],
        ['FP Density', (l.fingerprint_density * 100).toFixed(1) + '%'],
    ];
    document.getElementById('propsGrid').innerHTML = data.map(([lbl, val]) => `
        <div class="prop"><div class="prop-val">${val}</div><div class="prop-lbl">${lbl}</div></div>
    `).join('');
}

/* ── L2: Targets ── */
function renderTargets(l) {
    const rows = Object.entries(l.interactions).map(([name, d]) => {
        const occ = d.occupancy_pct;
        const cls = occ > 50 ? 'high' : occ > 20 ? 'mid' : 'low';
        return `<tr>
            <td><strong>${name}</strong><br><span style="font-size:10px;color:var(--t-4)">${d.protein_name}</span></td>
            <td class="v">${occ.toFixed(1)}%</td>
            <td><div class="bar-track"><div class="bar-fill ${cls}" style="width:${Math.max(2, occ)}%"></div></div></td>
            <td style="font-size:11px">${d.cardiac_effect}</td>
            <td style="font-size:10px;color:var(--t-4)">${d.prediction_source === 'experimental_database' ? '🔬' : '🤖'}</td>
        </tr>`;
    });
    document.getElementById('targetTableContainer').innerHTML = `
        <table class="tbl">
            <thead><tr><th>Target</th><th>Occupancy</th><th>Saturation</th><th>Effect</th><th></th></tr></thead>
            <tbody>${rows.join('')}</tbody>
        </table>
        <div style="margin-top:10px;font-size:11px;color:var(--t-3)">
            Primary: <strong style="color:var(--t-1)">${l.primary_targets.join(', ') || 'None'}</strong>
            &nbsp;·&nbsp; Selectivity: <span class="mono" style="color:var(--cyan)">${l.selectivity_index}</span>
        </div>
    `;
}

/* ── L3: PK Chart ── */
function renderPK(pk) {
    kill('chartPK');
    const t = pk.concentration_profile.time;
    const c1 = pk.concentration_profile.concentration;
    const c2 = pk.two_compartment.central;
    const c3 = pk.two_compartment.peripheral;
    const step = Math.max(1, Math.floor(t.length / 200));
    const ds = i => i.filter((_, idx) => idx % step === 0);

    C.chartPK = new Chart(document.getElementById('chartPK'), {
        type: 'line',
        data: {
            labels: ds(t).map(v => v.toFixed(1)),
            datasets: [
                { label: 'One-Compartment', data: ds(c1), borderColor: PAL.blue, backgroundColor: PAL.blueA, fill: true, tension: .4, pointRadius: 0, borderWidth: 2 },
                { label: 'Central', data: ds(c2), borderColor: PAL.cyan, backgroundColor: 'transparent', borderDash: [5, 3], tension: .4, pointRadius: 0, borderWidth: 1.5 },
                { label: 'Peripheral', data: ds(c3), borderColor: PAL.purple, backgroundColor: 'transparent', borderDash: [2, 4], tension: .4, pointRadius: 0, borderWidth: 1.5 },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: { tooltip: { ...tooltipStyle, callbacks: { title: i => `${i[0].label} hrs`, label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(4)} μg/mL` } } },
            scales: {
                x: { title: { display: true, text: 'Time (hours)', color: '#5a6f8a' }, ticks: { maxTicksLimit: 12 } },
                y: { title: { display: true, text: 'Concentration (μg/mL)', color: '#5a6f8a' }, beginAtZero: true },
            },
        },
    });
}

/* ── ADME ── */
function renderADME(adme) {
    const a = adme.absorption, d = adme.distribution, m = adme.metabolism, e = adme.excretion;
    document.getElementById('admeFlow').innerHTML = `
        <div class="adme-box"><div class="adme-ltr">A</div><div class="adme-name">Absorption</div>
            <div class="adme-val">F = ${(a.bioavailability * 100).toFixed(0)}%</div>
            <div class="adme-det">ka = ${a.ka} h⁻¹ · Tmax ${a.tmax_hours.toFixed(1)}h</div></div>
        <div class="adme-box"><div class="adme-ltr">D</div><div class="adme-name">Distribution</div>
            <div class="adme-val">Vd = ${d.vd_liters} L</div>
            <div class="adme-det">Protein bind: ${(d.protein_binding * 100).toFixed(0)}%</div></div>
        <div class="adme-box"><div class="adme-ltr">M</div><div class="adme-name">Metabolism</div>
            <div class="adme-val">CL = ${m.clearance_L_hr} L/h</div>
            <div class="adme-det">t½ = ${m.half_life_hours} h</div></div>
        <div class="adme-box"><div class="adme-ltr">E</div><div class="adme-name">Excretion</div>
            <div class="adme-val">ke = ${e.elimination_constant}</div>
            <div class="adme-det">95%: ${e.time_to_95pct_eliminated} h</div></div>
    `;
}

/* ── L4: PD ── */
function renderPD(pd) {
    kill('chartPD');
    const pr = pd.effect_profiles;
    const names = Object.keys(pr);
    const colMap = { heart_rate: PAL.pink, systolic_bp: PAL.red, diastolic_bp: PAL.amber, contractility: PAL.blue, qt_interval: PAL.purple };
    const lMap = { heart_rate: 'Heart Rate (bpm)', systolic_bp: 'Sys. BP', diastolic_bp: 'Dia. BP', contractility: 'Contractility (%)', qt_interval: 'QT (ms)' };
    const t0 = pr[names[0]].time;
    const step = Math.max(1, Math.floor(t0.length / 200));
    const d = i => i.filter((_, idx) => idx % step === 0);

    C.chartPD = new Chart(document.getElementById('chartPD'), {
        type: 'line',
        data: {
            labels: d(t0).map(v => v.toFixed(1)),
            datasets: names.map(n => ({ label: lMap[n] || n, data: d(pr[n].values), borderColor: colMap[n] || PAL.cyan, backgroundColor: 'transparent', tension: .4, pointRadius: 0, borderWidth: 2 })),
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: { tooltip: { ...tooltipStyle, callbacks: { title: i => `${i[0].label} hrs` } } },
            scales: {
                x: { title: { display: true, text: 'Time (hours)', color: '#5a6f8a' }, ticks: { maxTicksLimit: 12 } },
                y: { title: { display: true, text: 'Effect', color: '#5a6f8a' } },
            },
        },
    });
}

/* ── L5: ECG ── */
let ecgLoop = null;
function renderECG(ecg) {
    kill('chartECG');
    if (ecgLoop) clearInterval(ecgLoop);

    const step = Math.max(1, Math.floor(ecg.time.length / 800));
    const fullV = ecg.voltage.filter((_, i) => i % step === 0);
    const fullT = ecg.time.filter((_, i) => i % step === 0);

    // Display a 3-second window
    const windowSize = Math.floor(fullV.length * 0.6);
    let viewV = fullV.slice(0, windowSize);
    let viewT = fullT.slice(0, windowSize);

    C.chartECG = new Chart(document.getElementById('chartECG'), {
        type: 'line',
        data: { labels: viewT.map(x => x.toFixed(3)), datasets: [{ label: 'ECG', data: viewV, borderColor: PAL.green, backgroundColor: PAL.greenA, fill: true, tension: .1, pointRadius: 0, borderWidth: 1.5 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: false, // Smooth manual scrolling
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            scales: {
                x: { display: false },
                y: { title: { display: true, text: 'mV', color: '#5a6f8a' }, min: -.5, max: 1.5, grid: { color: 'rgba(30,60,100,0.06)' } }
            },
        },
    });

    let cursor = windowSize;
    ecgLoop = setInterval(() => {
        if (!C.chartECG) { clearInterval(ecgLoop); return; }

        // Push points smoothly
        for (let i = 0; i < 2; i++) {
            viewV.shift();
            viewT.shift();
            viewV.push(fullV[cursor % fullV.length]);
            viewT.push(fullT[cursor % fullT.length]);
            cursor++;
        }
        C.chartECG.update();
    }, 30);
}

/* ── L5: Windkessel ── */
function renderPressure(wk) {
    kill('chartWindkessel');
    const st = Math.max(0, wk.time[wk.time.length - 1] - 3), idx = [];
    for (let i = 0; i < wk.time.length; i++) if (wk.time[i] >= st) idx.push(i);
    const step = Math.max(1, Math.floor(idx.length / 400));
    const t = idx.filter((_, i) => i % step === 0).map(i => wk.time[i]);
    const p = idx.filter((_, i) => i % step === 0).map(i => wk.pressure[i]);

    C.chartWindkessel = new Chart(document.getElementById('chartWindkessel'), {
        type: 'line',
        data: { labels: t.map(x => x.toFixed(3)), datasets: [{ label: 'Pressure', data: p, borderColor: PAL.red, backgroundColor: PAL.redA, fill: true, tension: .2, pointRadius: 0, borderWidth: 2 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { ...tooltipStyle, callbacks: { title: i => `${i[0].label} s`, label: c => `${c.parsed.y.toFixed(1)} mmHg` } } },
            scales: { x: { title: { display: true, text: 'Time (s)', color: '#5a6f8a' }, ticks: { maxTicksLimit: 10 } }, y: { title: { display: true, text: 'mmHg', color: '#5a6f8a' } } },
        },
    });
}

/* ── L5: Perfusion ── */
function renderPerfusion(co) {
    const icons = { brain: '🧠', heart: '❤️', kidneys: '🫘', liver: '🫁', skeletal_muscle: '💪', skin: '🖐️', other: '🔬' };
    let html = `<div class="perf-item" style="grid-column:1/-1;background:var(--blue-dim);border-color:rgba(37,99,235,.18)">
        <div class="perf-icon">🫀</div><div class="perf-val">${co.cardiac_output_Lmin}</div><div class="perf-unit">L/min</div>
        <div class="perf-name">Cardiac Output</div>
        <div style="font-size:10px;color:var(--t-4);margin-top:3px">EF ${co.ejection_fraction_pct}% · SV ${co.stroke_volume_mL}mL</div>
    </div>`;
    html += Object.entries(co.organ_perfusion_Lmin).map(([o, f]) =>
        `<div class="perf-item"><div class="perf-icon">${icons[o] || '🔬'}</div><div class="perf-val">${f}</div><div class="perf-unit">L/min</div><div class="perf-name">${o.replace(/_/g, ' ')}</div></div>`
    ).join('');
    document.getElementById('perfusionGrid').innerHTML = html;
}

/* ── Warnings ── */
function renderWarnings(w) {
    const el = document.getElementById('warningsContainer');
    if (!w || !w.length) { el.innerHTML = '<div class="no-warns">✅ No clinical warnings</div>'; return; }
    el.innerHTML = '<ul class="warn-list">' + w.map(x => `<li class="warn-item">⚠️ ${x}</li>`).join('') + '</ul>';
}

/* ── L6: Predictions ── */
function renderPreds(p) {
    const cards = [
        { t: '⚡ Arrhythmia Risk', v: (p.arrhythmia_risk.probability * 100).toFixed(1) + '%', lv: p.arrhythmia_risk.risk_level, d: `QTc: ${p.arrhythmia_risk.contributing_factors.QTc_interval.toFixed(0)} ms` },
        { t: '💔 Cardiac Event', v: (p.cardiac_event_risk.probability * 100).toFixed(1) + '%', lv: p.cardiac_event_risk.risk_level, d: `CO: ${p.cardiac_event_risk.contributing_factors.cardiac_output} L/min` },
        { t: '🩸 BP Response', v: `${p.bp_response.systolic_change_mmHg > 0 ? '+' : ''}${p.bp_response.systolic_change_mmHg.toFixed(0)} mmHg`, lv: p.bp_response.response_category.includes('adverse') ? 'HIGH' : 'MINIMAL', d: p.bp_response.response_category },
        { t: '💊 Effectiveness', v: (p.treatment_effectiveness.effectiveness_score * 100).toFixed(0) + '%', lv: p.treatment_effectiveness.effectiveness_score > .5 ? 'LOW' : 'MODERATE', d: p.treatment_effectiveness.effectiveness_category },
    ];
    document.getElementById('predictionCards').innerHTML = cards.map(c => `
        <div class="pred-card">
            <div class="pred-head"><span class="pred-title">${c.t}</span><span class="risk ${c.lv}">${c.lv}</span></div>
            <div class="pred-val">${c.v}</div>
            <div class="pred-detail">${c.d}</div>
        </div>
    `).join('');
}

/* ── L7: Safety ── */
function renderSafety(s) {
    const pct = (s.safety_score / s.max_score) * 100;
    const barCls = s.color === 'green' ? 'green' : s.color === 'red' ? 'red' : s.color === 'yellow' ? 'yellow' : 'orange';
    const numColor = s.color === 'green' ? PAL.green : s.color === 'red' ? PAL.red : PAL.amber;
    const clsBg = s.classification === 'Safe' ? 'background:var(--green-dim);color:var(--green);border:1px solid rgba(5,150,105,.2)' :
        s.classification === 'Caution' ? 'background:var(--amber-dim);color:var(--amber);border:1px solid rgba(217,119,6,.2)' :
            s.classification === 'Warning' ? 'background:rgba(234,88,12,.10);color:#ea580c;border:1px solid rgba(234,88,12,.2)' :
                'background:var(--red-dim);color:var(--red);border:1px solid rgba(220,38,38,.2)';

    const demerits = Object.entries(s.demerits_breakdown).map(([n, v]) =>
        `<div class="demerit-row">
            <span class="demerit-label">${n.replace(/_/g, ' ')}</span>
            <div class="demerit-track"><div style="width:${Math.min(100, v * 33)}%;height:100%;background:${v > 2 ? PAL.red : v > 1 ? PAL.amber : PAL.green};border-radius:3px;"></div></div>
            <span class="demerit-val">${v.toFixed(1)}</span>
        </div>`
    ).join('');

    document.getElementById('safetyMeter').innerHTML = `
        <div class="safety-display">
            <span class="safety-num" style="color:${numColor}">${s.safety_score.toFixed(1)}</span>
            <span class="safety-max">/ ${s.max_score}</span>
            <span class="safety-class" style="${clsBg}">${s.classification}</span>
        </div>
        <div class="bar-outer"><div class="bar-inner ${barCls}" style="width:${pct}%"></div></div>
        <div style="margin-top:20px;font-size:10.5px;font-weight:700;color:var(--t-3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px;">Demerits Breakdown</div>
        ${demerits}
    `;
}

/* ── Times ── */
function renderTimes(info) {
    const lt = info.layer_times;
    const ms = v => `${(v * 1000).toFixed(0)}ms`;
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set('tL1', ms(lt.drug_representation));
    set('tL2', ms(lt.drug_target_interaction));
    set('tL3', ms(lt.pharmacokinetics));
    set('tL4', ms(lt.pharmacodynamics));
    set('tL5', ms(lt.cardiovascular_model));
    set('tL6', ms(lt.ml_predictions));
    set('tTotal', `Total: ${(info.total_computation_time_seconds * 1000).toFixed(0)}ms`);
}

/* ══════════════════════════════════
   HEART SYNC
   ══════════════════════════════════ */
function syncHeart(d) {
    const iframe = document.getElementById('heartIframe');
    if (!iframe || !iframe.contentWindow) return;
    const cv = d.layer_5_cardiovascular.cardiac_parameters;
    iframe.contentWindow.postMessage({
        type: 'drug-sim-params',
        drug_name: d.simulation_info.drug_name,
        dose_mg: d.simulation_info.dose_mg,
        heart_rate: cv.heart_rate_bpm,
        systolic_bp: cv.systolic_bp_mmHg,
        contractility: cv.contractility_pct,
        qt_interval: cv.qt_interval_ms,
        cardiac_state: d.layer_5_cardiovascular.cardiac_state,
        warnings: d.layer_5_cardiovascular.warnings || [],
    }, '*');
}

/* ── Utils ── */
function kill(id) { if (C[id]) { C[id].destroy(); delete C[id]; } }