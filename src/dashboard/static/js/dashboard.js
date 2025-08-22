function showTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  const tab = document.getElementById(tabId);
  if (tab) tab.classList.add('active');
  initializeChartsForTab(tabId);
}

function initializeChartsForTab(tabId) {
  setTimeout(() => {
    switch (tabId) {
      case 'advanced-analysis':
        initializeAdvancedCharts();
        break;
      case 'predictions':
        initializePredictionCharts();
        break;
    }
  }, 50);
}

function initializeAdvancedCharts() {
  const patternCtx = document.getElementById('patternChart');
  if (patternCtx && !patternCtx.chart) {
    patternCtx.chart = new Chart(patternCtx, {
      type: 'bar',
      data: { labels: ['Sector Rotation', 'Volume Spikes', 'Timing', 'Delays'], datasets: [{ data: [8,12,6,15], backgroundColor: ['#3b82f6','#ef4444','#f59e0b','#10b981'] }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
    });
  }

  const timingCtx = document.getElementById('timingChart');
  if (timingCtx && !timingCtx.chart) {
    timingCtx.chart = new Chart(timingCtx, {
      type: 'line',
      data: { labels: ['Pre-30d','Pre-15d','Pre-5d','Event','Post-5d','Post-15d'], datasets: [{ data: [2.1,3.8,5.2,1.0,0.8,1.2], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', fill: true }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
    });
  }

  const suspicionCtx = document.getElementById('suspicionChart');
  if (suspicionCtx && !suspicionCtx.chart) {
    suspicionCtx.chart = new Chart(suspicionCtx, {
      type: 'doughnut',
      data: { labels: ['Low','Medium','High','Extreme'], datasets: [{ data: [6,5,3,1], backgroundColor: ['#10b981','#f59e0b','#ef4444','#dc2626'] }] },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }
}

function initializePredictionCharts() {
  const impactCtx = document.getElementById('impactChart');
  if (impactCtx && !impactCtx.chart) {
    impactCtx.chart = new Chart(impactCtx, { type: 'bar', data: { labels: ['Nancy P.','Richard B.','Paul P.','Josh G.','Joe M.'], datasets: [{ data: [8.7,4.2,6.1,2.8,3.4], backgroundColor: '#3b82f6' }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } } });
  }

  const legislationCtx = document.getElementById('legislationChart');
  if (legislationCtx && !legislationCtx.chart) {
    legislationCtx.chart = new Chart(legislationCtx, { type: 'radar', data: { labels: ['AI Safety','Banking','Energy','Healthcare'], datasets: [{ data: [0.89,0.67,0.75,0.82], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)' }] }, options: { responsive: true, maintainAspectRatio: false, scales: { r: { beginAtZero: true, max: 1 } } } });
  }

  const performanceCtx = document.getElementById('performanceChart');
  if (performanceCtx && !performanceCtx.chart) {
    performanceCtx.chart = new Chart(performanceCtx, { type: 'line', data: { labels: ['Q1 2023','Q2 2023','Q3 2023','Q4 2023','Q1 2024'], datasets: [{ label: 'Congressional', data: [12.3,8.7,15.2,22.1,18.4], borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)' }, { label: 'S&P 500', data: [7.5,8.3,6.9,11.2,10.5], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)' }] }, options: { responsive: true, maintainAspectRatio: false, scales: { y: { title: { display: true, text: 'Returns (%)' } } } } });
  }
}

document.addEventListener('DOMContentLoaded', function () {
  // Load live stats from backend
  try {
    fetch('/api/stats')
      .then(r => r.json())
      .then(resp => {
        const data = resp && (resp.data || resp);
        if (!data) return;
        const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        setText('total-members', data.total_members ?? '—');
        setText('total-volume', data.total_volume ? `$${(data.total_volume/1_000_000).toFixed(1)}M` : '—');
        setText('avg-suspicion', data.avg_suspicion_score ?? '—');
        setText('anomalies-detected', data.high_risk_members ?? '—');
      })
      .catch(() => {});
  } catch (_) {}

  initializeChartsForTab('advanced-analysis');
});


