import React, { useEffect, useState } from 'react';
import { api } from '../api.js';
import { metricLabel } from './ui.jsx';

const METRICS = {
  regression: { r2: 0.7, rmse: 0.75, mae: 0.6 },
  classification: { roc_auc: 0.8, accuracy: 0.8 },
};

const METRIC_HELP = {
  r2: 'How much of the variation the model explains. 1 is perfect.',
  rmse: 'Typical prediction error, in the units of your target. Lower is better.',
  mae: 'Average prediction error, in the units of your target. Lower is better.',
  roc_auc: 'How well the model ranks positives above negatives. 1 is perfect, 0.5 is guessing.',
  accuracy: 'Share of molecules classified correctly. 1 is perfect.',
};

function Field({ label, hint, children }) {
  return (
    <label className="block min-w-0">
      <span className="block text-sm font-medium mb-1">{label}</span>
      {children}
      {hint && <span className="block text-xs text-ink-2 mt-1">{hint}</span>}
    </label>
  );
}

export default function NewProjectForm({ onCreated, onCancel }) {
  const [datasets, setDatasets] = useState(null);
  const [info, setInfo] = useState(null);
  const [form, setForm] = useState({
    name: '', question: '', dataset: '', smiles_column: '', target_column: '',
    task: 'regression', primary_metric: 'r2', success_threshold: 0.7,
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const update = (fields) => setForm(f => ({ ...f, ...fields }));
  const fail = (e) => setError(e.message);

  const chooseDataset = (name) => {
    update({ dataset: name, smiles_column: '', target_column: '' });
    setInfo(null);
    setError(null);
    if (!name) return;
    api.dataset(name).then(d => {
      setInfo(d);
      update({ smiles_column: d.smiles_column || '' });
      if (!d.smiles_column) setError('No SMILES column was detected. Pick the molecule column yourself, or choose another dataset.');
    }).catch(fail);
  };

  useEffect(() => {
    api.datasets().then(names => {
      setDatasets(names);
      if (names.includes('delaney-processed.csv')) chooseDataset('delaney-processed.csv');
    }).catch(fail);
  }, []);

  const chooseMetric = (task, metric) => update({ task, primary_metric: metric, success_threshold: METRICS[task][metric] });

  const chooseTarget = (name) => {
    const binary = info.columns.find(c => c.name === name)?.kind === 'binary';
    update({ target_column: name });
    if (binary && form.task !== 'classification') chooseMetric('classification', 'roc_auc');
    if (!binary && form.task !== 'regression') chooseMetric('regression', 'r2');
  };

  const upload = async (e) => {
    const file = e.target.files[0];
    e.target.value = '';
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const uploaded = await api.uploadDataset(file);
      setDatasets(names => [...names, uploaded.name].sort());
      chooseDataset(uploaded.name);
    } catch (err) {
      fail(err);
    }
    setBusy(false);
  };

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const { question, ...rest } = form;
      onCreated(await api.createProject({
        ...rest,
        name: form.name.trim(),
        objective: question.trim(),
        question: question.trim(),
        success_threshold: Number(form.success_threshold),
      }));
    } catch (err) {
      fail(err);
      setBusy(false);
    }
  };

  const targets = info?.columns.filter(c => c.kind !== 'text' && c.name !== form.smiles_column) || [];
  const binaryTarget = targets.find(c => c.name === form.target_column)?.kind === 'binary';
  const lowerBetter = ['rmse', 'mae'].includes(form.primary_metric);
  const ready = form.name.trim() && form.question.trim() && form.smiles_column && form.target_column && form.success_threshold !== '';

  return (
    <form onSubmit={submit} className="plate p-4 sm:p-5 mb-4 space-y-4">
      <div className="grid md:grid-cols-2 gap-4">
        <Field label="Project name">
          <input autoFocus className="field w-full" value={form.name} onChange={e => update({ name: e.target.value })} placeholder="BACE inhibitors" />
        </Field>
        <Field label="Research question">
          <input className="field w-full" value={form.question} onChange={e => update({ question: e.target.value })} placeholder="Can we predict binding strength from structure?" />
        </Field>
      </div>

      <Field label="Dataset" hint="A CSV with a column of SMILES molecules and a column of measured values, with at least 50 molecules.">
        <div className="flex flex-wrap gap-2">
          <select className="field flex-1 min-w-0" value={form.dataset} onChange={e => chooseDataset(e.target.value)} disabled={!datasets}>
            <option value="">{datasets ? 'Choose a dataset' : 'Loading datasets…'}</option>
            {datasets?.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
          <span className={`btn cursor-pointer ${busy ? 'opacity-45' : ''}`}>
            Upload CSV
            <input type="file" accept=".csv" className="sr-only" onChange={upload} disabled={busy} />
          </span>
        </div>
      </Field>

      {info && (
        <>
          <div className="grid md:grid-cols-2 gap-4">
            <Field label="Molecule column" hint="The column holding SMILES strings.">
              <select className="field w-full" value={form.smiles_column} onChange={e => update({ smiles_column: e.target.value, target_column: e.target.value === form.target_column ? '' : form.target_column })}>
                <option value="">Choose a column</option>
                {info.columns.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
            </Field>
            <Field label="Value to predict" hint={targets.length ? 'What the models should learn. Only numeric columns are listed.' : 'This dataset has no numeric columns to predict.'}>
              <select className="field w-full" value={form.target_column} onChange={e => chooseTarget(e.target.value)}>
                <option value="">Choose a column</option>
                {targets.map(c => <option key={c.name} value={c.name}>{c.name}{c.kind === 'binary' ? ' (0 or 1)' : ''}</option>)}
              </select>
            </Field>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <Field label="Score models by" hint={METRIC_HELP[form.primary_metric]}>
              <select className="field w-full" value={form.primary_metric} onChange={e => chooseMetric(form.task, e.target.value)}>
                {Object.keys(METRICS[form.task]).map(m => <option key={m} value={m}>{metricLabel(m)}</option>)}
              </select>
            </Field>
            <Field label={`Success target (${lowerBetter ? 'at most' : 'at least'})`} hint="A finding is only validated if a model reaches this on data it never trained on.">
              <input type="number" step="any" className="field w-full tnum" value={form.success_threshold} onChange={e => update({ success_threshold: e.target.value })} />
            </Field>
          </div>

          {binaryTarget && <p className="text-xs text-ink-2">This target only contains 0 and 1, so models will classify rather than predict a number.</p>}
        </>
      )}

      {error && <p role="alert" className="text-sm" style={{ color: 'var(--bad)' }}>{error}</p>}

      <div className="flex flex-wrap gap-2 justify-end">
        <button type="button" className="btn" onClick={onCancel}>Cancel</button>
        <button className="btn btn-primary" disabled={busy || !ready}>{busy ? 'Working…' : 'Create project'}</button>
      </div>
    </form>
  );
}
