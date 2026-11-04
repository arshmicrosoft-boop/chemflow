import React, { useLayoutEffect, useRef, useState } from 'react';

export const fmt = (x, digits = 3) => (x === null || x === undefined ? '—' : Number(x).toFixed(digits));

const METRIC_LABELS = { rmse: 'RMSE', mae: 'MAE', r2: 'R²', roc_auc: 'ROC AUC' };
export const metricLabel = (m) => METRIC_LABELS[m] || m;

const TERMS = {
  gbm: 'GBM',
  mlp: 'MLP',
  ridge: 'Ridge',
  random_forest: 'Random forest',
  ecfp: 'ECFP',
  maccs: 'MACCS',
  rdkit_descriptors: 'RDKit descriptors',
  rmse: 'RMSE',
  mae: 'MAE',
  r2: 'R²',
};
// Engine text names models and metrics by their config keys.
export const pretty = (text) => text
  .replace(/\b(gbm|mlp|ridge|random_forest|ecfp|maccs|rdkit_descriptors|rmse|mae|r2)\b/g, t => TERMS[t])
  .replaceAll('>=', '≥')
  .replaceAll('<=', '≤');
export const modelLabel = (m) => `${pretty(m.model)} · ${pretty(m.featurizer)}`;

const KINDS = {
  train_evaluate: 'Train and evaluate',
  replicate: 'Replicate',
  sar_test: 'Structure test',
  virtual_assay: 'Virtual assay',
  holdout_validation: 'Hold-out validation',
  lab_proposal: 'Lab proposal',
};
export const kindLabel = (kind) => KINDS[kind] || kind.replaceAll('_', ' ');

const STATUS = {
  validated: ['good', '✓', 'Validated'],
  completed: ['good', '✓', 'Completed'],
  approved: ['good', '✓', 'Approved'],
  provisional: ['warn', '◐', 'Provisional'],
  pending: ['warn', '◐', 'Awaiting decision'],
  rejected: ['bad', '✕', 'Rejected'],
  refuted: ['bad', '✕', 'Refuted'],
  failed: ['bad', '✕', 'Failed'],
  timeout: ['bad', '✕', 'Timed out'],
  inconclusive: ['calm', '?', 'Inconclusive'],
  blocked: ['calm', '⊘', 'Blocked'],
  abandoned: ['calm', '⊘', 'Abandoned'],
  proposed: ['calm', '○', 'Proposed'],
  tested: ['calm', '✓', 'Tested'],
};

export function Status({ value }) {
  const [tone, icon, label] = STATUS[value] || ['calm', '○', value];
  return <span className={`pill pill-${tone}`}><span aria-hidden="true">{icon}</span>{label}</span>;
}

export function Glass({ as: Tag = 'section', className = '', children, ...rest }) {
  return <Tag className={`glass ${className}`} {...rest}>{children}</Tag>;
}

export function Panel({ title, hint, action, children, className = '' }) {
  return (
    <Glass className={`p-5 sm:p-6 rise ${className}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 mb-4">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
          {hint && <p className="text-sm text-ink-2 mt-0.5">{hint}</p>}
        </div>
        {action}
      </div>
      {children}
    </Glass>
  );
}

export function Stat({ label, value, note }) {
  return (
    <div className="plate p-4 min-w-0">
      <div className="text-xs font-medium text-ink-2">{label}</div>
      <div className="tnum text-2xl sm:text-3xl font-semibold tracking-tight mt-1 truncate">{value}</div>
      {note && <div className="text-xs text-ink-2 mt-1">{note}</div>}
    </div>
  );
}

export function Budget({ label, used, total }) {
  const pct = total ? Math.min(100, (used / total) * 100) : 0;
  return (
    <div>
      <div className="flex justify-between text-sm mb-1.5">
        <span className="text-ink-2">{label}</span>
        <span className="tnum font-semibold">{used} / {total}</span>
      </div>
      <div className="meter" role="img" aria-label={`${label}: ${used} of ${total} used`}><i style={{ width: `${pct}%` }} /></div>
    </div>
  );
}

export function Notice({ tone = 'calm', title, children, action }) {
  return (
    <Glass className="p-5 rise flex flex-wrap items-center justify-between gap-3" role={tone === 'bad' ? 'alert' : undefined}>
      <div>
        <div className="font-semibold">{title}</div>
        {children && <div className="text-sm text-ink-2 mt-0.5">{children}</div>}
      </div>
      {action}
    </Glass>
  );
}

export function Segmented({ options, value, onChange, label }) {
  const ref = useRef(null);
  const [thumb, setThumb] = useState(null);

  useLayoutEffect(() => {
    const place = () => {
      const el = ref.current?.querySelector('[aria-selected="true"]');
      if (el) setThumb({ x: el.offsetLeft, width: el.offsetWidth });
    };
    place();
    window.addEventListener('resize', place);
    return () => window.removeEventListener('resize', place);
  }, [value, options.length]);

  return (
    <div ref={ref} role="tablist" aria-label={label} className="glass lens segmented">
      {thumb && <span className="thumb" style={{ width: thumb.width, transform: `translateX(${thumb.x}px)` }} />}
      {options.map(o => (
        <button key={o.value} role="tab" aria-selected={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
