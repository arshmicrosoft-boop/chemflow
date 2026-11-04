import React, { useEffect, useState } from 'react';
import { fmt, metricLabel } from './ui.jsx';

const useReducedMotion = () => {
  const [reduced] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  return reduced;
};

export function ElectricLogo({ size = 30 }) {
  const reduced = useReducedMotion();
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" className="electric-logo" aria-hidden="true">
      <defs>
        <linearGradient id="electric-stroke" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#6db0ff" />
          <stop offset="1" stopColor="#36d6b4" />
        </linearGradient>
        <filter id="electric-arc" x="-20%" y="-20%" width="140%" height="140%">
          <feTurbulence type="turbulence" baseFrequency="0.09" numOctaves="2" seed="2" result="noise">
            {!reduced && <animate attributeName="seed" values="1;4;2;7;3;9;5" dur="0.7s" repeatCount="indefinite" calcMode="discrete" />}
          </feTurbulence>
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="2.4" xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </defs>
      <path d="M16 3 27.3 9.5v13L16 29 4.7 22.5v-13Z" fill="none" stroke="url(#electric-stroke)" strokeWidth="2" strokeLinejoin="round" filter="url(#electric-arc)" />
      <path d="M17.8 8.5 10.6 17.4h4.6l-1.2 6.1 7.2-8.9h-4.6Z" fill="url(#electric-stroke)" className="electric-bolt" />
    </svg>
  );
}

const GLYPHS = '01<>/\\#%&*+=ABCDEF';

// Text that decodes into place. Screen readers get the plain text.
export function TechText({ text, className = '' }) {
  const reduced = useReducedMotion();
  const [shown, setShown] = useState(reduced ? text : '');

  useEffect(() => {
    if (reduced) return setShown(text);
    let frame = 0;
    const timer = setInterval(() => {
      frame += 1;
      const settled = Math.floor(frame / 1.6);
      if (settled >= text.length) {
        setShown(text);
        clearInterval(timer);
        return;
      }
      setShown(text.slice(0, settled) + [...text.slice(settled)].map(c => (
        c === ' ' ? ' ' : GLYPHS[Math.floor(Math.random() * GLYPHS.length)]
      )).join(''));
    }, 28);
    return () => clearInterval(timer);
  }, [text, reduced]);

  return (
    <span className={`tech ${className}`}>
      <span className="sr-only">{text}</span>
      <span aria-hidden="true">{shown}</span>
    </span>
  );
}

function Sparkline({ values, label }) {
  const width = 72;
  const height = 22;
  const lo = Math.min(...values);
  const span = Math.max(...values) - lo || 1;
  const x = (i) => 2 + (i / (values.length - 1)) * (width - 6);
  const y = (v) => height - 3 - ((v - lo) / span) * (height - 6);
  const last = values.length - 1;
  return (
    <svg width={width} height={height} role="img" aria-label={label} className="shrink-0">
      <title>{label}</title>
      <polyline points={values.map((v, i) => `${x(i)},${y(v)}`).join(' ')} fill="none" stroke="var(--accent)" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={x(last)} cy={y(values[last])} r="2.5" fill="var(--accent)" />
    </svg>
  );
}

export function MicroStats({ items, className = '' }) {
  return (
    <dl className={`grid grid-cols-2 lg:grid-cols-4 gap-2 ${className}`}>
      {items.map(item => (
        <div key={item.label} className="micro-stat">
          <dt className="tech text-[10px] text-ink-2 truncate">{item.label}</dt>
          <dd className="flex items-center justify-between gap-2 mt-0.5 min-w-0">
            <span className="tnum text-base font-semibold truncate">{item.value}</span>
            {item.spark?.length > 1 && <Sparkline values={item.spark} label={`${item.label} trend, from ${fmt(item.spark[0])} to ${fmt(item.spark[item.spark.length - 1])}`} />}
          </dd>
          {item.share !== undefined && (
            <div className="meter mt-1.5" style={{ height: 3 }}><i style={{ width: `${Math.min(100, item.share * 100)}%` }} /></div>
          )}
        </div>
      ))}
    </dl>
  );
}

export function projectMicroStats(results) {
  const { counts, budgets, experiments, learning_curve: curve, objective, champion } = results;
  const done = counts.experiments.completed || 0;
  const hypotheses = Object.values(counts.hypotheses).reduce((a, b) => a + b, 0);
  const assessed = hypotheses - (counts.hypotheses.proposed || 0);
  return [
    { label: `Best ${metricLabel(objective.metric)}`, value: fmt(champion?.metric), spark: curve.map(c => c.metric) },
    { label: 'Runs completed', value: `${done} / ${experiments.length}`, share: experiments.length ? done / experiments.length : 0 },
    { label: 'Hypotheses assessed', value: `${assessed} / ${hypotheses}`, share: hypotheses ? assessed / hypotheses : 0 },
    { label: 'Assays used', value: `${budgets.assays.used} / ${budgets.assays.total}`, share: budgets.assays.used / budgets.assays.total },
  ];
}

const STEPS = [
  ['Hypothesize', 'Proposes testable statements from the data and past results.'],
  ['Find gaps', 'Looks for what is still unknown or under-tested.'],
  ['Select', 'Picks the experiment expected to teach the most.'],
  ['Run', 'Trains DeepChem models inside the sandbox.'],
  ['Evaluate', 'Checks results against the target and confidence limits.'],
  ['Learn', 'Keeps a better model only when validated evidence supports it.'],
  ['Remember', 'Records every result, including failures, then repeats.'],
];
const ROTATE_MS = 3200;
const RADIUS = 225;

export function CircularCarousel() {
  const reduced = useReducedMotion();
  const [turn, setTurn] = useState(0);
  const [paused, setPaused] = useState(false);
  const count = STEPS.length;
  const active = ((turn % count) + count) % count;
  const angle = 360 / count;

  useEffect(() => {
    if (reduced || paused) return undefined;
    const timer = setInterval(() => setTurn(t => t + 1), ROTATE_MS);
    return () => clearInterval(timer);
  }, [reduced, paused]);

  return (
    <div onPointerEnter={() => setPaused(true)} onPointerLeave={() => setPaused(false)} onFocus={() => setPaused(true)} onBlur={() => setPaused(false)}>
      <div className="carousel" aria-hidden="true">
        <div className="carousel-ring" style={{ transform: `translateZ(-${RADIUS}px) rotateY(${-turn * angle}deg)` }}>
          {STEPS.map(([name], i) => {
            const distance = Math.min((i - active + count) % count, (active - i + count) % count);
            return (
              <div key={name} className={`carousel-card ${i === active ? 'is-active' : ''}`} style={{ transform: `rotateY(${i * angle}deg) translateZ(${RADIUS}px)`, opacity: 1 - distance * 0.2 }}>
                <span className="tech text-[10px] text-ink-2">Step {i + 1}</span>
                <span className="font-semibold">{name}</span>
              </div>
            );
          })}
        </div>
      </div>
      <div className="flex items-center justify-center gap-3 mt-3">
        <button className="btn !px-3" aria-label="Previous step" onClick={() => setTurn(t => t - 1)}><span aria-hidden="true">‹</span></button>
        <p className="text-sm text-ink-2 text-center min-h-10 max-w-sm" aria-live={paused ? 'polite' : 'off'}>
          <strong className="text-ink">{active + 1}. {STEPS[active][0]}.</strong> {STEPS[active][1]}
        </p>
        <button className="btn !px-3" aria-label="Next step" onClick={() => setTurn(t => t + 1)}><span aria-hidden="true">›</span></button>
      </div>
    </div>
  );
}
