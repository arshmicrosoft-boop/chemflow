import React, { useEffect, useRef, useState } from 'react';
import { Status, fmt, metricLabel, modelLabel, pretty } from './ui.jsx';

function ticks(lo, hi, count = 5) {
  const raw = (hi - lo) / (count - 1) || 1;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw);
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-6; t += step) out.push(Number(t.toFixed(6)));
  return out;
}

function padded(values, ratio = 0.08) {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = (hi - lo || 1) * ratio;
  return [lo - pad, hi + pad];
}

const percent = ([lo, hi]) => (v) => ((v - lo) / (hi - lo)) * 100;

function Mark({ shape, color, at, hollow }) {
  const inline = at === undefined;
  const size = shape === 'circle' ? 12 : 10;
  const rotate = shape === 'diamond' ? 'rotate(45deg)' : '';
  const style = {
    width: size,
    height: size,
    background: hollow ? 'var(--plate)' : color,
    border: hollow ? `2px solid ${color}` : undefined,
    borderRadius: shape === 'circle' ? '50%' : 2,
    boxShadow: inline ? undefined : '0 0 0 2px var(--plate)',
    transform: inline ? rotate : `translate(-50%, -50%) ${rotate}`,
    ...(inline ? {} : { position: 'absolute', top: '50%', left: `${at}%` }),
  };
  return <i style={style} className="inline-block" />;
}

function Legend({ items }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2 mb-3">
      {items.map(i => (
        <span key={i.label} className="inline-flex items-center gap-1.5">
          <Mark shape={i.shape} color={i.color} hollow={i.hollow} />{i.label}
        </span>
      ))}
    </div>
  );
}

const ROW_GRID = 'sm:grid-cols-[190px_minmax(0,1fr)_128px]';

// Rows of labels and values around a shared horizontal scale. The values stay
// visible in each row, so the chart is its own table view.
function IntervalRows({ domain, rows, axisTitle, reference }) {
  const x = percent(domain);
  const [active, setActive] = useState(null);

  return (
    <div className="plate p-3 sm:p-4">
      <ul>
        {rows.map(row => (
          <li
            key={row.key}
            tabIndex={0}
            onPointerEnter={() => setActive(row.key)}
            onPointerLeave={() => setActive(null)}
            onFocus={() => setActive(row.key)}
            onBlur={() => setActive(null)}
            className={`relative grid grid-cols-[minmax(0,1fr)_auto] ${ROW_GRID} gap-x-3 items-center rounded-lg px-2 py-1 hover:bg-[var(--calm-bg)]`}
          >
            <div className="text-sm font-medium sm:truncate first-letter:uppercase" title={row.label}>{row.label}</div>
            <div className="sm:order-last text-right text-sm">{row.aside}</div>
            <div className="relative h-7 col-span-2 sm:col-span-1">
              <span className="absolute inset-x-0 top-1/2 h-px" style={{ background: 'var(--hairline)' }} />
              {reference !== undefined && (
                <span className="absolute -inset-y-1 border-l border-dashed" style={{ left: `${x(reference)}%`, borderColor: 'var(--ink-3)' }} />
              )}
              {row.marks(x)}
              {active === row.key && (
                <div className="tip" style={{ bottom: '100%', left: `${Math.min(70, Math.max(30, row.anchor(x)))}%`, transform: 'translate(-50%, -4px)' }}>
                  {row.tip}
                </div>
              )}
            </div>
          </li>
        ))}
      </ul>
      <div className={`grid grid-cols-1 ${ROW_GRID} gap-x-3 px-2 mt-1`}>
        <div className="hidden sm:block" />
        <div className="relative h-5 text-[11px] text-ink-2 tnum">
          {ticks(domain[0], domain[1], 7).map(t => (
            <span key={t} className="absolute top-1" style={{ left: `${x(t)}%`, transform: 'translateX(-50%)' }}>{t}</span>
          ))}
        </div>
      </div>
      <div className="text-xs text-ink-2 text-center mt-1">{axisTitle}</div>
    </div>
  );
}

export function ModelLeaderboard({ models, objective }) {
  const { threshold, metric } = objective;
  const domain = padded(models.flatMap(m => m.ci).concat(threshold), 0.05);
  const name = metricLabel(metric);
  return (
    <>
      <Legend items={[
        { label: 'Meets target', shape: 'circle', color: 'var(--series-1)' },
        { label: 'Misses target', shape: 'circle', color: 'var(--ink-3)', hollow: true },
      ]} />
      <IntervalRows
        domain={domain}
        reference={threshold}
        axisTitle={`Validation ${name} with ${Math.round(objective.confidence_level * 100)}% confidence interval. Dashed line: target ${threshold}`}
        rows={models.map(m => {
          const color = m.meets_target ? 'var(--series-1)' : 'var(--ink-3)';
          return {
            key: `${m.model}/${m.featurizer}`,
            label: modelLabel(m),
            anchor: (x) => x(m.metric),
            aside: (
              <span className="inline-flex items-center gap-2 justify-end">
                <span className="tnum font-semibold">{fmt(m.metric)}</span>
                <Status value={m.status} />
              </span>
            ),
            marks: (x) => (
              <>
                <span className="absolute top-1/2 h-0.5 -mt-px rounded" style={{ left: `${x(m.ci[0])}%`, width: `${x(m.ci[1]) - x(m.ci[0])}%`, background: color }} />
                <Mark shape="circle" at={x(m.metric)} color={color} hollow={!m.meets_target} />
              </>
            ),
            tip: (
              <>
                <div className="tnum font-semibold">{name} {fmt(m.metric)} <span className="font-normal text-ink-2">({fmt(m.ci[0])} to {fmt(m.ci[1])})</span></div>
                <div className="text-ink-2">{m.n_train} molecules · {m.configs_tested} configuration{m.configs_tested === 1 ? '' : 's'}</div>
                {m.replicate_metric !== null && <div className="text-ink-2 tnum">Replicate {fmt(m.replicate_metric)}</div>}
                {m.holdout_metric !== null && <div className="text-ink-2 tnum">Hold-out {fmt(m.holdout_metric)}</div>}
              </>
            ),
          };
        })}
      />
    </>
  );
}

const EFFECT_SERIES = [
  { key: 'discovery', label: 'Discovery (training data)', shape: 'circle', color: 'var(--series-1)' },
  { key: 'replication', label: 'Validation', shape: 'diamond', color: 'var(--series-2)' },
  { key: 'holdout', label: 'Hold-out', shape: 'square', color: 'var(--series-3)' },
];

export function StructureEffects({ effects }) {
  const series = EFFECT_SERIES.filter(s => effects.some(e => e[s.key]));
  const sizes = effects.flatMap(e => series.filter(s => e[s.key]).map(s => e[s.key].effect_size));
  const reach = Math.max(...sizes.map(Math.abs), 0.5) * 1.1;

  return (
    <>
      <Legend items={series} />
      <IntervalRows
        domain={[-reach, reach]}
        reference={0}
        axisTitle="Effect size d: gap between molecules with and without the group, in standard deviations"
        rows={effects.map(e => {
          const present = series.filter(s => e[s.key]);
          return {
            key: e.group,
            label: e.group,
            anchor: (x) => x(e.discovery.effect_size),
            aside: (
              <span className="inline-flex items-center gap-2 justify-end">
                <span className="tnum font-semibold">{fmt(e.discovery.effect_size, 2)}</span>
                <Status value={e.status} />
              </span>
            ),
            marks: (x) => {
              const at = present.map(s => x(e[s.key].effect_size));
              return (
                <>
                  <span className="absolute top-1/2 h-0.5 -mt-px" style={{ left: `${Math.min(...at)}%`, width: `${Math.max(...at) - Math.min(...at)}%`, background: 'var(--ink-3)' }} />
                  {present.map(s => <Mark key={s.key} shape={s.shape} color={s.color} at={x(e[s.key].effect_size)} />)}
                </>
              );
            },
            tip: present.map(s => (
              <div key={s.key} className="tnum">
                <span className="font-semibold">d = {fmt(e[s.key].effect_size, 2)}</span>{' '}
                <span className="text-ink-2">{s.label}, p = {e[s.key].p_value === null ? 'n/a' : e[s.key].p_value.toExponential(1)}, n = {e[s.key].n_with}</span>
              </div>
            )),
          };
        })}
      />
    </>
  );
}

function useWidth() {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

export function LearningCurve({ curve, objective }) {
  const [ref, width] = useWidth();
  const [active, setActive] = useState(null);
  const height = 260;
  const pad = { top: 18, right: 18, bottom: 34, left: 46 };
  const { threshold, metric, lower_is_better: lowerBetter } = objective;
  const cycles = curve.map(c => c.cycle);
  const xLo = Math.min(...cycles);
  const xHi = Math.max(...cycles);
  const yDomain = padded(curve.map(c => c.metric).concat(threshold));
  const plotW = Math.max(0, width - pad.left - pad.right);
  const plotH = height - pad.top - pad.bottom;
  const px = (c) => pad.left + (xHi === xLo ? plotW / 2 : ((c - xLo) / (xHi - xLo)) * plotW);
  const py = (v) => pad.top + (1 - percent(yDomain)(v) / 100) * plotH;
  const accepted = curve.filter(c => c.accepted);
  const line = accepted.map((c, i) => (
    i ? `H${px(c.cycle)}V${py(c.metric)}` : `M${px(c.cycle)},${py(c.metric)}`
  )).join('') + (accepted.length ? `H${px(xHi)}` : '');

  const nearest = (clientX) => {
    const pointer = clientX - ref.current.getBoundingClientRect().left;
    return curve.reduce((best, c, i) => (
      Math.abs(px(c.cycle) - pointer) < Math.abs(px(curve[best].cycle) - pointer) ? i : best
    ), 0);
  };
  const onKey = (e) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (step) setActive(Math.max(0, Math.min(curve.length - 1, (active ?? (step > 0 ? -1 : curve.length)) + step)));
  };
  const hover = active === null ? null : curve[active];

  return (
    <div
      ref={ref}
      className="plate relative"
      tabIndex={0}
      role="img"
      aria-label={`Champion ${metricLabel(metric)} over research cycles, from ${fmt(curve[0].metric)} to ${fmt(curve[curve.length - 1].metric)}`}
      onPointerMove={(e) => setActive(nearest(e.clientX))}
      onPointerLeave={() => setActive(null)}
      onBlur={() => setActive(null)}
      onKeyDown={onKey}
    >
      {width > 0 && (
        <svg width={width} height={height} className="block">
          {ticks(yDomain[0], yDomain[1], 4).map(t => (
            <g key={t}>
              <line x1={pad.left} x2={width - pad.right} y1={py(t)} y2={py(t)} stroke="var(--grid)" />
              <text x={pad.left - 8} y={py(t) + 4} textAnchor="end" fontSize="11" fill="var(--ink-2)" className="tnum">{t}</text>
            </g>
          ))}
          {ticks(xLo, Math.max(xHi, xLo + 1), 6).filter(t => Number.isInteger(t)).map(t => (
            <text key={t} x={px(t)} y={height - 18} textAnchor="middle" fontSize="11" fill="var(--ink-2)" className="tnum">{t}</text>
          ))}
          <text x={pad.left + plotW / 2} y={height - 3} textAnchor="middle" fontSize="11" fill="var(--ink-2)">Research cycle</text>
          <line x1={pad.left} x2={width - pad.right} y1={py(threshold)} y2={py(threshold)} stroke="var(--ink-3)" strokeDasharray="5 4" />
          <text x={width - pad.right} y={py(threshold) + (lowerBetter ? -6 : 14)} textAnchor="end" fontSize="11" fontWeight="600" fill="var(--ink-2)">Target {threshold}</text>
          <path d={line} fill="none" stroke="var(--series-1)" strokeWidth="2" strokeLinejoin="round" />
          {hover && <line x1={px(hover.cycle)} x2={px(hover.cycle)} y1={pad.top} y2={pad.top + plotH} stroke="var(--ink-3)" />}
          {curve.map((c) => (
            <circle
              key={c.version}
              cx={px(c.cycle)}
              cy={py(c.metric)}
              r={hover === c ? 6 : 4.5}
              fill={c.accepted ? 'var(--series-1)' : 'var(--plate)'}
              stroke={c.accepted ? 'var(--plate)' : 'var(--ink-3)'}
              strokeWidth="2"
            />
          ))}
        </svg>
      )}
      {hover && (
        <div
          className="tip"
          style={{
            top: 10,
            left: px(hover.cycle),
            transform: `translateX(${px(hover.cycle) > width / 2 ? 'calc(-100% - 12px)' : '12px'})`,
          }}
        >
          <div className="tnum font-semibold">{metricLabel(metric)} {fmt(hover.metric)}</div>
          <div className="text-ink-2">Cycle {hover.cycle} · {pretty(hover.model)}</div>
          <div className="text-ink-2">{hover.n_train} training molecules{hover.accepted ? '' : ' · not accepted'}</div>
        </div>
      )}
    </div>
  );
}
