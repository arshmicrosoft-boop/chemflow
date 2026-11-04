import React, { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api.js';
import { LearningCurve, ModelLeaderboard, StructureEffects } from '../components/charts.jsx';
import { MicroStats, TechText, projectMicroStats } from '../components/flair.jsx';
import { Budget, Glass, Notice, Panel, Segmented, Stat, Status, fmt, kindLabel, metricLabel, pretty } from '../components/ui.jsx';

const VERDICTS = {
  validated: ['pill-good', '✓', 'Validated'],
  promising: ['pill-warn', '◐', 'Promising'],
  below_target: ['pill-bad', '✕', 'Below target'],
  not_started: ['pill-calm', '○', 'No results'],
};

const CHECKS = [
  ['meets_confidence', 'Meets target'],
  ['reproducible', 'Reproduced'],
  ['independent_validation', 'Hold-out'],
];

const FINDING_ORDER = ['validated', 'provisional', 'inconclusive', 'rejected'];

function Hero({ project, results }) {
  const { verdict, objective, champion, models, counts, progress } = results;
  const [tone, icon, label] = VERDICTS[verdict.state];
  const name = metricLabel(objective.metric);
  const op = objective.lower_is_better ? '≤' : '≥';
  const best = champion && models.find(m => m.model === champion.model && m.featurizer === champion.featurizer);

  return (
    <Glass className="p-6 sm:p-8 rise">
      <TechText text={`Results // ${project?.name}`} className="block text-xs text-ink-2 mb-3" />
      <div className="grid lg:grid-cols-[minmax(0,1fr)_auto] gap-6 items-start">
        <div className="min-w-0">
          <span className={`pill ${tone} mb-3`}><span aria-hidden="true">{icon}</span>{label}</span>
          <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">{pretty(verdict.headline)}</h1>
          <p className="text-ink-2 mt-2 max-w-2xl">{pretty(verdict.detail)}</p>
          {objective.question !== objective.name && <p className="text-sm text-ink-2 mt-3 max-w-2xl">{objective.question}</p>}
        </div>
        {champion && (
          <div className="plate p-5 lg:min-w-64">
            <div className="tech text-[11px] text-ink-2">Current best {name}</div>
            <div className="tnum text-5xl font-semibold tracking-tight mt-1">{fmt(champion.metric)}</div>
            <div className="tnum text-xs text-ink-2 mt-1">
              {Math.round(objective.confidence_level * 100)}% CI {fmt(champion.ci[0])} to {fmt(champion.ci[1])}
            </div>
            <div className="tnum text-xs text-ink-2 mt-1">Target {op} {objective.threshold} · {champion.n_train} molecules</div>
          </div>
        )}
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-6">
        <Stat label="Hold-out score" value={fmt(best?.holdout_metric)} note={best?.holdout_metric != null ? 'Locked data, never trained on' : 'Not evaluated yet'} />
        <Stat label="Validated findings" value={counts.findings.validated || 0} note={`${counts.findings.provisional || 0} provisional`} />
        <Stat label="Experiments" value={results.experiments.length} note={`${counts.experiments.failed || 0} failed, ${counts.experiments.blocked || 0} blocked`} />
        <Stat label="Research cycles" value={progress.cycles} note={progress.stop_reason ? `Stopped: ${progress.stop_reason}` : 'Not run yet'} />
      </div>
      <MicroStats className="mt-3" items={projectMicroStats(results)} />
    </Glass>
  );
}

function Approval({ approval }) {
  return (
    <div className="plate p-4">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
        <div className="font-medium text-sm">Lab experiment proposal</div>
        <Status value={approval.status} />
      </div>
      <p className="text-sm text-ink-2 mb-3">{pretty(approval.description)}</p>
      {approval.molecules.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-ink-2">
                <th className="font-medium py-1 pr-4">Molecule (SMILES)</th>
                <th className="font-medium py-1 pr-4 text-right">Predicted</th>
                <th className="font-medium py-1 text-right">Uncertainty</th>
              </tr>
            </thead>
            <tbody>
              {approval.molecules.map(m => (
                <tr key={m.smiles} className="border-t" style={{ borderColor: 'var(--hairline)' }}>
                  <td className="py-1.5 pr-4 font-mono text-xs break-all">{m.smiles}</td>
                  <td className="py-1.5 pr-4 text-right tnum">{fmt(m.predicted, 2)}</td>
                  <td className="py-1.5 text-right tnum">±{fmt(m.uncertainty, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {approval.status === 'pending' && (
        <p className="text-xs text-ink-2 mt-3">Decide from the command line: <code>python -m chemflow approve</code></p>
      )}
    </div>
  );
}

function Overview({ results }) {
  const { learning_curve: curve, objective, budgets, gaps, approvals, self_evaluation: selfEval, limitations } = results;
  const sortedGaps = [...gaps].sort((a, b) => b.severity - a.severity);
  return (
    <div className="grid lg:grid-cols-3 gap-5">
      <Panel
        className="lg:col-span-2"
        title="How the best model improved"
        hint={curve.length ? `Champion ${metricLabel(objective.metric)} as research progressed` : undefined}
      >
        {curve.length
          ? <LearningCurve curve={curve} objective={objective} />
          : <p className="text-sm text-ink-2">No model has been trained yet.</p>}
      </Panel>
      <Panel title="Sandbox budget">
        <div className="space-y-4">
          <Budget label="Experiments" {...budgets.experiments} />
          <Budget label="Virtual assays" {...budgets.assays} />
          <Budget label="Hold-out evaluations" {...budgets.holdout} />
        </div>
        {selfEval && (
          <p className="text-sm text-ink-2 mt-5">
            The agent predicted each outcome before running it, and missed by <strong className="tnum text-ink">{fmt(selfEval.mean_absolute_error)}</strong> {metricLabel(objective.metric)} on average over {selfEval.n} experiments.
          </p>
        )}
      </Panel>
      <Panel className="lg:col-span-3" title="What to do next" hint="Open questions and decisions waiting on you">
        <div className="space-y-2">
          {approvals.map(a => <Approval key={a.id} approval={a} />)}
          {sortedGaps.map(g => (
            <div key={g.kind + g.description} className="plate p-3 text-sm">
              <div className="font-medium capitalize">{g.kind.replaceAll('_', ' ')}</div>
              <div className="text-ink-2">{g.description}</div>
            </div>
          ))}
          {approvals.length === 0 && gaps.length === 0 && <p className="text-sm text-ink-2">No open questions recorded yet.</p>}
        </div>
      </Panel>
      <Panel className="lg:col-span-3" title="Limitations" hint="Read these before relying on the results">
        <ul className="grid md:grid-cols-2 gap-x-8 gap-y-2 text-sm text-ink-2 list-disc pl-5">
          {limitations.map(l => <li key={l}>{l}</li>)}
        </ul>
      </Panel>
    </div>
  );
}

function Models({ results }) {
  const { models, objective } = results;
  return (
    <Panel title="Model leaderboard" hint="Best tested configuration of each model and feature set">
      {models.length
        ? <ModelLeaderboard models={models} objective={objective} />
        : <p className="text-sm text-ink-2">No models have been evaluated yet.</p>}
    </Panel>
  );
}

function Chemistry({ results }) {
  const { structure_effects: effects, objective } = results;
  return (
    <Panel title="Structural features" hint={`Which substructures are associated with ${objective.target}. Associations, not causes.`}>
      {effects.length
        ? <StructureEffects effects={effects} />
        : <p className="text-sm text-ink-2">No structural hypotheses have been tested yet. The agent screens models first, so run more cycles.</p>}
    </Panel>
  );
}

function FindingCard({ finding }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="plate p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="font-semibold truncate first-letter:uppercase">{pretty(finding.title)}</h3>
          <div className="text-xs text-ink-2">{finding.kind === 'sar' ? 'Structure effect' : 'Model performance'} · cycle {finding.updated_cycle}</div>
        </div>
        <Status value={finding.status} />
      </div>
      <ul className="flex flex-wrap gap-x-4 gap-y-1 mt-3 text-xs">
        {CHECKS.map(([key, label]) => {
          const passed = finding.checks[key];
          return (
            <li key={key} className={passed === true ? 'text-ink' : 'text-ink-2'}>
              <span aria-hidden="true">{passed === true ? '✓' : passed === false ? '✕' : '–'}</span>{' '}
              {label}<span className="sr-only">: {passed === true ? 'passed' : passed === false ? 'failed' : 'not tested'}</span>
            </li>
          );
        })}
      </ul>
      <button className="text-xs font-semibold mt-3" style={{ color: 'var(--accent)' }} aria-expanded={open} onClick={() => setOpen(!open)}>
        {open ? 'Hide evidence' : 'Show evidence'}
      </button>
      {open && (
        <div className="text-sm text-ink-2 mt-2 space-y-1">
          <p>{pretty(finding.claim)}</p>
          {finding.reasons.length > 0 && <p>Why this status: {finding.reasons.join('; ')}.</p>}
          {finding.supporting_experiments.length > 0 && (
            <p className="font-mono text-xs">{finding.supporting_experiments.join(', ')}</p>
          )}
        </div>
      )}
    </li>
  );
}

function Findings({ results }) {
  const { findings, counts } = results;
  const available = FINDING_ORDER.filter(s => counts.findings[s]);
  const [filter, setFilter] = useState(available[0]);

  return (
    <Panel title="Findings" hint="Every claim, with the evidence checks behind it">
      {available.length === 0 && <p className="text-sm text-ink-2">No findings yet.</p>}
      {available.length > 0 && (
        <>
          <div className="mb-4">
            <Segmented
              label="Filter findings by status"
              value={filter}
              onChange={setFilter}
              options={available.map(s => ({ value: s, label: `${s[0].toUpperCase()}${s.slice(1)} ${counts.findings[s]}` }))}
            />
          </div>
          <ul className="grid md:grid-cols-2 gap-3">
            {findings.filter(f => f.status === filter).map(f => <FindingCard key={f.id} finding={f} />)}
          </ul>
        </>
      )}
    </Panel>
  );
}

function ExperimentLog({ results }) {
  const rows = [...results.experiments].reverse();
  return (
    <Panel title="Experiment log" hint="Every experiment, including failed and blocked ones">
      {rows.length === 0 && <p className="text-sm text-ink-2">No experiments yet.</p>}
      {rows.length > 0 && (
        <div className="plate overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-ink-2">
                <th className="font-medium p-3">Cycle</th>
                <th className="font-medium p-3">Experiment</th>
                <th className="font-medium p-3">Status</th>
                <th className="font-medium p-3">Result</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(e => (
                <tr key={e.id} className="border-t" style={{ borderColor: 'var(--hairline)' }}>
                  <td className="p-3 tnum">{e.cycle}</td>
                  <td className="p-3 whitespace-nowrap">{kindLabel(e.kind)}</td>
                  <td className="p-3"><Status value={e.status} /></td>
                  <td className="p-3 tnum text-ink-2 whitespace-nowrap">{e.summary || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

const TABS = [
  { value: 'overview', label: 'Overview', view: Overview },
  { value: 'models', label: 'Models', view: Models },
  { value: 'chemistry', label: 'Chemistry', view: Chemistry },
  { value: 'findings', label: 'Findings', view: Findings },
  { value: 'log', label: 'Experiment log', view: ExperimentLog },
];

export default function Results() {
  const { id } = useParams();
  const [project, setProject] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState('overview');
  const View = TABS.find(t => t.value === tab).view;

  useEffect(() => {
    setResults(null);
    setError(null);
    Promise.all([api.project(id), api.results(id)])
      .then(([p, r]) => { setProject(p); setResults(r); })
      .catch(e => setError(e.message));
  }, [id]);

  if (error) {
    return <Notice tone="bad" title="Could not load these results" action={<Link className="btn" to="/">Back to workspace</Link>}>{error}</Notice>;
  }
  if (!results) return <div className="skeleton h-96" />;

  return (
    <main className="space-y-5">
      <Hero project={project} results={results} />
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Segmented label="Result sections" options={TABS} value={tab} onChange={setTab} />
        <a className="btn" href={api.reportUrl(id)} target="_blank" rel="noreferrer">Full report</a>
      </div>
      <View key={tab} results={results} />
    </main>
  );
}
