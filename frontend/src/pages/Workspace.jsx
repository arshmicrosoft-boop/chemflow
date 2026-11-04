import React, { useCallback, useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { api } from '../api.js';
import { Budget, Glass, Notice, Panel, Stat, Status, fmt, kindLabel, metricLabel, pretty } from '../components/ui.jsx';

const describeEvent = (e) => {
  const p = e.payload || {};
  return p.claim || p.statement || p.rationale || p.reason || p.objective || '';
};

export default function Workspace() {
  const [params, setParams] = useSearchParams();
  const selected = params.get('project');
  const [projects, setProjects] = useState(null);
  const [results, setResults] = useState(null);
  const [hypotheses, setHypotheses] = useState([]);
  const [timeline, setTimeline] = useState([]);
  const [newName, setNewName] = useState('');
  const [running, setRunning] = useState(false);
  const [lastRun, setLastRun] = useState(null);
  const [error, setError] = useState(null);

  const loadProjects = useCallback(() => {
    setError(null);
    return api.projects().then(setProjects).catch(e => setError(e.message));
  }, []);

  const loadProject = useCallback(async (pid) => {
    // Results read stored data, so they load even when the session cannot
    // resume; a failed side panel should not take the whole project down.
    const [r, h, t] = await Promise.allSettled([api.results(pid), api.hypotheses(pid), api.timeline(pid)]);
    if (r.status === 'rejected') throw r.reason;
    setResults(r.value);
    setHypotheses(h.value || []);
    setTimeline((t.value || []).filter(e => e.kind !== 'session_resumed').slice(-8).reverse());
    const failed = h.reason || t.reason;
    if (failed) setError(failed.message);
  }, []);

  useEffect(() => { loadProjects(); }, [loadProjects]);

  useEffect(() => {
    setResults(null);
    setLastRun(null);
    setError(null);
    if (selected) loadProject(selected).catch(e => setError(e.message));
  }, [selected, loadProject]);

  const select = (pid) => setParams({ project: pid });

  const create = async (e) => {
    e.preventDefault();
    if (!newName.trim()) return;
    try {
      const project = await api.createProject(newName.trim());
      setNewName('');
      await loadProjects();
      select(project.id);
    } catch (err) {
      setError(err.message);
    }
  };

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const { reason } = await api.run(selected);
      await loadProject(selected);
      setLastRun(reason);
    } catch (err) {
      setError(err.message);
    }
    setRunning(false);
  };

  const champion = results?.champion;
  const experiments = results ? [...results.experiments].reverse() : [];

  return (
    <main className="space-y-5">
      <header className="rise pt-2 pb-3">
        <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight">Research workspace</h1>
        <p className="text-ink-2 mt-2 max-w-2xl">Start a project and the agent proposes hypotheses, runs real DeepChem experiments in the sandbox, and only claims what the evidence supports.</p>
      </header>

      {error && <Notice tone="bad" title="ChemFlow hit a problem" action={<button className="btn" onClick={loadProjects}>Retry</button>}>{error}</Notice>}

      <Panel
        title="Projects"
        action={(
          <form onSubmit={create} className="flex gap-2 w-full sm:w-auto">
            <input aria-label="Project name" value={newName} onChange={e => setNewName(e.target.value)} placeholder="New project name" className="field flex-1 min-w-0 sm:w-56" />
            <button className="btn" disabled={!newName.trim()}>Create</button>
          </form>
        )}
      >
        {projects === null && !error && <div className="skeleton h-16" />}
        {projects?.length === 0 && <p className="text-sm text-ink-2">No projects yet. Name one above to begin.</p>}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {projects?.map(p => (
            <button
              key={p.id}
              onClick={() => select(p.id)}
              aria-pressed={String(p.id) === selected}
              className="plate p-4 text-left transition hover:-translate-y-0.5"
              style={String(p.id) === selected ? { outline: '2px solid var(--accent)', outlineOffset: -1 } : undefined}
            >
              <div className="font-semibold truncate">{p.name}</div>
              <div className="text-xs text-ink-2 truncate mt-0.5">{p.objective}</div>
            </button>
          ))}
        </div>
      </Panel>

      {selected && results && (
        <>
          <Glass className="p-5 sm:p-6 rise">
            <div className="flex flex-wrap items-start justify-between gap-4 mb-5">
              <div className="min-w-0">
                <div className="text-sm font-medium text-ink-2">{results.objective.name}</div>
                <h2 className="text-2xl font-semibold tracking-tight mt-0.5">{pretty(results.verdict.headline)}</h2>
                <p className="text-sm text-ink-2 mt-1">{lastRun ? `Last run stopped: ${lastRun}.` : pretty(results.verdict.detail)}</p>
              </div>
              <div className="flex flex-wrap gap-2">
                <button className="btn" onClick={run} disabled={running}>{running ? 'Running experiments…' : 'Run research'}</button>
                <Link className="btn btn-primary" to={`/projects/${selected}/results`}>View results</Link>
              </div>
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <Stat label="Research cycles" value={results.progress.cycles} />
              <Stat label="Experiments" value={results.budgets.experiments.used} note={`of ${results.budgets.experiments.total} allowed`} />
              <Stat label={`Best ${metricLabel(results.objective.metric)}`} value={fmt(champion?.metric)} note={champion ? pretty(champion.model) : 'No model yet'} />
              <Stat label="Validated findings" value={results.counts.findings.validated || 0} note={`of ${results.findings.length} assessed`} />
            </div>
          </Glass>

          <div className="grid lg:grid-cols-2 gap-5">
            <Panel title="Hypotheses" hint={`${hypotheses.length} generated`}>
              <ul className="scroll-area max-h-96 space-y-2 pr-1">
                {hypotheses.map(h => (
                  <li key={h.id} className="plate p-3 flex items-start justify-between gap-3">
                    <span className="text-sm min-w-0">{pretty(h.statement)}</span>
                    <Status value={h.status} />
                  </li>
                ))}
                {hypotheses.length === 0 && <li className="text-sm text-ink-2">The agent generates hypotheses on its first run.</li>}
              </ul>
            </Panel>

            <Panel title="Experiments" hint="Newest first">
              <ul className="scroll-area max-h-96 space-y-2 pr-1">
                {experiments.map(e => (
                  <li key={e.id} className="plate p-3 flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-sm font-medium">Cycle {e.cycle} · {kindLabel(e.kind)}</div>
                      <div className="text-xs text-ink-2 tnum truncate">{e.summary || '—'}</div>
                    </div>
                    <Status value={e.status} />
                  </li>
                ))}
                {experiments.length === 0 && <li className="text-sm text-ink-2">No experiments yet.</li>}
              </ul>
            </Panel>
          </div>

          <div className="grid lg:grid-cols-2 gap-5">
            <Panel title="Recent activity">
              <ul className="space-y-3">
                {timeline.map(e => (
                  <li key={e.id} className="text-sm">
                    <span className="font-medium capitalize">{e.kind.replaceAll('_', ' ')}</span>
                    <span className="text-ink-2"> · cycle {e.cycle}</span>
                    {describeEvent(e) && <div className="text-xs text-ink-2 truncate">{describeEvent(e)}</div>}
                  </li>
                ))}
                {timeline.length === 0 && <li className="text-sm text-ink-2">Nothing yet.</li>}
              </ul>
            </Panel>
            <Panel title="Sandbox budget" hint="Limits enforced on every experiment">
              <div className="space-y-4">
                <Budget label="Experiments" {...results.budgets.experiments} />
                <Budget label="Virtual assays" {...results.budgets.assays} />
                <Budget label="Hold-out evaluations" {...results.budgets.holdout} />
              </div>
            </Panel>
          </div>
        </>
      )}

      {selected && !results && !error && <div className="skeleton h-64" />}
      {!selected && projects?.length > 0 && (
        <Notice title="Select a project">Choose a project above to see its research and run more cycles.</Notice>
      )}

      <footer className="text-xs text-ink-2 pt-4">
        Results come from the ChemFlow engine (<code>chemflow/agent.py</code>, <code>sandbox.py</code>, <code>memory.py</code>). Hold-out labels stay locked.
      </footer>
    </main>
  );
}
