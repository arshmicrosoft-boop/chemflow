import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { api } from '../api.js';
import { TechText } from '../components/flair.jsx';
import { Glass, Notice, Panel } from '../components/ui.jsx';

function inline(text, sources) {
  return text.split(/(\*\*[^*]+\*\*|\[\d+\])/).map((part, i) => {
    if (part.startsWith('**')) return <strong key={i}>{part.slice(2, -2)}</strong>;
    const source = /^\[(\d+)\]$/.exec(part)?.[1];
    if (source && sources[source - 1]) {
      return <sup key={i}><a href={`#source-${source}`} title={sources[source - 1].title} className="font-semibold px-0.5" style={{ color: 'var(--accent)' }}>{source}</a></sup>;
    }
    return part;
  });
}

function Report({ text, sources }) {
  const blocks = [];
  text.split('\n').forEach((line) => {
    const item = /^\s*[-*] (.*)/.exec(line)?.[1];
    if (item) {
      const last = blocks[blocks.length - 1];
      if (last?.type === 'list') last.items.push(item);
      else blocks.push({ type: 'list', items: [item] });
    } else if (line.startsWith('#')) {
      blocks.push({ type: 'heading', text: line.replace(/^#+\s*/, '') });
    } else if (line.trim()) {
      blocks.push({ type: 'text', text: line });
    }
  });
  return blocks.map((b, i) => {
    if (b.type === 'heading') return <h2 key={i} className="text-lg font-semibold tracking-tight mt-6 first:mt-0 mb-2">{b.text}</h2>;
    if (b.type === 'list') {
      return (
        <ul key={i} className="list-disc pl-5 space-y-2 text-sm leading-relaxed">
          {b.items.map((item, j) => <li key={j}>{inline(item, sources)}</li>)}
        </ul>
      );
    }
    return <p key={i} className="text-sm leading-relaxed mb-2">{inline(b.text, sources)}</p>;
  });
}

function SetupSteps({ onCheck }) {
  return (
    <Panel title="Add your API key to start" hint="Topic research uses an AI model, which needs a key of your own.">
      <ol className="list-decimal pl-5 space-y-2 text-sm text-ink-2">
        <li>Create a key at <a className="font-semibold" style={{ color: 'var(--accent)' }} href="https://openrouter.ai/keys" target="_blank" rel="noreferrer">OpenRouter</a> or <a className="font-semibold" style={{ color: 'var(--accent)' }} href="https://console.anthropic.com/settings/keys" target="_blank" rel="noreferrer">Anthropic</a>. Using it costs money, so add credits there first.</li>
        <li>Open <code>backend/.env</code> in the project folder and paste the key after <code>OPENROUTER_API_KEY=</code> or <code>ANTHROPIC_API_KEY=</code>, then save.</li>
        <li>Never paste the key into a chat or commit it. That file stays on your computer.</li>
      </ol>
      <button className="btn mt-4" onClick={onCheck}>I've added it</button>
    </Panel>
  );
}

export default function Topics() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [report, setReport] = useState(null);
  const [topic, setTopic] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const refresh = () => {
    setError(null);
    api.topicStatus().then(setStatus).catch(e => setError(e.message));
    api.topics().then(setHistory).catch(e => setError(e.message));
  };

  useEffect(refresh, []);

  useEffect(() => {
    setReport(null);
    if (id) api.topic(id).then(setReport).catch(e => setError(e.message));
  }, [id]);

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const created = await api.research(topic.trim());
      setTopic('');
      setHistory(h => [created, ...h]);
      navigate(`/topics/${created.id}`);
    } catch (err) {
      setError(err.message);
    }
    setBusy(false);
  };

  return (
    <main className="space-y-5">
      <header className="rise pt-2 pb-3">
        <TechText text="ChemFlow // AI web research" className="text-xs text-ink-2" />
        <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight mt-1">Topic research</h1>
        <p className="text-ink-2 mt-2 max-w-2xl">Give the AI a topic and it finds sources and writes a report that cites them. For testing models on your own measurements, use the <Link to="/" className="font-semibold" style={{ color: 'var(--accent)' }}>workspace</Link>.</p>
      </header>

      {error && <Notice tone="bad" title="ChemFlow hit a problem">{error}</Notice>}
      {status && !status.configured && <SetupSteps onCheck={refresh} />}

      {status?.configured && (
        <Glass className="p-5 sm:p-6 rise">
          <form onSubmit={submit} className="flex flex-wrap gap-2">
            <input aria-label="Research topic" className="field flex-1 min-w-0" maxLength={500} value={topic} onChange={e => setTopic(e.target.value)} placeholder="What limits solid-state lithium battery electrolytes?" disabled={busy} />
            <button className="btn btn-primary" disabled={busy || !topic.trim()}>{busy ? 'Researching…' : 'Research'}</button>
          </form>
          <p className="text-xs text-ink-2 mt-2">{busy ? 'This can take a few minutes. Keep this page open.' : status.provider === 'openrouter' ? 'Uses free models and free paper databases, so reports cost nothing. Free models have daily limits.' : 'Each report uses your API credits.'}</p>
        </Glass>
      )}

      {report && (
        <Panel title={report.topic} hint={`${report.searches ? `${report.searches} web searches · ` : ''}${report.sources.length} sources · ${report.model} · ${report.created_at} UTC`}>
          {report.sources.length === 0 && <p className="text-sm mb-3" style={{ color: 'var(--warn)' }}>This report cites no sources, so treat it as unverified.</p>}
          {report.search_errors.length > 0 && <p className="text-sm mb-3" style={{ color: 'var(--warn)' }}>Some web searches failed, so this report may be incomplete.</p>}
          <div className="plate p-4 sm:p-5">
            <Report text={report.report} sources={report.sources} />
          </div>
          {report.sources.length > 0 && (
            <>
              <h2 className="text-lg font-semibold tracking-tight mt-5 mb-2">Sources</h2>
              <ol className="list-decimal pl-5 space-y-1 text-sm">
                {report.sources.map((s, i) => (
                  <li key={s.url} id={`source-${i + 1}`}><a href={s.url} target="_blank" rel="noreferrer noopener" className="break-words" style={{ color: 'var(--accent)' }}>{s.title}</a></li>
                ))}
              </ol>
            </>
          )}
          <p className="text-xs text-ink-2 mt-4">Written by AI from web sources and not checked by ChemFlow's sandbox. Check the sources before relying on it.</p>
        </Panel>
      )}

      {history.length > 0 && (
        <Panel title="Past reports">
          <ul className="space-y-2">
            {history.map(h => (
              <li key={h.id}>
                <Link to={`/topics/${h.id}`} className="plate p-3 block text-sm font-medium truncate">{h.topic}</Link>
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </main>
  );
}
