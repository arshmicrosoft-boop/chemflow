import React, { useEffect, useRef, useState } from 'react';
import { api } from '../api.js';

export default function ChatBar() {
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState('');
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const log = useRef(null);

  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight });
  }, [messages, busy, error, open]);

  const send = async (e) => {
    e.preventDefault();
    const question = text.trim();
    if (!question || busy) return;
    const next = [...messages, { role: 'user', content: question }];
    setMessages(next);
    setText('');
    setOpen(true);
    setBusy(true);
    setError(null);
    try {
      const { reply } = await api.chat(next);
      setMessages([...next, { role: 'assistant', content: reply }]);
    } catch (err) {
      setError(err.message);
    }
    setBusy(false);
  };

  return (
    <div className="chat-dock" onKeyDown={e => { if (e.key === 'Escape') setOpen(false); }}>
      {open && (
        <section className="glass chat-panel rise" aria-label="AI assistant">
          <div className="flex items-center justify-between gap-3 px-4 py-2 border-b" style={{ borderColor: 'var(--hairline)' }}>
            <span className="tech text-[11px] text-ink-2">AI assistant // Free models</span>
            <button className="text-ink-2 hover:text-ink w-8 h-8 rounded-full" aria-label="Close assistant" onClick={() => setOpen(false)}>
              <span aria-hidden="true">✕</span>
            </button>
          </div>
          <div ref={log} className="scroll-area flex-1 min-h-0 p-4 space-y-3" aria-live="polite">
            {messages.length === 0 && <p className="text-sm text-ink-2">Ask anything. Answers come from an AI model and can be wrong.</p>}
            {messages.map((m, i) => (
              <div key={i} className={`chat-bubble ${m.role === 'user' ? 'is-user' : ''}`}>{m.content}</div>
            ))}
            {busy && <div className="chat-bubble text-ink-2">Thinking…</div>}
            {error && <p role="alert" className="text-sm" style={{ color: 'var(--bad)' }}>{error}</p>}
          </div>
        </section>
      )}
      <form onSubmit={send} className="glass lens !rounded-full flex items-center gap-2 p-2">
        <input
          aria-label="Ask the AI assistant"
          className="chat-input"
          maxLength={2000}
          value={text}
          onChange={e => setText(e.target.value)}
          onFocus={() => setOpen(messages.length > 0)}
          placeholder="Ask anything…"
        />
        <button className="btn btn-primary" disabled={busy || !text.trim()}>Ask</button>
      </form>
    </div>
  );
}
