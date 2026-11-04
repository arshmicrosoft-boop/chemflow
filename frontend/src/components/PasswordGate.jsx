import React, { useEffect, useState } from 'react';
import { PASSWORD_REQUIRED, api, savePassword } from '../api.js';
import { ElectricLogo } from './flair.jsx';
import { Glass } from './ui.jsx';

// Shows the app once the API accepts us. Any other failure lets the app load,
// so its own pages can explain the problem.
export default function PasswordGate({ children }) {
  const [state, setState] = useState('checking');
  const [password, setPassword] = useState('');
  const [wrong, setWrong] = useState(false);

  const check = (attempted) => api.projects()
    .then(() => setState('open'))
    .catch((e) => {
      setState(e.message === PASSWORD_REQUIRED ? 'locked' : 'open');
      setWrong(attempted && e.message === PASSWORD_REQUIRED);
    });

  useEffect(() => { check(false); }, []);

  const submit = (e) => {
    e.preventDefault();
    savePassword(password);
    check(true);
  };

  if (state === 'open') return children;

  return (
    <main className="min-h-screen flex items-center justify-center px-4">
      <Glass className="p-6 sm:p-8 w-full max-w-sm rise">
        <div className="flex items-center gap-2 text-lg font-semibold mb-4"><ElectricLogo />ChemFlow</div>
        {state === 'checking' && <p className="text-sm text-ink-2">Connecting. A sleeping server can take up to a minute to wake.</p>}
        {state === 'locked' && (
          <form onSubmit={submit} className="space-y-3">
            <label className="block">
              <span className="block text-sm font-medium mb-1">Password</span>
              <input autoFocus type="password" className="field w-full" value={password} onChange={e => setPassword(e.target.value)} />
            </label>
            {wrong && <p role="alert" className="text-sm" style={{ color: 'var(--bad)' }}>That password isn't right.</p>}
            <button className="btn btn-primary w-full" disabled={!password}>Open ChemFlow</button>
          </form>
        )}
      </Glass>
    </main>
  );
}
