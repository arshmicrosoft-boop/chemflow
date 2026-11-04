import React, { useEffect, useState } from 'react';
import { Link, NavLink, Route, Routes, matchPath, useLocation, useSearchParams } from 'react-router-dom';
import Workspace from './pages/Workspace.jsx';
import Results from './pages/Results.jsx';
import Topics from './pages/Topics.jsx';
import { ElectricLogo } from './components/flair.jsx';
import ChatBar from './components/ChatBar.jsx';
import PasswordGate from './components/PasswordGate.jsx';

function useTheme() {
  const [dark, setDark] = useState(() => {
    try {
      const saved = localStorage.getItem('chemflow-theme');
      if (saved) return saved === 'dark';
    } catch { /* storage may be blocked */ }
    return true;
  });
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    try { localStorage.setItem('chemflow-theme', dark ? 'dark' : 'light'); } catch { /* not essential */ }
  }, [dark]);
  return [dark, setDark];
}

function useLiquidGlass() {
  useEffect(() => {
    if (navigator.userAgentData?.brands.some(b => b.brand === 'Chromium')) {
      document.documentElement.dataset.refract = '';
    }
    const follow = (e) => {
      const el = e.target.closest?.('.glass');
      if (!el) return;
      const box = el.getBoundingClientRect();
      el.style.setProperty('--mx', `${e.clientX - box.left}px`);
      el.style.setProperty('--my', `${e.clientY - box.top}px`);
    };
    window.addEventListener('pointermove', follow, { passive: true });
    return () => window.removeEventListener('pointermove', follow);
  }, []);
}

function Nav({ dark, onTheme }) {
  const { pathname } = useLocation();
  const [params] = useSearchParams();
  const pid = matchPath('/projects/:id/results', pathname)?.params.id || params.get('project');
  const home = pid ? `/?project=${pid}` : '/';
  const link = ({ isActive }) => `px-2 sm:px-3 py-1.5 rounded-full text-sm font-semibold ${isActive ? 'text-ink bg-[var(--glass-strong)]' : 'text-ink-2 hover:text-ink'}`;

  return (
    <div className="sticky top-3 z-30 mb-8">
      <nav aria-label="Main" className="glass lens flex items-center justify-between gap-3 !rounded-full pl-3 sm:pl-5 pr-2 py-2">
        <Link to={home} aria-label="ChemFlow" className="logo-link inline-flex items-center gap-2 font-semibold tracking-tight text-lg shrink-0"><ElectricLogo /><span className="hidden sm:inline">ChemFlow</span></Link>
        <div className="flex items-center sm:gap-1">
          <NavLink to={home} end className={link}>Workspace</NavLink>
          {pid && <NavLink to={`/projects/${pid}/results`} className={link}>Results</NavLink>}
          <NavLink to="/topics" className={link}>Topics</NavLink>
          <button onClick={() => onTheme(!dark)} aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'} className="w-9 h-9 rounded-full text-ink-2 hover:text-ink">
            <span aria-hidden="true">{dark ? '☀' : '☽'}</span>
          </button>
        </div>
      </nav>
    </div>
  );
}

export default function App() {
  const [dark, setDark] = useTheme();
  useLiquidGlass();

  return (
    <>
      <div className="scene" aria-hidden="true"><i /><i /><i /><i /></div>
      <svg width="0" height="0" className="absolute" aria-hidden="true">
        <filter id="liquid-lens" x="0" y="0" width="100%" height="100%" colorInterpolationFilters="sRGB">
          <feTurbulence type="fractalNoise" baseFrequency="0.006 0.014" numOctaves="2" seed="7" result="noise" />
          <feGaussianBlur in="noise" stdDeviation="2" result="soft" />
          <feDisplacementMap in="SourceGraphic" in2="soft" scale="48" xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </svg>
      <PasswordGate>
      <div className="max-w-6xl mx-auto px-4 sm:px-6 pb-28">
        <Nav dark={dark} onTheme={setDark} />
        <Routes>
          <Route path="/" element={<Workspace />} />
          <Route path="/projects/:id/results" element={<Results />} />
          <Route path="/topics" element={<Topics />} />
          <Route path="/topics/:id" element={<Topics />} />
        </Routes>
        <p className="text-center text-sm text-ink-2 mt-10">Made by Arsh and Shivang</p>
      </div>
      <ChatBar />
      </PasswordGate>
    </>
  );
}
