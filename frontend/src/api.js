const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const PASSWORD_KEY = 'chemflow-password';
export const PASSWORD_REQUIRED = 'Password required';

export function savePassword(password) {
  try { sessionStorage.setItem(PASSWORD_KEY, password); } catch { /* asked again next load */ }
}

function withPassword(options = {}) {
  let password = '';
  try { password = sessionStorage.getItem(PASSWORD_KEY) || ''; } catch { /* storage may be blocked */ }
  return { ...options, headers: { ...options.headers, 'X-App-Password': password } };
}

async function send(path, options) {
  const res = await fetch(`${API}${path}`, withPassword(options));
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res;
}

const request = (path, options) => send(path, options).then(res => res.json());

export const api = {
  projects: () => request('/projects'),
  project: (id) => request(`/projects/${id}`),
  createProject: (project) => request('/projects', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(project),
  }),
  chat: (messages) => request('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
  }),
  topicStatus: () => request('/topics/status'),
  topics: () => request('/topics'),
  topic: (id) => request(`/topics/${id}`),
  research: (topic) => request('/topics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic }),
  }),
  datasets: () => request('/datasets'),
  dataset: (name) => request(`/datasets/info?name=${encodeURIComponent(name)}`),
  uploadDataset: (file) => request(`/datasets?filename=${encodeURIComponent(file.name)}`, { method: 'POST', body: file }),
  results: (id) => request(`/projects/${id}/results`),
  hypotheses: (id) => request(`/projects/${id}/hypotheses`),
  timeline: (id) => request(`/projects/${id}/timeline`),
  run: (id, cycles = 5) => request(`/projects/${id}/run?cycles=${cycles}`, { method: 'POST' }),
  report: (id) => send(`/projects/${id}/report`).then(res => res.blob()),
};
