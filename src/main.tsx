// Prevent app crash on Vite HMR disconnects
window.addEventListener('unhandledrejection', (event) => {
  const message = event.reason?.message || String(event.reason);
  if (message.includes('WebSocket') || message.includes('closed without opened')) {
    event.preventDefault();
  }
});

window.addEventListener('error', (event) => {
  const message = event.message || '';
  if (message.includes('WebSocket') || message.includes('failed to connect to websocket')) {
    event.preventDefault();
  }
});

import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
