/**
 * OidcCallback.tsx
 *
 * Landing page for the OIDC redirect: /auth/callback
 * The gateway sends `#token=<access_token>` as a hash fragment so the token
 * is never sent to the server in a Referer header and is not recorded in
 * server access logs.
 *
 * This component:
 *  1. Reads the token from window.location.hash
 *  2. JWT-decodes the payload to extract userId and roles (no library needed)
 *  3. Stores them in the Zustand auth store (token in memory only)
 *  4. Replaces the hash in history to avoid leaking the token on back-navigation
 *  5. Redirects to /dashboard
 */
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '@store/authStore';

function decodeJwtPayload(token: string): Record<string, unknown> {
  try {
    const base64 = token.split('.')[1];
    if (!base64) return {};
    // atob is available in all modern browsers
    const json = atob(base64.replace(/-/g, '+').replace(/_/g, '/'));
    return JSON.parse(json) as Record<string, unknown>;
  } catch {
    return {};
  }
}

export default function OidcCallback() {
  const navigate = useNavigate();
  const { setAuth } = useAuthStore();

  useEffect(() => {
    const hash = window.location.hash; // e.g. "#token=eyJ..."
    const params = new URLSearchParams(hash.startsWith('#') ? hash.slice(1) : hash);
    const token = params.get('token');

    if (token) {
      const payload = decodeJwtPayload(token);
      const userId = (payload.sub as string) || null;
      const roles = Array.isArray(payload.roles) ? (payload.roles as string[]) : [];

      setAuth(token, userId, roles);

      // Remove the token from the browser history so it isn't visible after navigation
      window.history.replaceState(null, '', window.location.pathname);
    }

    navigate('/dashboard', { replace: true });
  }, [navigate, setAuth]);

  return (
    <div className="flex items-center justify-center min-h-screen">
      <p className="text-gray-500 text-sm">Signing you in…</p>
    </div>
  );
}
