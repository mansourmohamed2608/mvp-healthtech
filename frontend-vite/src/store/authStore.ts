import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface AuthState {
  token: string | null;
  userId: string | null;
  roles: string[];
  setAuth: (token: string, userId: string | null, roles: string[]) => void;
  clearAuth: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      userId: null,
      roles: [],
      setAuth: (token, userId, roles) => set({ token, userId, roles }),
      clearAuth: () => set({ token: null, userId: null, roles: [] }),
    }),
    {
      name: 'healthtech-auth-storage',
      // Never persist the access token — keep it in memory only so it isn't
      // readable by third-party scripts via localStorage.  userId + roles are
      // safe to persist so the UI can restore the session identity on reload.
      partialize: (state) => ({ userId: state.userId, roles: state.roles }),
    }
  )
);
