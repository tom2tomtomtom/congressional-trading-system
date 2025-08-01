import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import { AppState, DashboardFilters } from '../types';

// Initial state
const initialState: AppState = {
  user: {
    preferences: {
      theme: 'light',
      notifications: true,
      defaultView: 'dashboard'
    }
  },
  filters: {
    dateRange: {
      start: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0], // 1 year ago
      end: new Date().toISOString().split('T')[0] // today
    },
    members: [],
    parties: ['D', 'R', 'I'],
    chambers: ['House', 'Senate'],
    committees: [],
    symbols: [],
    transactionTypes: ['Purchase', 'Sale', 'Exchange'],
    amountRange: {
      min: 0,
      max: 10000000
    }
  },
  loading: {
    members: false,
    trades: false,
    analytics: false,
    network: false,
    intelligence: false
  },
  errors: {}
};

// Action types
type AppAction = 
  | { type: 'SET_LOADING'; payload: { key: keyof AppState['loading']; value: boolean } }
  | { type: 'SET_ERROR'; payload: { key: keyof AppState['errors']; value: string } }
  | { type: 'CLEAR_ERROR'; payload: { key: keyof AppState['errors'] } }
  | { type: 'UPDATE_FILTERS'; payload: Partial<DashboardFilters> }
  | { type: 'RESET_FILTERS' }
  | { type: 'SET_THEME'; payload: 'light' | 'dark' }
  | { type: 'SET_NOTIFICATIONS'; payload: boolean }
  | { type: 'SET_DEFAULT_VIEW'; payload: string };

// Reducer
const appStateReducer = (state: AppState, action: AppAction): AppState => {
  switch (action.type) {
    case 'SET_LOADING':
      return {
        ...state,
        loading: {
          ...state.loading,
          [action.payload.key]: action.payload.value
        }
      };

    case 'SET_ERROR':
      return {
        ...state,
        errors: {
          ...state.errors,
          [action.payload.key]: action.payload.value
        }
      };

    case 'CLEAR_ERROR':
      const { [action.payload.key]: _, ...remainingErrors } = state.errors;
      return {
        ...state,
        errors: remainingErrors
      };

    case 'UPDATE_FILTERS':
      return {
        ...state,
        filters: {
          ...state.filters,
          ...action.payload
        }
      };

    case 'RESET_FILTERS':
      return {
        ...state,
        filters: initialState.filters
      };

    case 'SET_THEME':
      return {
        ...state,
        user: {
          ...state.user,
          preferences: {
            ...state.user.preferences,
            theme: action.payload
          }
        }
      };

    case 'SET_NOTIFICATIONS':
      return {
        ...state,
        user: {
          ...state.user,
          preferences: {
            ...state.user.preferences,
            notifications: action.payload
          }
        }
      };

    case 'SET_DEFAULT_VIEW':
      return {
        ...state,
        user: {
          ...state.user,
          preferences: {
            ...state.user.preferences,
            defaultView: action.payload
          }
        }
      };

    default:
      return state;
  }
};

// Context
interface AppStateContextType {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
  
  // Convenience methods
  setLoading: (key: keyof AppState['loading'], value: boolean) => void;
  setError: (key: keyof AppState['errors'], value: string) => void;
  clearError: (key: keyof AppState['errors']) => void;
  updateFilters: (filters: Partial<DashboardFilters>) => void;
  resetFilters: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
  setNotifications: (enabled: boolean) => void;
  setDefaultView: (view: string) => void;
}

const AppStateContext = createContext<AppStateContextType | undefined>(undefined);

// Provider component
interface AppStateProviderProps {
  children: ReactNode;
}

export const AppStateProvider: React.FC<AppStateProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(appStateReducer, initialState);

  // Convenience methods
  const setLoading = (key: keyof AppState['loading'], value: boolean) => {
    dispatch({ type: 'SET_LOADING', payload: { key, value } });
  };

  const setError = (key: keyof AppState['errors'], value: string) => {
    dispatch({ type: 'SET_ERROR', payload: { key, value } });
  };

  const clearError = (key: keyof AppState['errors']) => {
    dispatch({ type: 'CLEAR_ERROR', payload: { key } });
  };

  const updateFilters = (filters: Partial<DashboardFilters>) => {
    dispatch({ type: 'UPDATE_FILTERS', payload: filters });
  };

  const resetFilters = () => {
    dispatch({ type: 'RESET_FILTERS' });
  };

  const setTheme = (theme: 'light' | 'dark') => {
    dispatch({ type: 'SET_THEME', payload: theme });
  };

  const setNotifications = (enabled: boolean) => {
    dispatch({ type: 'SET_NOTIFICATIONS', payload: enabled });
  };

  const setDefaultView = (view: string) => {
    dispatch({ type: 'SET_DEFAULT_VIEW', payload: view });
  };

  const contextValue: AppStateContextType = {
    state,
    dispatch,
    setLoading,
    setError,
    clearError,
    updateFilters,
    resetFilters,
    setTheme,
    setNotifications,
    setDefaultView
  };

  return (
    <AppStateContext.Provider value={contextValue}>
      {children}
    </AppStateContext.Provider>
  );
};

// Hook to use the context
export const useAppState = (): AppStateContextType => {
  const context = useContext(AppStateContext);
  if (context === undefined) {
    throw new Error('useAppState must be used within an AppStateProvider');
  }
  return context;
};

// Selectors for specific parts of state
export const useFilters = () => {
  const { state, updateFilters, resetFilters } = useAppState();
  return {
    filters: state.filters,
    updateFilters,
    resetFilters
  };
};

export const useLoading = () => {
  const { state, setLoading } = useAppState();
  return {
    loading: state.loading,
    setLoading
  };
};

export const useErrors = () => {
  const { state, setError, clearError } = useAppState();
  return {
    errors: state.errors,
    setError,
    clearError
  };
};

export const useUserPreferences = () => {
  const { state, setTheme, setNotifications, setDefaultView } = useAppState();
  return {
    preferences: state.user.preferences,
    setTheme,
    setNotifications,
    setDefaultView
  };
};