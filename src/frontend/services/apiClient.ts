/**
 * API Client for Congressional Trading Intelligence System
 * Advanced HTTP client with authentication, caching, and error handling
 */

import axios, { 
  AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError 
} from 'axios';
import { toast } from 'react-toastify';

// Types
export interface ApiResponse<T = any> {
  data: T;
  message?: string;
  pagination?: {
    page: number;
    per_page: number;
    total: number;
    pages: number;
    has_next: boolean;
    has_prev: boolean;
  };
}

export interface ApiError {
  error: string;
  details?: any;
  status?: number;
}

// Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api/v1';
const REQUEST_TIMEOUT = 30000; // 30 seconds
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

class ApiClient {
  private axiosInstance: AxiosInstance;
  private authToken: string | null = null;
  private refreshPromise: Promise<string> | null = null;

  constructor() {
    this.axiosInstance = axios.create({
      baseURL: API_BASE_URL,
      timeout: REQUEST_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
    this.loadAuthToken();
  }

  /**
   * Setup request and response interceptors
   */
  private setupInterceptors(): void {
    // Request interceptor
    this.axiosInstance.interceptors.request.use(
      (config) => {
        // Add auth token if available
        if (this.authToken) {
          config.headers.Authorization = `Bearer ${this.authToken}`;
        }

        // Add request timestamp for performance monitoring
        config.metadata = { startTime: new Date() };

        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.axiosInstance.interceptors.response.use(
      (response) => {
        // Calculate request duration
        const endTime = new Date();
        const duration = endTime.getTime() - response.config.metadata?.startTime?.getTime();
        
        // Log slow requests
        if (duration > 5000) {
          console.warn(`Slow API request: ${response.config.url} took ${duration}ms`);
        }

        return response;
      },
      async (error: AxiosError) => {
        const originalRequest = error.config as any;

        // Handle 401 Unauthorized - attempt token refresh
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            const newToken = await this.refreshToken();
            if (newToken) {
              originalRequest.headers.Authorization = `Bearer ${newToken}`;
              return this.axiosInstance(originalRequest);
            }
          } catch (refreshError) {
            // Refresh failed, redirect to login
            this.handleAuthenticationError();
            return Promise.reject(refreshError);
          }
        }

        // Handle rate limiting (429)
        if (error.response?.status === 429) {
          const retryAfter = error.response.headers['retry-after'];
          const delay = retryAfter ? parseInt(retryAfter) * 1000 : RETRY_DELAY;
          
          await this.delay(delay);
          return this.axiosInstance(originalRequest);
        }

        // Handle network errors with retry logic
        if (this.isRetryableError(error) && originalRequest.retryCount < MAX_RETRIES) {
          originalRequest.retryCount = (originalRequest.retryCount || 0) + 1;
          
          const delay = RETRY_DELAY * Math.pow(2, originalRequest.retryCount - 1); // Exponential backoff
          await this.delay(delay);
          
          return this.axiosInstance(originalRequest);
        }

        return Promise.reject(this.handleError(error));
      }
    );
  }

  /**
   * Load authentication token from storage
   */
  private loadAuthToken(): void {
    const token = localStorage.getItem('auth_token');
    if (token) {
      this.authToken = token;
    }
  }

  /**
   * Set authentication token
   */
  public setAuthToken(token: string): void {
    this.authToken = token;
    localStorage.setItem('auth_token', token);
  }

  /**
   * Clear authentication token
   */
  public clearAuthToken(): void {
    this.authToken = null;
    localStorage.removeItem('auth_token');
    localStorage.removeItem('refresh_token');
  }

  /**
   * Refresh authentication token
   */
  private async refreshToken(): Promise<string | null> {
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    const refreshToken = localStorage.getItem('refresh_token');
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    this.refreshPromise = this.performTokenRefresh(refreshToken);
    
    try {
      const newToken = await this.refreshPromise;
      this.refreshPromise = null;
      return newToken;
    } catch (error) {
      this.refreshPromise = null;
      throw error;
    }
  }

  /**
   * Perform the actual token refresh
   */
  private async performTokenRefresh(refreshToken: string): Promise<string> {
    try {
      const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {}, {
        headers: {
          Authorization: `Bearer ${refreshToken}`,
        },
      });

      const { access_token } = response.data;
      this.setAuthToken(access_token);
      
      return access_token;
    } catch (error) {
      this.clearAuthToken();
      throw error;
    }
  }

  /**
   * Handle authentication errors
   */
  private handleAuthenticationError(): void {
    this.clearAuthToken();
    
    // Redirect to login page
    if (typeof window !== 'undefined') {
      window.location.href = '/login';
    }
  }

  /**
   * Check if error is retryable
   */
  private isRetryableError(error: AxiosError): boolean {
    if (!error.response) {
      // Network error
      return true;
    }

    const status = error.response.status;
    return status >= 500 || status === 408 || status === 429;
  }

  /**
   * Handle API errors
   */
  private handleError(error: AxiosError): ApiError {
    if (error.response) {
      // Server responded with error status
      const status = error.response.status;
      const data = error.response.data as any;
      
      const apiError: ApiError = {
        error: data?.error || error.message,
        details: data?.details,
        status,
      };

      // Show user-friendly error messages
      if (status >= 500) {
        toast.error('Server error. Please try again later.');
      } else if (status === 404) {
        toast.error('Requested resource not found.');
      } else if (status === 403) {
        toast.error('Access denied. You don\'t have permission for this action.');
      } else if (status === 400) {
        toast.error(data?.error || 'Invalid request. Please check your input.');
      }

      return apiError;
    } else if (error.request) {
      // Network error
      const apiError: ApiError = {
        error: 'Network error. Please check your internet connection.',
      };
      
      toast.error('Network error. Please check your internet connection.');
      return apiError;
    } else {
      // Request setup error
      const apiError: ApiError = {
        error: error.message,
      };
      
      toast.error('Request failed. Please try again.');
      return apiError;
    }
  }

  /**
   * Delay helper for retries
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Generic GET request
   */
  public async get<T = any>(
    url: string, 
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    const response = await this.axiosInstance.get<ApiResponse<T>>(url, config);
    return response.data;
  }

  /**
   * Generic POST request
   */
  public async post<T = any>(
    url: string, 
    data?: any, 
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    const response = await this.axiosInstance.post<ApiResponse<T>>(url, data, config);
    return response.data;
  }

  /**
   * Generic PUT request
   */
  public async put<T = any>(
    url: string, 
    data?: any, 
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    const response = await this.axiosInstance.put<ApiResponse<T>>(url, data, config);
    return response.data;
  }

  /**
   * Generic PATCH request
   */
  public async patch<T = any>(
    url: string, 
    data?: any, 
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    const response = await this.axiosInstance.patch<ApiResponse<T>>(url, data, config);
    return response.data;
  }

  /**
   * Generic DELETE request
   */
  public async delete<T = any>(
    url: string, 
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    const response = await this.axiosInstance.delete<ApiResponse<T>>(url, config);
    return response.data;
  }

  /**
   * Upload file
   */
  public async uploadFile<T = any>(
    url: string,
    file: File,
    onUploadProgress?: (progressEvent: any) => void
  ): Promise<ApiResponse<T>> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.axiosInstance.post<ApiResponse<T>>(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress,
    });

    return response.data;
  }

  /**
   * Download file
   */
  public async downloadFile(
    url: string,
    filename?: string,
    config?: AxiosRequestConfig
  ): Promise<void> {
    const response = await this.axiosInstance.get(url, {
      ...config,
      responseType: 'blob',
    });

    // Create blob link to download
    const blob = new Blob([response.data]);
    const link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.download = filename || 'download';
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Clean up
    window.URL.revokeObjectURL(link.href);
  }

  /**
   * Authentication methods
   */
  public async login(username: string, password: string): Promise<ApiResponse<any>> {
    const response = await this.post('/auth/login', { username, password });
    
    if (response.data.access_token) {
      this.setAuthToken(response.data.access_token);
      
      if (response.data.refresh_token) {
        localStorage.setItem('refresh_token', response.data.refresh_token);
      }
    }
    
    return response;
  }

  /**
   * Logout
   */
  public async logout(): Promise<void> {
    try {
      await this.post('/auth/logout');
    } catch (error) {
      // Continue with logout even if API call fails
      console.warn('Logout API call failed:', error);
    } finally {
      this.clearAuthToken();
    }
  }

  /**
   * Get current user profile
   */
  public async getProfile(): Promise<ApiResponse<any>> {
    return this.get('/auth/profile');
  }

  /**
   * Health check
   */
  public async healthCheck(): Promise<boolean> {
    try {
      await this.get('/health');
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Get API information
   */
  public async getApiInfo(): Promise<ApiResponse<any>> {
    return this.get('/info');
  }
}

// Create and export singleton instance
export const apiClient = new ApiClient();

// Export types and utilities
export default apiClient;