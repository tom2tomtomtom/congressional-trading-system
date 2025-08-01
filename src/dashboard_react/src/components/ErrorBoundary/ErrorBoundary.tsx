import React, { Component, ErrorInfo, ReactNode } from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  Paper, 
  Container,
  Alert,
  AlertTitle 
} from '@mui/material';
import { ErrorOutline, Refresh } from '@mui/icons-material';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Congressional Trading System Error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo
    });

    // Log error to monitoring service in production
    if (process.env.NODE_ENV === 'production') {
      // Log to error tracking service
      this.logErrorToService(error, errorInfo);
    }
  }

  private logErrorToService = (error: Error, errorInfo: ErrorInfo) => {
    // In production, integrate with error tracking service like Sentry
    console.error('Error logged to monitoring service:', {
      error: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    });
  };

  private handleReload = () => {
    window.location.reload();
  };

  private handleReset = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  public render() {
    if (this.state.hasError) {
      return (
        <Container maxWidth="md" sx={{ py: 4 }}>
          <Paper 
            elevation={3} 
            sx={{ 
              p: 4, 
              textAlign: 'center',
              background: 'linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%)'
            }}
          >
            <Box sx={{ mb: 3 }}>
              <ErrorOutline 
                sx={{ 
                  fontSize: 80, 
                  color: 'error.main',
                  mb: 2 
                }} 
              />
              
              <Typography variant="h4" component="h1" gutterBottom color="error">
                Congressional Trading System Error
              </Typography>
              
              <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
                We apologize for the inconvenience. An unexpected error has occurred.
              </Typography>
            </Box>

            <Alert severity="error" sx={{ mb: 3, textAlign: 'left' }}>
              <AlertTitle>Error Details</AlertTitle>
              <Typography variant="body2" component="div">
                <strong>Error:</strong> {this.state.error?.message || 'Unknown error occurred'}
              </Typography>
              
              {process.env.NODE_ENV === 'development' && this.state.error?.stack && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" component="div">
                    <strong>Stack Trace:</strong>
                  </Typography>
                  <Box 
                    component="pre" 
                    sx={{ 
                      fontSize: '0.75rem',
                      background: '#f5f5f5',
                      p: 1,
                      borderRadius: 1,
                      overflow: 'auto',
                      maxHeight: 200,
                      mt: 1
                    }}
                  >
                    {this.state.error.stack}
                  </Box>
                </Box>
              )}
            </Alert>

            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
              <Button
                variant="contained"
                color="primary"
                startIcon={<Refresh />}
                onClick={this.handleReload}
                size="large"
              >
                Reload Application
              </Button>
              
              <Button
                variant="outlined"
                color="primary"
                onClick={this.handleReset}
                size="large"
              >
                Try Again
              </Button>
            </Box>

            <Box sx={{ mt: 4, pt: 3, borderTop: '1px solid #e0e0e0' }}>
              <Typography variant="body2" color="text.secondary">
                If this error persists, please contact our support team with the error details above.
              </Typography>
              
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                <strong>Error ID:</strong> {Date.now().toString(36)}
              </Typography>
              
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                <strong>Time:</strong> {new Date().toLocaleString()}
              </Typography>

              {process.env.NODE_ENV === 'development' && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    <strong>Development Mode:</strong> Additional error information is available in the browser console.
                  </Typography>
                </Box>
              )}
            </Box>

            {/* Educational disclaimer even in error state */}
            <Alert severity="info" sx={{ mt: 3, textAlign: 'left' }}>
              <Typography variant="body2">
                <strong>🎓 Educational System:</strong> This Congressional Trading Intelligence System 
                is designed for educational and research purposes only. All data is from publicly 
                disclosed sources under the STOCK Act.
              </Typography>
            </Alert>
          </Paper>
        </Container>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;