import React, { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  useTheme,
  useMediaQuery,
  Divider,
  Alert,
  Collapse,
  AlertTitle
} from '@mui/material';
import {
  Menu as MenuIcon,
  Close as CloseIcon,
  Info as InfoIcon
} from '@mui/icons-material';

import Sidebar from './Sidebar';
import { useAppState } from '../../contexts/AppStateContext';

const drawerWidth = 280;

const Layout: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const location = useLocation();
  const { state } = useAppState();
  
  const [mobileOpen, setMobileOpen] = useState(false);
  const [showDisclaimer, setShowDisclaimer] = useState(true);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const getPageTitle = (pathname: string): string => {
    const routes: Record<string, string> = {
      '/': 'Dashboard',
      '/dashboard': 'Dashboard',
      '/members': 'Congressional Members',
      '/trading': 'Trading Analysis',
      '/analytics': 'Advanced Analytics',
      '/network': 'Network Analysis',
      '/intelligence': 'Intelligence & News',
      '/settings': 'Settings'
    };
    return routes[pathname] || 'Congressional Trading Intelligence';
  };

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Sidebar onItemClick={() => isMobile && setMobileOpen(false)} />
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
          backgroundColor: 'white',
          color: 'text.primary',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          borderBottom: '1px solid #e0e0e0'
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="h6" noWrap component="div" color="primary">
              {getPageTitle(location.pathname)}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
              Congressional Trading Intelligence System - Phase 2
            </Typography>
          </Box>

          {/* Global loading indicator */}
          {Object.values(state.loading).some(loading => loading) && (
            <Box sx={{ mr: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Loading...
              </Typography>
            </Box>
          )}

          <IconButton
            color="inherit"
            aria-label="system info"
            onClick={() => setShowDisclaimer(!showDisclaimer)}
          >
            <InfoIcon />
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Navigation Drawer */}
      <Box
        component="nav"
        sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
        aria-label="navigation menu"
      >
        {/* Mobile drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better mobile performance
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              backgroundColor: '#fafafa',
              borderRight: '1px solid #e0e0e0'
            },
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', p: 1 }}>
            <IconButton onClick={handleDrawerToggle}>
              <CloseIcon />
            </IconButton>
          </Box>
          <Divider />
          {drawer}
        </Drawer>

        {/* Desktop drawer */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              backgroundColor: '#fafafa',
              borderRight: '1px solid #e0e0e0'
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { md: `calc(100% - ${drawerWidth}px)` },
          minHeight: '100vh',
          backgroundColor: '#f5f5f5'
        }}
      >
        <Toolbar /> {/* Spacer for fixed app bar */}
        
        {/* Educational Disclaimer */}
        <Collapse in={showDisclaimer}>
          <Box sx={{ p: 2, pb: 0 }}>
            <Alert 
              severity="info" 
              onClose={() => setShowDisclaimer(false)}
              sx={{ 
                mb: 2,
                '& .MuiAlert-message': { width: '100%' }
              }}
            >
              <AlertTitle>🎓 Educational Research Platform</AlertTitle>
              <Typography variant="body2">
                This system analyzes publicly disclosed congressional trading data under the STOCK Act 
                for educational and research purposes only. All information is sourced from official 
                government records. <strong>This is not financial advice.</strong>
              </Typography>
            </Alert>
          </Box>
        </Collapse>

        {/* Error Display */}
        {Object.entries(state.errors).map(([key, error]) => (
          <Box key={key} sx={{ p: 2, pb: 0 }}>
            <Alert severity="error" sx={{ mb: 2 }}>
              <AlertTitle>Error in {key}</AlertTitle>
              {error}
            </Alert>
          </Box>
        ))}

        {/* Page Content */}
        <Box sx={{ p: 2 }}>
          <Outlet />
        </Box>
      </Box>
    </Box>
  );
};

export default Layout;