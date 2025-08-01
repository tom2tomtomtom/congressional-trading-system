import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Divider,
  Paper,
  Chip,
  alpha
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  People as PeopleIcon,
  TrendingUp as TradingIcon,
  Analytics as AnalyticsIcon,
  AccountTree as NetworkIcon,
  NewspaperOutlined as IntelligenceIcon,
  Settings as SettingsIcon,
  Gavel as GavelIcon,
  School as SchoolIcon
} from '@mui/icons-material';

interface SidebarProps {
  onItemClick?: () => void;
}

interface NavigationItem {
  id: string;
  label: string;
  icon: React.ReactElement;
  path: string;
  description: string;
  badge?: string;
  isNew?: boolean;
}

const navigationItems: NavigationItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: <DashboardIcon />,
    path: '/dashboard',
    description: 'Overview and key metrics'
  },
  {
    id: 'members',
    label: 'Members',
    icon: <PeopleIcon />,
    path: '/members',
    description: 'Congressional member profiles'
  },
  {
    id: 'trading',
    label: 'Trading Analysis',
    icon: <TradingIcon />,
    path: '/trading',
    description: 'STOCK Act disclosures & analysis'
  },
  {
    id: 'analytics',
    label: 'Advanced Analytics',
    icon: <AnalyticsIcon />,
    path: '/analytics',
    description: 'ML models & predictions',
    badge: 'Phase 2',
    isNew: true
  },
  {
    id: 'network',
    label: 'Network Analysis',
    icon: <NetworkIcon />,
    path: '/network',
    description: 'Relationship mapping & graphs',
    badge: 'Phase 2',
    isNew: true
  },
  {
    id: 'intelligence',
    label: 'Intelligence & News',
    icon: <IntelligenceIcon />,
    path: '/intelligence',
    description: 'Real-time monitoring & sentiment',
    badge: 'Phase 2',
    isNew: true
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: <SettingsIcon />,
    path: '/settings',
    description: 'System configuration'
  }
];

const Sidebar: React.FC<SidebarProps> = ({ onItemClick }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleNavigation = (path: string) => {
    navigate(path);
    onItemClick?.();
  };

  const isActive = (path: string) => {
    return location.pathname === path || (path === '/dashboard' && location.pathname === '/');
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Logo/Header */}
      <Box sx={{ p: 3, pb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <GavelIcon sx={{ color: 'primary.main', mr: 1, fontSize: 28 }} />
          <Typography variant="h6" color="primary" sx={{ fontWeight: 600 }}>
            Congressional Trading
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
          Intelligence System v2.0
        </Typography>
        
        <Paper 
          elevation={0} 
          sx={{ 
            mt: 2, 
            p: 1.5, 
            backgroundColor: alpha('#1976d2', 0.1),
            border: '1px solid',
            borderColor: alpha('#1976d2', 0.2)
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
            <SchoolIcon sx={{ color: 'primary.main', mr: 1, fontSize: 16 }} />
            <Typography variant="caption" color="primary" sx={{ fontWeight: 600 }}>
              Educational Platform
            </Typography>
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem', lineHeight: 1.2 }}>
            For research & transparency purposes only
          </Typography>
        </Paper>
      </Box>

      <Divider />

      {/* Navigation Items */}
      <Box sx={{ flexGrow: 1, py: 1 }}>
        <List sx={{ px: 1 }}>
          {navigationItems.map((item) => (
            <ListItem key={item.id} disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton
                onClick={() => handleNavigation(item.path)}
                selected={isActive(item.path)}
                sx={{
                  borderRadius: 2,
                  py: 1.5,
                  px: 2,
                  '&.Mui-selected': {
                    backgroundColor: alpha('#1976d2', 0.12),
                    '&:hover': {
                      backgroundColor: alpha('#1976d2', 0.16),
                    },
                    '& .MuiListItemIcon-root': {
                      color: 'primary.main',
                    },
                    '& .MuiListItemText-primary': {
                      color: 'primary.main',
                      fontWeight: 600,
                    }
                  },
                  '&:hover': {
                    backgroundColor: alpha('#1976d2', 0.08),
                  }
                }}
              >
                <ListItemIcon sx={{ minWidth: 40 }}>
                  {item.icon}
                </ListItemIcon>
                
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body2" sx={{ fontWeight: isActive(item.path) ? 600 : 400 }}>
                        {item.label}
                      </Typography>
                      {item.badge && (
                        <Chip
                          label={item.badge}
                          size="small"
                          color={item.isNew ? "secondary" : "primary"}
                          variant={item.isNew ? "filled" : "outlined"}
                          sx={{ 
                            height: 18, 
                            fontSize: '0.6rem',
                            '& .MuiChip-label': { px: 0.8 }
                          }}
                        />
                      )}
                    </Box>
                  }
                  secondary={
                    <Typography 
                      variant="caption" 
                      color="text.secondary" 
                      sx={{ fontSize: '0.7rem', lineHeight: 1.2 }}
                    >
                      {item.description}
                    </Typography>
                  }
                />
              </ListItemButton>
            </ListItem>
          ))}
        </List>

        {/* Phase 2 Features Highlight */}
        <Box sx={{ px: 2, mt: 2 }}>
          <Paper 
            elevation={0} 
            sx={{ 
              p: 2, 
              backgroundColor: alpha('#dc004e', 0.1),
              border: '1px solid',
              borderColor: alpha('#dc004e', 0.2)
            }}
          >
            <Typography variant="caption" color="secondary" sx={{ fontWeight: 600, display: 'block', mb: 0.5 }}>
              🚀 Phase 2 Features
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem', lineHeight: 1.2 }}>
              Advanced ML analytics, network analysis, and real-time intelligence monitoring
            </Typography>
          </Paper>
        </Box>
      </Box>

      <Divider />

      {/* Footer */}
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary" align="center" sx={{ fontSize: '0.7rem' }}>
          Data updated: {new Date().toLocaleDateString()}
        </Typography>
        <Typography variant="caption" color="text.secondary" align="center" sx={{ fontSize: '0.7rem', display: 'block', mt: 0.5 }}>
          All data from official STOCK Act disclosures
        </Typography>
      </Box>
    </Box>
  );
};

export default Sidebar;