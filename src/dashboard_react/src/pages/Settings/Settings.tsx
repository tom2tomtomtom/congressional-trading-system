import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  TextField,
  FormControl,
  FormControlLabel,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  Button,
  Divider,
  Alert,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Paper,
  Slider
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Notifications,
  Security,
  Storage,
  Palette,
  Api,
  Delete,
  Edit,
  Add,
  Refresh,
  Download,
  Upload,
  Save
} from '@mui/icons-material';

import { useAppState } from '../../contexts/AppStateContext';

interface APIKey {
  id: string;
  name: string;
  service: string;
  status: 'active' | 'inactive' | 'expired';
  lastUsed: string;
  created: string;
}

const Settings: React.FC = () => {
  const { state, setTheme, setNotifications, setDefaultView } = useAppState();
  const [apiKeys, setApiKeys] = useState<APIKey[]>([
    {
      id: 'key_1',
      name: 'Congress.gov API',
      service: 'congress_gov',
      status: 'active',
      lastUsed: '2024-12-15T10:30:00Z',
      created: '2024-12-01T09:00:00Z'
    },
    {
      id: 'key_2',
      name: 'SEC EDGAR API',
      service: 'sec_edgar',
      status: 'active',
      lastUsed: '2024-12-15T08:15:00Z',
      created: '2024-12-01T09:00:00Z'
    },
    {
      id: 'key_3',
      name: 'Alpha Vantage',
      service: 'alpha_vantage',
      status: 'inactive',
      lastUsed: '2024-12-10T14:20:00Z',
      created: '2024-12-01T09:00:00Z'
    }
  ]);

  const [addKeyDialog, setAddKeyDialog] = useState(false);
  const [cacheRetention, setCacheRetention] = useState(30);
  const [dataRefreshInterval, setDataRefreshInterval] = useState(6);
  const [alertThreshold, setAlertThreshold] = useState(0.75);

  const handleSaveSettings = () => {
    // Save settings logic here
    console.log('Settings saved');
  };

  const handleExportData = () => {
    console.log('Exporting data...');
  };

  const handleImportData = () => {
    console.log('Importing data...');
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'inactive': return 'warning';
      case 'expired': return 'error';
      default: return 'default';
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" color="primary" gutterBottom>
          Settings
        </Typography>
        <Typography variant="body1" color="text.secondary">
          System configuration and preferences
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* User Preferences */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <Palette color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6" color="primary">
                  User Preferences
                </Typography>
              </Box>

              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Theme</InputLabel>
                    <Select
                      value={state.user.preferences.theme}
                      label="Theme"
                      onChange={(e) => setTheme(e.target.value as 'light' | 'dark')}
                    >
                      <MenuItem value="light">Light</MenuItem>
                      <MenuItem value="dark">Dark</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Default View</InputLabel>
                    <Select
                      value={state.user.preferences.defaultView}
                      label="Default View"
                      onChange={(e) => setDefaultView(e.target.value)}
                    >
                      <MenuItem value="dashboard">Dashboard</MenuItem>
                      <MenuItem value="members">Members</MenuItem>
                      <MenuItem value="trading">Trading Analysis</MenuItem>
                      <MenuItem value="analytics">Advanced Analytics</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={state.user.preferences.notifications}
                        onChange={(e) => setNotifications(e.target.checked)}
                      />
                    }
                    label="Enable Notifications"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* System Configuration */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <SettingsIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6" color="primary">
                  System Configuration
                </Typography>
              </Box>

              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Typography gutterBottom>
                    Cache Retention (days): {cacheRetention}
                  </Typography>
                  <Slider
                    value={cacheRetention}
                    onChange={(_, value) => setCacheRetention(value as number)}
                    min={1}
                    max={90}
                    step={1}
                    marks={[
                      { value: 1, label: '1' },
                      { value: 30, label: '30' },
                      { value: 60, label: '60' },
                      { value: 90, label: '90' }
                    ]}
                    valueLabelDisplay="auto"
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography gutterBottom>
                    Data Refresh Interval (hours): {dataRefreshInterval}
                  </Typography>
                  <Slider
                    value={dataRefreshInterval}
                    onChange={(_, value) => setDataRefreshInterval(value as number)}
                    min={1}
                    max={24}
                    step={1}
                    marks={[
                      { value: 1, label: '1h' },
                      { value: 6, label: '6h' },
                      { value: 12, label: '12h' },
                      { value: 24, label: '24h' }
                    ]}
                    valueLabelDisplay="auto"
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography gutterBottom>
                    Alert Threshold: {alertThreshold}
                  </Typography>
                  <Slider
                    value={alertThreshold}
                    onChange={(_, value) => setAlertThreshold(value as number)}
                    min={0.1}
                    max={1.0}
                    step={0.05}
                    marks={[
                      { value: 0.1, label: '0.1' },
                      { value: 0.5, label: '0.5' },
                      { value: 0.75, label: '0.75' },
                      { value: 1.0, label: '1.0' }
                    ]}
                    valueLabelDisplay="auto"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* API Keys Management */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Api color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h6" color="primary">
                    API Keys
                  </Typography>
                </Box>
                <Button
                  variant="outlined"
                  startIcon={<Add />}
                  onClick={() => setAddKeyDialog(true)}
                >
                  Add API Key
                </Button>
              </Box>

              <List>
                {apiKeys.map((key) => (
                  <ListItem key={key.id} divider>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body1" sx={{ fontWeight: 500 }}>
                            {key.name}
                          </Typography>
                          <Chip
                            label={key.status.toUpperCase()}
                            size="small"
                            color={getStatusColor(key.status) as any}
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="body2" color="text.secondary">
                            Service: {key.service}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Last used: {new Date(key.lastUsed).toLocaleDateString()} • 
                            Created: {new Date(key.created).toLocaleDateString()}
                          </Typography>
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <IconButton size="small" sx={{ mr: 1 }}>
                        <Edit fontSize="small" />
                      </IconButton>
                      <IconButton size="small" color="error">
                        <Delete fontSize="small" />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Data Management */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <Storage color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6" color="primary">
                  Data Management
                </Typography>
              </Box>

              <Alert severity="info" sx={{ mb: 3 }}>
                <Typography variant="body2">
                  All data is sourced from official STOCK Act disclosures and public records. 
                  No private or confidential information is stored.
                </Typography>
              </Alert>

              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Button
                    variant="outlined"
                    startIcon={<Download />}
                    fullWidth
                    onClick={handleExportData}
                  >
                    Export Data
                  </Button>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Button
                    variant="outlined"
                    startIcon={<Upload />}
                    fullWidth
                    onClick={handleImportData}
                  >
                    Import Data
                  </Button>
                </Grid>
                <Grid item xs={12}>
                  <Button
                    variant="outlined"
                    startIcon={<Refresh />}
                    fullWidth
                    color="warning"
                  >
                    Clear Cache
                  </Button>
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              <Typography variant="body2" color="text.secondary" gutterBottom>
                <strong>Storage Usage:</strong>
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Database: 2.3 GB
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Cache: 456 MB
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Logs: 89 MB
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Security & Privacy */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <Security color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6" color="primary">
                  Security & Privacy
                </Typography>
              </Box>

              <Alert severity="success" sx={{ mb: 3 }}>
                <Typography variant="body2">
                  <strong>🔒 Privacy Protected:</strong> This system processes only publicly 
                  available information and maintains strict data security standards.
                </Typography>
              </Alert>

              <List>
                <ListItem>
                  <ListItemText
                    primary="Data Encryption"
                    secondary="All data is encrypted at rest and in transit"
                  />
                  <ListItemSecondaryAction>
                    <Chip label="ENABLED" size="small" color="success" />
                  </ListItemSecondaryAction>
                </ListItem>
                
                <ListItem>
                  <ListItemText
                    primary="Access Logging"
                    secondary="All system access is logged and monitored"
                  />
                  <ListItemSecondaryAction>
                    <Chip label="ACTIVE" size="small" color="success" />
                  </ListItemSecondaryAction>
                </ListItem>
                
                <ListItem>
                  <ListItemText
                    primary="Data Retention"
                    secondary="Automatic cleanup of old data per retention policy"
                  />
                  <ListItemSecondaryAction>
                    <Chip label="30 DAYS" size="small" color="primary" />
                  </ListItemSecondaryAction>
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Save Settings */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, textAlign: 'center' }}>
            <Button
              variant="contained"
              size="large"
              startIcon={<Save />}
              onClick={handleSaveSettings}
              sx={{ mr: 2 }}
            >
              Save Settings
            </Button>
            <Button
              variant="outlined"
              size="large"
              onClick={() => window.location.reload()}
            >
              Reset to Defaults
            </Button>
          </Paper>
        </Grid>
      </Grid>

      {/* Add API Key Dialog */}
      <Dialog open={addKeyDialog} onClose={() => setAddKeyDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add New API Key</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Key Name"
                placeholder="e.g., Congress.gov API"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Service</InputLabel>
                <Select label="Service">
                  <MenuItem value="congress_gov">Congress.gov</MenuItem>
                  <MenuItem value="sec_edgar">SEC EDGAR</MenuItem>
                  <MenuItem value="alpha_vantage">Alpha Vantage</MenuItem>
                  <MenuItem value="news_api">News API</MenuItem>
                  <MenuItem value="custom">Custom</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="API Key"
                type="password"
                placeholder="Enter your API key"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddKeyDialog(false)}>Cancel</Button>
          <Button 
            variant="contained" 
            onClick={() => {
              setAddKeyDialog(false);
              // Add key logic here
            }}
          >
            Add Key
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Settings;