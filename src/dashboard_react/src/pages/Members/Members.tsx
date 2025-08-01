import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Avatar,
  Chip,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  Alert
} from '@mui/material';
import { Search, FilterList } from '@mui/icons-material';

import { CongressionalMember } from '../../types';
import { useAppState } from '../../contexts/AppStateContext';

const Members: React.FC = () => {
  const { state, setLoading, setError, clearError } = useAppState();
  const [members, setMembers] = useState<CongressionalMember[]>([]);
  const [filteredMembers, setFilteredMembers] = useState<CongressionalMember[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [partyFilter, setPartyFilter] = useState('all');
  const [chamberFilter, setChamberFilter] = useState('all');

  // Generate mock members data
  const generateMockMembers = (): CongressionalMember[] => {
    const firstNames = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'Robert', 'Maria', 'James', 'Jennifer'];
    const lastNames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez'];
    const states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 'WA', 'AZ', 'MA'];
    const parties: Array<'D' | 'R' | 'I'> = ['D', 'R', 'I'];
    const chambers: Array<'House' | 'Senate'> = ['House', 'Senate'];
    
    return Array.from({ length: 50 }, (_, i) => ({
      bioguide_id: `M${i.toString().padStart(6, '0')}`,
      first_name: firstNames[Math.floor(Math.random() * firstNames.length)],
      last_name: lastNames[Math.floor(Math.random() * lastNames.length)],
      full_name: '',
      party: parties[Math.floor(Math.random() * parties.length)],
      state: states[Math.floor(Math.random() * states.length)],
      district: Math.random() > 0.2 ? Math.floor(Math.random() * 20) + 1 : undefined,
      chamber: chambers[Math.floor(Math.random() * chambers.length)],
      served_from: '2019-01-03',
      served_to: undefined,
      birth_date: `19${50 + Math.floor(Math.random() * 30)}-${String(Math.floor(Math.random() * 12) + 1).padStart(2, '0')}-${String(Math.floor(Math.random() * 28) + 1).padStart(2, '0')}`,
      leadership_position: Math.random() > 0.9 ? ['Speaker', 'Majority Leader', 'Minority Leader', 'Whip'][Math.floor(Math.random() * 4)] : undefined,
      net_worth_estimate: `$${Math.floor(Math.random() * 50)}M - $${Math.floor(Math.random() * 50) + 50}M`,
      education: [`${['Harvard', 'Yale', 'Stanford', 'MIT', 'Georgetown'][Math.floor(Math.random() * 5)]} University`],
      occupation: ['Lawyer', 'Business Owner', 'Teacher', 'Doctor', 'Military'][Math.floor(Math.random() * 5)],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    })).map(member => ({
      ...member,
      full_name: `${member.first_name} ${member.last_name}`
    }));
  };

  const loadMembers = async () => {
    setLoading('members', true);
    clearError('members');

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 800));
      
      const mockMembers = generateMockMembers();
      setMembers(mockMembers);
      setFilteredMembers(mockMembers);
    } catch (error) {
      setError('members', 'Failed to load congressional members');
    } finally {
      setLoading('members', false);
    }
  };

  const filterMembers = () => {
    let filtered = members;

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(member =>
        member.full_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        member.state.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Party filter
    if (partyFilter !== 'all') {
      filtered = filtered.filter(member => member.party === partyFilter);
    }

    // Chamber filter
    if (chamberFilter !== 'all') {
      filtered = filtered.filter(member => member.chamber === chamberFilter);
    }

    setFilteredMembers(filtered);
  };

  useEffect(() => {
    loadMembers();
  }, []);

  useEffect(() => {
    filterMembers();
  }, [searchTerm, partyFilter, chamberFilter, members]);

  const getPartyColor = (party: string) => {
    switch (party) {
      case 'D': return '#1976d2';
      case 'R': return '#d32f2f';
      case 'I': return '#388e3c';
      default: return '#757575';
    }
  };

  const getPartyLabel = (party: string) => {
    switch (party) {
      case 'D': return 'Democrat';
      case 'R': return 'Republican';
      case 'I': return 'Independent';
      default: return party;
    }
  };

  const getInitials = (name: string) => {
    return name.split(' ').map(n => n[0]).join('').toUpperCase();
  };

  if (state.loading.members && members.length === 0) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
          Loading congressional members...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" color="primary" gutterBottom>
          Congressional Members
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Profiles and information for all 535 members of Congress
        </Typography>
      </Box>

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                placeholder="Search members..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />
                }}
              />
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Party</InputLabel>
                <Select
                  value={partyFilter}
                  label="Party"
                  onChange={(e) => setPartyFilter(e.target.value)}
                >
                  <MenuItem value="all">All Parties</MenuItem>
                  <MenuItem value="D">Democrat</MenuItem>
                  <MenuItem value="R">Republican</MenuItem>
                  <MenuItem value="I">Independent</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Chamber</InputLabel>
                <Select
                  value={chamberFilter}
                  label="Chamber"
                  onChange={(e) => setChamberFilter(e.target.value)}
                >
                  <MenuItem value="all">Both Chambers</MenuItem>
                  <MenuItem value="House">House</MenuItem>
                  <MenuItem value="Senate">Senate</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={2}>
              <Button
                variant="outlined"
                startIcon={<FilterList />}
                onClick={() => {
                  setSearchTerm('');
                  setPartyFilter('all');
                  setChamberFilter('all');
                }}
                fullWidth
              >
                Clear
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Results Summary */}
      <Alert severity="info" sx={{ mb: 2 }}>
        Showing {filteredMembers.length} of {members.length} congressional members
      </Alert>

      {/* Members Grid */}
      <Grid container spacing={2}>
        {filteredMembers.map((member) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={member.bioguide_id}>
            <Card 
              sx={{ 
                height: '100%',
                transition: 'transform 0.2s, box-shadow 0.2s',
                '&:hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: 4
                }
              }}
            >
              <CardContent>
                {/* Avatar and Basic Info */}
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar
                    sx={{
                      width: 48,
                      height: 48,
                      bgcolor: getPartyColor(member.party),
                      mr: 2
                    }}
                  >
                    {getInitials(member.full_name)}
                  </Avatar>
                  
                  <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                    <Typography variant="h6" noWrap>
                      {member.full_name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {member.state}{member.district ? `-${member.district}` : ''}
                    </Typography>
                  </Box>
                </Box>

                {/* Party and Chamber */}
                <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                  <Chip
                    label={getPartyLabel(member.party)}
                    size="small"
                    sx={{
                      bgcolor: getPartyColor(member.party),
                      color: 'white',
                      '& .MuiChip-label': { fontWeight: 600 }
                    }}
                  />
                  <Chip
                    label={member.chamber}
                    size="small"
                    variant="outlined"
                  />
                </Box>

                {/* Leadership Position */}
                {member.leadership_position && (
                  <Box sx={{ mb: 2 }}>
                    <Chip
                      label={member.leadership_position}
                      size="small"
                      color="secondary"
                      variant="outlined"
                    />
                  </Box>
                )}

                {/* Additional Info */}
                <Box sx={{ mb: 2 }}>
                  {member.occupation && (
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      <strong>Background:</strong> {member.occupation}
                    </Typography>
                  )}
                  
                  {member.net_worth_estimate && (
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      <strong>Est. Net Worth:</strong> {member.net_worth_estimate}
                    </Typography>
                  )}
                  
                  {member.education && member.education.length > 0 && (
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      <strong>Education:</strong> {member.education[0]}
                    </Typography>
                  )}
                </Box>

                {/* Action Button */}
                <Button
                  variant="outlined"
                  size="small"
                  fullWidth
                  onClick={() => {
                    // Would navigate to member detail page
                    console.log('View member:', member.bioguide_id);
                  }}
                >
                  View Profile
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* No Results */}
      {filteredMembers.length === 0 && !state.loading.members && (
        <Card sx={{ mt: 2 }}>
          <CardContent sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No members found
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Try adjusting your search criteria or filters
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default Members;