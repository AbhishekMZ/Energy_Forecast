import React from 'react';
import { useQuery } from 'react-query';
import {
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CircularProgress,
} from '@mui/material';
import Plot from 'react-plotly.js';
import api from '../api/axiosConfig';

function Dashboard() {
  const { data: performanceData, isLoading: loadingPerformance } = useQuery(
    'performance',
    async () => {
      const response = await api.get('/monitoring/performance');
      return response.data;
    }
  );

  const { data: errorMetrics, isLoading: loadingErrors } = useQuery(
    'error-metrics',
    async () => {
      const response = await api.get('/monitoring/errors');
      return response.data;
    }
  );

  if (loadingPerformance || loadingErrors) {
    return <CircularProgress />;
  }

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h4" gutterBottom>
          System Dashboard
        </Typography>
      </Grid>

      {/* Performance Metrics */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2, height: '100%' }}>
          <Typography variant="h6" gutterBottom>
            Performance Metrics
          </Typography>
          <Grid container spacing={2}>
            {performanceData?.performance_stats?.map((stat, index) => (
              <Grid item xs={6} key={index}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      {stat.name}
                    </Typography>
                    <Typography variant="h5">
                      {stat.value.toFixed(2)}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      </Grid>

      {/* Resource Usage */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2, height: '100%' }}>
          <Typography variant="h6" gutterBottom>
            Resource Usage
          </Typography>
          <Plot
            data={[
              {
                type: 'indicator',
                mode: 'gauge+number',
                value: performanceData?.resource_usage?.cpu_percent || 0,
                title: { text: 'CPU Usage %' },
                gauge: {
                  axis: { range: [null, 100] },
                  bar: { color: '#2196f3' },
                },
              },
            ]}
            layout={{
              width: 300,
              height: 250,
              margin: { t: 0, b: 0 },
            }}
          />
          <Plot
            data={[
              {
                type: 'indicator',
                mode: 'gauge+number',
                value: performanceData?.resource_usage?.memory_percent || 0,
                title: { text: 'Memory Usage %' },
                gauge: {
                  axis: { range: [null, 100] },
                  bar: { color: '#f50057' },
                },
              },
            ]}
            layout={{
              width: 300,
              height: 250,
              margin: { t: 0, b: 0 },
            }}
          />
        </Paper>
      </Grid>

      {/* Error Statistics */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Error Statistics
          </Typography>
          <Plot
            data={[
              {
                type: 'bar',
                x: errorMetrics?.map((stat) => stat.type) || [],
                y: errorMetrics?.map((stat) => stat.count) || [],
                marker: {
                  color: '#2196f3',
                },
              },
            ]}
            layout={{
              height: 300,
              margin: { t: 0 },
              xaxis: { title: 'Error Type' },
              yaxis: { title: 'Count' },
            }}
          />
        </Paper>
      </Grid>
    </Grid>
  );
}

export default Dashboard;
