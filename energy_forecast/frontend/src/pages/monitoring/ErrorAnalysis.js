import React from 'react';
import { useQuery } from 'react-query';
import {
  Grid,
  Paper,
  Typography,
  CircularProgress,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  Divider,
} from '@mui/material';
import Plot from 'react-plotly.js';
import axios from 'axios';

function ErrorAnalysis() {
  const { data: errors, isLoading } = useQuery(
    'error-metrics',
    async () => {
      const response = await axios.get('http://localhost:8000/monitoring/errors');
      return response.data;
    },
    {
      refetchInterval: 5000, // Update every 5 seconds
    }
  );

  if (isLoading) {
    return <CircularProgress />;
  }

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h5" gutterBottom>
          Error Analysis
        </Typography>
      </Grid>

      {/* Error Summary */}
      <Grid item xs={12} md={4}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Error Summary
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Total Errors
                  </Typography>
                  <Typography variant="h4">
                    {errors.total_errors}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Error Types
                  </Typography>
                  <Typography variant="h4">
                    {errors.error_types}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Paper>
      </Grid>

      {/* Error Distribution */}
      <Grid item xs={12} md={8}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Error Distribution
          </Typography>
          <Plot
            data={[
              {
                type: 'pie',
                labels: errors.most_common.map(([type]) => type),
                values: errors.most_common.map(([, data]) => data.count),
                hole: 0.4,
                marker: {
                  colors: [
                    '#2196f3',
                    '#f50057',
                    '#ff9800',
                    '#4caf50',
                    '#9c27b0',
                  ],
                },
              },
            ]}
            layout={{
              height: 400,
              margin: { t: 10 },
              showlegend: true,
            }}
          />
        </Paper>
      </Grid>

      {/* Error Timeline */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Error Occurrence Timeline
          </Typography>
          <Plot
            data={Object.entries(errors.details).map(([type, data]) => ({
              type: 'scatter',
              mode: 'lines+markers',
              name: type,
              x: data.examples.map((e) => e.timestamp),
              y: data.examples.map((_, i) => i + 1),
              line: { shape: 'hv' },
            }))}
            layout={{
              height: 400,
              margin: { t: 10 },
              xaxis: { title: 'Time' },
              yaxis: { title: 'Cumulative Errors' },
              showlegend: true,
            }}
          />
        </Paper>
      </Grid>

      {/* Recent Errors */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Recent Errors
          </Typography>
          <Grid container spacing={2}>
            {Object.entries(errors.details).map(([type, data]) => (
              <Grid item xs={12} md={6} key={type}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1" color="primary" gutterBottom>
                      {type}
                    </Typography>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                      First seen: {new Date(data.first_seen).toLocaleString()}
                    </Typography>
                    <Divider sx={{ my: 1 }} />
                    <List dense>
                      {data.examples.map((example, index) => (
                        <ListItem key={index}>
                          <ListItemText
                            primary={example.message}
                            secondary={new Date(
                              example.timestamp
                            ).toLocaleString()}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      </Grid>

      {/* Error Patterns */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Error Patterns
          </Typography>
          <Grid container spacing={2}>
            {/* Time of Day Distribution */}
            <Grid item xs={12} md={6}>
              <Plot
                data={[
                  {
                    type: 'bar',
                    x: Array.from({ length: 24 }, (_, i) => i),
                    y: errors.time_distribution || Array(24).fill(0),
                    name: 'Errors by Hour',
                    marker: { color: '#2196f3' },
                  },
                ]}
                layout={{
                  height: 300,
                  margin: { t: 10 },
                  xaxis: { title: 'Hour of Day' },
                  yaxis: { title: 'Error Count' },
                }}
              />
            </Grid>
            {/* Error Duration */}
            <Grid item xs={12} md={6}>
              <Plot
                data={[
                  {
                    type: 'box',
                    y: errors.duration_stats || [],
                    name: 'Error Resolution Time',
                    marker: { color: '#f50057' },
                  },
                ]}
                layout={{
                  height: 300,
                  margin: { t: 10 },
                  yaxis: { title: 'Resolution Time (minutes)' },
                }}
              />
            </Grid>
          </Grid>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default ErrorAnalysis;
