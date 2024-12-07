import React from 'react';
import { useQuery } from 'react-query';
import {
  Grid,
  Paper,
  Typography,
  CircularProgress,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import Plot from 'react-plotly.js';
import axios from 'axios';

function ModelMetrics() {
  const { data: metrics, isLoading } = useQuery(
    'model-metrics',
    async () => {
      const response = await axios.get('http://localhost:8000/monitoring/models');
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
          Model Performance Metrics
        </Typography>
      </Grid>

      {/* Training Progress */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Training Progress
          </Typography>
          <Plot
            data={Object.entries(metrics.training_history).map(
              ([model_id, history]) => ({
                type: 'scatter',
                mode: 'lines+markers',
                name: model_id,
                x: history.map((h) => h.timestamp),
                y: history.map((h) => h.metrics.loss),
                line: { shape: 'spline' },
              })
            )}
            layout={{
              height: 400,
              margin: { t: 10 },
              xaxis: { title: 'Time' },
              yaxis: { title: 'Loss' },
              showlegend: true,
            }}
          />
        </Paper>
      </Grid>

      {/* Model Performance Comparison */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Model Performance Comparison
          </Typography>
          <Plot
            data={[
              {
                type: 'bar',
                x: Object.keys(metrics.model_metrics),
                y: Object.values(metrics.model_metrics).map(
                  (m) => m.accuracy[m.accuracy.length - 1].value
                ),
                marker: {
                  color: [
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
              xaxis: { title: 'Model' },
              yaxis: { title: 'Accuracy' },
            }}
          />
        </Paper>
      </Grid>

      {/* Prediction Statistics */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Prediction Statistics
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Model</TableCell>
                  <TableCell align="right">Count</TableCell>
                  <TableCell align="right">Avg Time (ms)</TableCell>
                  <TableCell align="right">Avg Confidence</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(metrics.prediction_stats).map(
                  ([model_id, stats]) => (
                    <TableRow key={model_id}>
                      <TableCell>{model_id}</TableCell>
                      <TableCell align="right">{stats.count}</TableCell>
                      <TableCell align="right">
                        {stats.avg_time.toFixed(2)}
                      </TableCell>
                      <TableCell align="right">
                        {stats.confidences.length > 0
                          ? (
                              stats.confidences.reduce((a, b) => a + b) /
                              stats.confidences.length
                            ).toFixed(2)
                          : 'N/A'}
                      </TableCell>
                    </TableRow>
                  )
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      </Grid>

      {/* Performance Over Time */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Performance Metrics Over Time
          </Typography>
          <Grid container spacing={2}>
            {Object.entries(metrics.model_metrics).map(([model_id, model]) => (
              <Grid item xs={12} md={6} key={model_id}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      {model_id}
                    </Typography>
                    <Plot
                      data={Object.entries(model).map(([metric, values]) => ({
                        type: 'scatter',
                        mode: 'lines',
                        name: metric,
                        x: values.map((v) => v.timestamp),
                        y: values.map((v) => v.value),
                        line: { shape: 'spline' },
                      }))}
                      layout={{
                        height: 300,
                        margin: { t: 10 },
                        showlegend: true,
                        legend: { orientation: 'h' },
                      }}
                    />
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      </Grid>

      {/* Resource Usage by Model */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Resource Usage by Model
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Plot
                data={[
                  {
                    type: 'bar',
                    name: 'Memory Usage',
                    x: Object.keys(metrics.model_metrics),
                    y: Object.values(metrics.resource_usage).map(
                      (r) => r.memory_mb
                    ),
                    marker: { color: '#2196f3' },
                  },
                ]}
                layout={{
                  height: 300,
                  margin: { t: 10 },
                  xaxis: { title: 'Model' },
                  yaxis: { title: 'Memory (MB)' },
                }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Plot
                data={[
                  {
                    type: 'bar',
                    name: 'CPU Usage',
                    x: Object.keys(metrics.model_metrics),
                    y: Object.values(metrics.resource_usage).map(
                      (r) => r.cpu_percent
                    ),
                    marker: { color: '#f50057' },
                  },
                ]}
                layout={{
                  height: 300,
                  margin: { t: 10 },
                  xaxis: { title: 'Model' },
                  yaxis: { title: 'CPU %' },
                }}
              />
            </Grid>
          </Grid>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default ModelMetrics;
