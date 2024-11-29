import React from 'react';
import { useQuery } from 'react-query';
import {
  Grid,
  Paper,
  Typography,
  CircularProgress,
  Card,
  CardContent,
} from '@mui/material';
import Plot from 'react-plotly.js';
import axios from 'axios';

function SystemMetrics() {
  const { data: metrics, isLoading } = useQuery(
    'system-metrics',
    async () => {
      const response = await axios.get('http://localhost:8000/monitoring/system');
      return response.data;
    },
    {
      refetchInterval: 1000, // Update every second
    }
  );

  if (isLoading) {
    return <CircularProgress />;
  }

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h5" gutterBottom>
          System Metrics
        </Typography>
      </Grid>

      {/* CPU Usage */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            CPU Usage
          </Typography>
          <Plot
            data={[
              {
                type: 'indicator',
                mode: 'gauge+number',
                value: metrics.cpu.current,
                title: { text: 'Current CPU %' },
                gauge: {
                  axis: { range: [null, 100] },
                  bar: { color: '#2196f3' },
                  steps: [
                    { range: [0, 50], color: '#e3f2fd' },
                    { range: [50, 75], color: '#90caf9' },
                    { range: [75, 100], color: '#1565c0' },
                  ],
                },
              },
            ]}
            layout={{
              width: 400,
              height: 300,
              margin: { t: 25, r: 25, l: 25, b: 25 },
            }}
          />
          <Grid container spacing={2} sx={{ mt: 2 }}>
            <Grid item xs={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Min
                  </Typography>
                  <Typography variant="h6">
                    {metrics.cpu.min.toFixed(1)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Average
                  </Typography>
                  <Typography variant="h6">
                    {metrics.cpu.avg.toFixed(1)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Max
                  </Typography>
                  <Typography variant="h6">
                    {metrics.cpu.max.toFixed(1)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Paper>
      </Grid>

      {/* Memory Usage */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Memory Usage
          </Typography>
          <Plot
            data={[
              {
                type: 'indicator',
                mode: 'gauge+number',
                value: metrics.memory.current,
                title: { text: 'Current Memory %' },
                gauge: {
                  axis: { range: [null, 100] },
                  bar: { color: '#f50057' },
                  steps: [
                    { range: [0, 50], color: '#fce4ec' },
                    { range: [50, 75], color: '#f48fb1' },
                    { range: [75, 100], color: '#c51162' },
                  ],
                },
              },
            ]}
            layout={{
              width: 400,
              height: 300,
              margin: { t: 25, r: 25, l: 25, b: 25 },
            }}
          />
          <Grid container spacing={2} sx={{ mt: 2 }}>
            <Grid item xs={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Available
                  </Typography>
                  <Typography variant="h6">
                    {(metrics.memory.available / 1024 / 1024 / 1024).toFixed(1)} GB
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Used
                  </Typography>
                  <Typography variant="h6">
                    {(metrics.memory.used / 1024 / 1024 / 1024).toFixed(1)} GB
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Swap
                  </Typography>
                  <Typography variant="h6">
                    {(metrics.memory.swap_used / 1024 / 1024 / 1024).toFixed(1)} GB
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Paper>
      </Grid>

      {/* Disk Usage */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Disk Usage
          </Typography>
          <Plot
            data={[
              {
                type: 'pie',
                values: [
                  metrics.disk.used,
                  metrics.disk.free,
                ],
                labels: ['Used', 'Free'],
                hole: 0.4,
                marker: {
                  colors: ['#f50057', '#2196f3'],
                },
              },
            ]}
            layout={{
              width: 400,
              height: 300,
              margin: { t: 25, r: 25, l: 25, b: 25 },
              showlegend: true,
              legend: { orientation: 'h' },
            }}
          />
          <Grid container spacing={2} sx={{ mt: 2 }}>
            <Grid item xs={6}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Read Speed
                  </Typography>
                  <Typography variant="h6">
                    {(metrics.disk.read_bytes / 1024 / 1024).toFixed(1)} MB/s
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Write Speed
                  </Typography>
                  <Typography variant="h6">
                    {(metrics.disk.write_bytes / 1024 / 1024).toFixed(1)} MB/s
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Paper>
      </Grid>

      {/* Network Usage */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Network Usage
          </Typography>
          <Plot
            data={[
              {
                type: 'scatter',
                mode: 'lines',
                name: 'Sent',
                x: metrics.network_history.map((p) => p.timestamp),
                y: metrics.network_history.map((p) => p.bytes_sent / 1024 / 1024),
                line: { color: '#2196f3' },
              },
              {
                type: 'scatter',
                mode: 'lines',
                name: 'Received',
                x: metrics.network_history.map((p) => p.timestamp),
                y: metrics.network_history.map((p) => p.bytes_recv / 1024 / 1024),
                line: { color: '#f50057' },
              },
            ]}
            layout={{
              width: 400,
              height: 300,
              margin: { t: 25, r: 25, l: 25, b: 25 },
              showlegend: true,
              legend: { orientation: 'h' },
              xaxis: { title: 'Time' },
              yaxis: { title: 'MB/s' },
            }}
          />
          <Grid container spacing={2} sx={{ mt: 2 }}>
            <Grid item xs={6}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Packets Sent
                  </Typography>
                  <Typography variant="h6">
                    {metrics.network.packets_sent.toLocaleString()}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Packets Received
                  </Typography>
                  <Typography variant="h6">
                    {metrics.network.packets_recv.toLocaleString()}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default SystemMetrics;
