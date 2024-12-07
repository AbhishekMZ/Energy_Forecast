import React from 'react';
import { useQuery } from 'react-query';
import {
  Grid,
  Paper,
  Typography,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import Plot from 'react-plotly.js';
import axios from 'axios';

function Performance() {
  const { data: performanceData, isLoading } = useQuery(
    'performance',
    async () => {
      const response = await axios.get('http://localhost:8000/performance');
      return response.data;
    },
    {
      refetchInterval: 5000, // Refresh every 5 seconds
    }
  );

  if (isLoading) {
    return <CircularProgress />;
  }

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h4" gutterBottom>
          System Performance
        </Typography>
      </Grid>

      {/* Resource Usage Over Time */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Resource Usage Over Time
          </Typography>
          <Plot
            data={[
              {
                type: 'scatter',
                mode: 'lines',
                name: 'CPU Usage',
                x: performanceData?.resource_history?.map((p) => p.timestamp),
                y: performanceData?.resource_history?.map((p) => p.cpu_percent),
                line: { color: '#2196f3' },
              },
              {
                type: 'scatter',
                mode: 'lines',
                name: 'Memory Usage',
                x: performanceData?.resource_history?.map((p) => p.timestamp),
                y: performanceData?.resource_history?.map(
                  (p) => p.memory_percent
                ),
                line: { color: '#f50057' },
              },
            ]}
            layout={{
              height: 400,
              margin: { t: 10 },
              xaxis: { title: 'Time' },
              yaxis: { title: 'Usage %' },
            }}
          />
        </Paper>
      </Grid>

      {/* Error Distribution */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Error Distribution
          </Typography>
          <Plot
            data={[
              {
                type: 'pie',
                labels: performanceData?.error_stats?.map((e) => e.type),
                values: performanceData?.error_stats?.map((e) => e.count),
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
            }}
          />
        </Paper>
      </Grid>

      {/* Performance Metrics */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Performance Metrics
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Metric</TableCell>
                  <TableCell align="right">Value</TableCell>
                  <TableCell align="right">Change</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {performanceData?.performance_stats?.map((stat) => (
                  <TableRow key={stat.name}>
                    <TableCell>{stat.name}</TableCell>
                    <TableCell align="right">
                      {typeof stat.value === 'number'
                        ? stat.value.toFixed(2)
                        : stat.value}
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{
                        color: stat.change > 0 ? '#4caf50' : '#f50057',
                      }}
                    >
                      {stat.change > 0 ? '+' : ''}
                      {stat.change.toFixed(2)}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      </Grid>

      {/* Memory Usage Details */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Memory Usage Details
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Component</TableCell>
                  <TableCell align="right">Used (MB)</TableCell>
                  <TableCell align="right">Total (MB)</TableCell>
                  <TableCell align="right">Usage %</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {performanceData?.memory_details?.map((detail) => (
                  <TableRow key={detail.component}>
                    <TableCell>{detail.component}</TableCell>
                    <TableCell align="right">
                      {(detail.used / 1024 / 1024).toFixed(2)}
                    </TableCell>
                    <TableCell align="right">
                      {(detail.total / 1024 / 1024).toFixed(2)}
                    </TableCell>
                    <TableCell align="right">
                      {((detail.used / detail.total) * 100).toFixed(1)}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default Performance;
