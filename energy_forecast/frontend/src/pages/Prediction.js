import React, { useState } from 'react';
import { useQuery, useMutation } from 'react-query';
import {
  Grid,
  Paper,
  Typography,
  Button,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import Plot from 'react-plotly.js';

function Prediction() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [data, setData] = useState(null);

  // File dropzone
  const { getRootProps, getInputProps } = useDropzone({
    accept: {
      'text/csv': ['.csv'],
    },
    onDrop: async (acceptedFiles) => {
      const file = acceptedFiles[0];
      setFile(file);

      // Read file
      const reader = new FileReader();
      reader.onload = async (e) => {
        const text = e.target.result;
        const rows = text.split('\n').map((row) => row.split(','));
        const headers = rows[0];
        const data = rows.slice(1).map((row) =>
          headers.reduce((obj, header, i) => {
            obj[header] = row[i];
            return obj;
          }, {})
        );
        setData(data);
      };
      reader.readAsText(file);
    },
  });

  // Prediction mutation
  const predictMutation = useMutation(async () => {
    const response = await axios.post(
      'http://localhost:8000/predict/latest',
      {
        data: data,
      }
    );
    return response.data;
  });

  const handlePredict = async () => {
    const result = await predictMutation.mutateAsync();
    setPredictions(result.predictions);
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h4" gutterBottom>
          Energy Consumption Prediction
        </Typography>
      </Grid>

      {/* File Upload */}
      <Grid item xs={12}>
        <Paper
          {...getRootProps()}
          sx={{
            p: 2,
            cursor: 'pointer',
            backgroundColor: (theme) =>
              theme.palette.mode === 'dark' ? '#1A2027' : '#fff',
          }}
        >
          <input {...getInputProps()} />
          <Typography variant="body1" align="center">
            {file
              ? `Selected file: ${file.name}`
              : 'Drag and drop a CSV file here, or click to select file'}
          </Typography>
        </Paper>
      </Grid>

      {/* Prediction Controls */}
      <Grid item xs={12}>
        <Button
          variant="contained"
          onClick={handlePredict}
          disabled={!file || predictMutation.isLoading}
          fullWidth
        >
          {predictMutation.isLoading ? (
            <CircularProgress size={24} />
          ) : (
            'Make Predictions'
          )}
        </Button>
      </Grid>

      {/* Prediction Results */}
      {predictions && (
        <>
          {/* Results Table */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Prediction Results
              </Typography>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Time</TableCell>
                      <TableCell align="right">Predicted Consumption</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {predictions.map((pred, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          {data[index]?.timestamp || `Point ${index + 1}`}
                        </TableCell>
                        <TableCell align="right">
                          {typeof pred === 'number' ? pred.toFixed(2) : pred}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>

          {/* Results Plot */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Prediction Visualization
              </Typography>
              <Plot
                data={[
                  {
                    type: 'scatter',
                    mode: 'lines+markers',
                    x: data?.map((d, i) => d.timestamp || i),
                    y: predictions,
                    name: 'Predicted',
                    line: { color: '#2196f3' },
                  },
                ]}
                layout={{
                  title: 'Energy Consumption Forecast',
                  xaxis: { title: 'Time' },
                  yaxis: { title: 'Consumption' },
                  height: 400,
                  margin: { t: 30 },
                }}
              />
            </Paper>
          </Grid>
        </>
      )}
    </Grid>
  );
}

export default Prediction;
