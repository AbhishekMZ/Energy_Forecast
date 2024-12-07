import React, { useState } from 'react';
import { useQuery, useMutation } from 'react-query';
import {
  Grid,
  Paper,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  CircularProgress,
  Alert,
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import Plot from 'react-plotly.js';

function Training() {
  const [selectedModel, setSelectedModel] = useState('');
  const [optimizeConfig, setOptimizeConfig] = useState(true);
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);

  // Get available models
  const { data: modelsData } = useQuery('models', async () => {
    const response = await axios.get('http://localhost:8000/models');
    return response.data;
  });

  // Get job status
  const { data: jobStatus, isLoading: isJobLoading } = useQuery(
    ['job', jobId],
    async () => {
      const response = await axios.get(`http://localhost:8000/jobs/${jobId}`);
      return response.data;
    },
    {
      enabled: !!jobId,
      refetchInterval: (data) =>
        data?.status === 'completed' || data?.status === 'failed' ? false : 1000,
    }
  );

  // Training mutation
  const trainMutation = useMutation(async () => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append(
      'request',
      JSON.stringify({
        model_type: selectedModel,
        optimize_config: optimizeConfig,
      })
    );

    const response = await axios.post('http://localhost:8000/train', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  });

  // File dropzone
  const { getRootProps, getInputProps } = useDropzone({
    accept: {
      'text/csv': ['.csv'],
    },
    onDrop: (acceptedFiles) => {
      setFile(acceptedFiles[0]);
    },
  });

  const handleTrain = async () => {
    const result = await trainMutation.mutateAsync();
    setJobId(result.job_id);
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h4" gutterBottom>
          Model Training
        </Typography>
      </Grid>

      {/* Training Configuration */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Configuration
          </Typography>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Model Type</InputLabel>
            <Select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {modelsData?.models?.map((model) => (
                <MenuItem key={model} value={model}>
                  {model}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControlLabel
            control={
              <Switch
                checked={optimizeConfig}
                onChange={(e) => setOptimizeConfig(e.target.checked)}
              />
            }
            label="Optimize Configuration"
          />
        </Paper>
      </Grid>

      {/* File Upload */}
      <Grid item xs={12} md={6}>
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

      {/* Training Controls */}
      <Grid item xs={12}>
        <Button
          variant="contained"
          onClick={handleTrain}
          disabled={!file || !selectedModel || trainMutation.isLoading}
          fullWidth
        >
          {trainMutation.isLoading ? (
            <CircularProgress size={24} />
          ) : (
            'Start Training'
          )}
        </Button>
      </Grid>

      {/* Training Status */}
      {jobId && (
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Training Status
            </Typography>
            {isJobLoading ? (
              <CircularProgress />
            ) : (
              <>
                <Alert
                  severity={
                    jobStatus?.status === 'completed'
                      ? 'success'
                      : jobStatus?.status === 'failed'
                      ? 'error'
                      : 'info'
                  }
                >
                  Status: {jobStatus?.status}
                </Alert>
                {jobStatus?.results && (
                  <Grid container spacing={2} sx={{ mt: 2 }}>
                    {Object.entries(jobStatus.results.metrics).map(
                      ([metric, value]) => (
                        <Grid item xs={6} md={3} key={metric}>
                          <Paper sx={{ p: 2 }}>
                            <Typography variant="body2" color="textSecondary">
                              {metric}
                            </Typography>
                            <Typography variant="h6">
                              {typeof value === 'number'
                                ? value.toFixed(4)
                                : value}
                            </Typography>
                          </Paper>
                        </Grid>
                      )
                    )}
                  </Grid>
                )}
              </>
            )}
          </Paper>
        </Grid>
      )}
    </Grid>
  );
}

export default Training;
