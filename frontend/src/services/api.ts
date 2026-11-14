
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 15000,
})

export default api

export const dashboardApi = {
  getSummary: () => api.get('/api/dashboard/summary'),
  getMaintenanceRequests: (params: {
    page?: number;
    page_size?: number;
    search?: string;
    department?: string;
    priority?: string;
    corridor?: string;
  } = {}) => api.get('/api/maintenance/requests', { params }),
  getPriority: (params = {}) =>
    api.get('/api/maintenance/priority', { params }),
  getOptimizedSchedule: () => api.get('/api/schedule/optimized'),
  getUnscheduled: () => api.get('/api/schedule/unscheduled'),
  getWeeklyPlan: () => api.get('/api/planning/weekly'),
  getMonthlyPlan: () => api.get('/api/planning/monthly'),
  getDepartments: () => api.get('/api/planning/departments'),
  getCorridors: () => api.get('/api/planning/corridors'),
  getModelComparison: () => api.get('/api/ml/model-comparison'),
  getBestModel: () => api.get('/api/ml/best-model'),
  getFeatureImportance: () => api.get('/api/ml/feature-importance'),
  getPredictionExplanations: () => api.get('/api/ml/prediction-explanations'),
  getValidation: () => api.get('/api/optimization/validation'),
  getBlockWindows: () => api.get('/api/simulation/block-windows'),
  predict: (records: unknown[]) =>
    api.post('/api/ml/predict', { records }),
  scoreRisk: (records: unknown[]) =>
    api.post('/api/risk/score', { records }),
  getMetricsFull: () => api.get('/api/ml/metrics/full'),
  getMetricsVal: () => api.get('/api/ml/metrics/val'),
  getMetricsTest: () => api.get('/api/ml/metrics/test'),
  getThresholdAnalysis: () => api.get('/api/ml/threshold-analysis'),
  getCvSummary: () => api.get('/api/ml/cv-summary'),
  getSeedStability: () => api.get('/api/ml/seed-stability'),
  getPermutationImportance: () => api.get('/api/ml/permutation-importance'),
  getShapV4: () => api.get('/api/ml/shap/v4'),
  getShapExamples: () => api.get('/api/ml/shap/examples'),
  getMetadata: () => api.get('/api/ml/metadata'),
  getPredictionHistory: () => api.get('/api/predict/history'),
  getDatasetStats: () => api.get('/api/dataset/stats'),
  // V5 Endpoints
  getV5Comparison: (split = 'test') => api.get('/api/ml/v5/comparison', { params: { split } }),
  getV5Metrics: (model: string) => api.get(`/api/ml/v5/metrics/${model}`),
  getV5Thresholds: () => api.get('/api/ml/v5/thresholds'),
  getV5Cv: () => api.get('/api/ml/v5/cv'),
  getV5FeatureSelection: () => api.get('/api/ml/v5/feature-selection'),
  getV5FeatureStats: () => api.get('/api/ml/v5/feature-stats'),
  getV5Experiments: () => api.get('/api/ml/v5/experiments'),
  getV5Metadata: () => api.get('/api/ml/v5/metadata'),
  getPredictions: (params = {}) => api.get('/api/ml/predictions', { params }),
  getPredictionsDownloadUrl: (limit = 500) => `${api.defaults.baseURL || 'http://127.0.0.1:8000'}/api/ml/predictions/download?limit=${limit}`,
}
