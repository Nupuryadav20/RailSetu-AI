
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
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
}
