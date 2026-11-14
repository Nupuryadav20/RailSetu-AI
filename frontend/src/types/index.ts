
export interface DashboardSummary {
  total_requests: number
  scheduled_requests: number
  unscheduled_requests: number
  total_work_hours: number
}

export interface MaintenanceRequest {
  request_id: string
  department: string
  priority: string
  corridor: string
  status?: string
  [key: string]: unknown
}

export interface BlockWindow {
  window_id: string
  corridor: string
  start_time: string
  end_time: string
  [key: string]: unknown
}

export interface PredictionResult {
  predicted_class: number
  probability_percent: number
  model_name: string
  [key: string]: unknown
}

export interface ApiResponse<T> {
  data: T
  message?: string
}
