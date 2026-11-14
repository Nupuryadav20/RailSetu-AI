
import { Routes, Route } from 'react-router-dom'
import AppLayout from './components/layout/AppLayout'
import Dashboard from './pages/Dashboard'
import Maintenance from './pages/Maintenance'
import PredictiveMaintenance from './pages/PredictiveMaintenance'
import BlockPlanning from './pages/BlockPlanning'
import ModelInsights from './pages/ModelInsights'
import PrivacyPolicy from './pages/PrivacyPolicy'
import NotFound from './pages/NotFound'

function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/prediction" element={<PredictiveMaintenance />} />
        <Route path="/maintenance" element={<Maintenance />} />
        <Route path="/planning" element={<BlockPlanning />} />
        <Route path="/insights" element={<ModelInsights />} />
        <Route path="/privacy" element={<PrivacyPolicy />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}

export default App
