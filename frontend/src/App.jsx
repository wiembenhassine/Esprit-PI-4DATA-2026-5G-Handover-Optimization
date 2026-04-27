import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './AuthContext'
import './styles.css'

import Login    from './pages/Login'
import Layout   from './components/Layout'
import Overview    from './pages/Overview'
import Predict     from './pages/Predict'
import Monitoring  from './pages/Monitoring'
import MLOps       from './pages/MLOps'
import History     from './pages/History'

function ProtectedRoute({ children, requireScientist, requireAdmin }) {
  const { isAuthenticated, isScientist, isAdmin, loading } = useAuth()
  if (loading) return <div style={{ display:'flex',alignItems:'center',justifyContent:'center',height:'100vh',color:'#3d5166',fontSize:12 }}>Loading...</div>
  if (!isAuthenticated) return <Navigate to="/login" replace />
  if (requireAdmin    && !isAdmin)     return <Navigate to="/"  replace />
  if (requireScientist && !isScientist) return <Navigate to="/" replace />
  return children
}

function AppRoutes() {
  const { isAuthenticated, loading } = useAuth()
  if (loading) return null
  return (
    <Routes>
      <Route path="/login" element={isAuthenticated ? <Navigate to="/" replace /> : <Login />} />
      <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
        <Route index          element={<Overview />} />
        <Route path="predict" element={<Predict />} />
        <Route path="history" element={<History />} />
        <Route path="monitoring" element={<Monitoring />} />
        <Route path="mlops"   element={<ProtectedRoute requireScientist><MLOps /></ProtectedRoute>} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  )
}
