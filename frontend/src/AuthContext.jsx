import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { login as apiLogin, getMe } from './api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser]       = useState(null)   // { username, role, user_id }
  const [token, setToken]     = useState(null)
  const [loading, setLoading] = useState(true)   // true while checking localStorage

  // On mount: restore session from localStorage
  useEffect(() => {
    const savedToken = localStorage.getItem('token')
    const savedUser  = localStorage.getItem('user')
    if (savedToken && savedUser) {
      setToken(savedToken)
      setUser(JSON.parse(savedUser))
    }
    setLoading(false)
  }, [])

  const login = useCallback(async (username, password) => {
    const data = await apiLogin(username, password)
    // data = { access_token, token_type, expires_in, role }
    localStorage.setItem('token', data.access_token)
    setToken(data.access_token)

    // Fetch full user profile
    const me = await getMe()
    localStorage.setItem('user', JSON.stringify(me))
    setUser(me)
    return me
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setToken(null)
    setUser(null)
  }, [])

  const isAuthenticated = Boolean(token && user)

  // Role helpers
  const isAdmin      = user?.role === 'admin'
  const isScientist  = user?.role === 'data_scientist' || isAdmin
  const isOperator   = Boolean(user)

  return (
    <AuthContext.Provider value={{ user, token, loading, login, logout, isAuthenticated, isAdmin, isScientist, isOperator }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
