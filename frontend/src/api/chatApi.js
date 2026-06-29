const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const INTERNAL_API_KEY = 'medmitra123'

// -----------------------------
// Chat Query API
// -----------------------------
export async function sendMessage(message, history = []) {
  const response = await fetch(`${API_BASE}/api/v1/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Internal-API-Key': INTERNAL_API_KEY,
    },
    body: JSON.stringify({
      question: message,
      history,
    }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `Request failed (${response.status})`)
  }

  return response.json()
}

// -----------------------------
// Backend Health Check
// -----------------------------
export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`, {
    headers: {
      'X-Internal-API-Key': INTERNAL_API_KEY,
    },
  })

  if (!response.ok) {
    throw new Error('Health check failed')
  }

  return response.json()
}

// -----------------------------
// Prescription Upload API
// -----------------------------
export async function uploadPrescription(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE}/api/v1/prescription/upload`, {
    method: 'POST',
    headers: {
      'X-Internal-API-Key': INTERNAL_API_KEY,
    },
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `Prescription upload failed (${response.status})`)
  }

  return response.json()
}