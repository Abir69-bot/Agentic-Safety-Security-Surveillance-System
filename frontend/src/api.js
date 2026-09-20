const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || `${window.location.protocol}//${window.location.hostname}:8000`

async function request(path, options) {
  const response = await fetch(`${API_BASE_URL}${path}`, options)
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    throw new Error(data.detail || 'The server could not complete that request.')
  }
  return response.json()
}

export function uploadVideo(file) {
  const body = new FormData()
  body.append('video', file)
  return request('/upload', { method: 'POST', body })
}

export const getStatus = (jobId) => request(`/status/${jobId}`)
export const outputUrl = (filename) => `${API_BASE_URL}/outputs/${encodeURIComponent(filename)}`
