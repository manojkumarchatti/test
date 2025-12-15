import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom/client'
import axios from 'axios'
import './styles.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api'

interface SourceAttribution {
  snippet: string
  metadata: { source: string; page?: number | null; title?: string | null }
  score: number
}

interface AnswerResponse {
  answer: string
  sources: SourceAttribution[]
  from_cache: boolean
}

const App: React.FC = () => {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<AnswerResponse | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState('')

  const ingest = async () => {
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    setStatus('Uploading and indexing...')
    await axios.post(`${API_BASE}/ingest`, form, { headers: { 'Content-Type': 'multipart/form-data' } })
    setStatus('Ingestion complete')
  }

  const ask = async () => {
    if (!question) return
    setLoading(true)
    setStatus('Retrieving answer...')
    const { data } = await axios.post<AnswerResponse>(`${API_BASE}/query`, { question, top_k: 5, mmr_k: 4 })
    setAnswer(data)
    setLoading(false)
    setStatus(data.from_cache ? 'Served from cache' : 'Served fresh')
  }

  return (
    <div className="app">
      <header>
        <h1>Company Knowledge Base</h1>
        <p>Upload PDFs/HTML/Markdown and ask questions with citations.</p>
      </header>

      <section className="upload">
        <h2>1. Ingest documents</h2>
        <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <button onClick={ingest} disabled={!file}>Upload</button>
        <span className="status">{status}</span>
      </section>

      <section className="chat">
        <h2>2. Ask the knowledge base</h2>
        <textarea
          placeholder="Ask about benefits, architecture, or playbooks"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button onClick={ask} disabled={loading}>Ask</button>
      </section>

      {answer && (
        <section className="answer">
          <h3>Answer</h3>
          <p>{answer.answer}</p>
          <div className="sources">
            <h4>Sources</h4>
            {answer.sources.map((source, idx) => (
              <div key={idx} className="source">
                <strong>{source.metadata.source}</strong>
                <p>{source.snippet}</p>
                <small>Score: {source.score.toFixed(2)}</small>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
