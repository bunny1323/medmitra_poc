import { useCallback, useEffect, useState } from 'react'
import { sendMessage } from '../api/chatApi'

const STORAGE_KEY = 'medmitra_chat_sessions'

function loadSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveSessions(sessions) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions))
}

function createSession() {
  return {
    id: crypto.randomUUID(),
    title: 'New Chat',
    messages: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  }
}

export function useChat() {
  const [sessions, setSessions] = useState(loadSessions)
  const [activeSessionId, setActiveSessionId] = useState(() => {
    const stored = loadSessions()
    return stored[0]?.id || null
  })
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (sessions.length === 0) {
      const session = createSession()
      setSessions([session])
      setActiveSessionId(session.id)
    } else if (!activeSessionId) {
      setActiveSessionId(sessions[0].id)
    }
  }, [sessions.length, activeSessionId])

  const activeSession = sessions.find((s) => s.id === activeSessionId)

  const updateSessions = useCallback((updater) => {
    setSessions((prev) => {
      const next = typeof updater === 'function' ? updater(prev) : updater
      saveSessions(next)
      return next
    })
  }, [])

  const newChat = useCallback(() => {
    const session = createSession()
    updateSessions((prev) => [session, ...prev])
    setActiveSessionId(session.id)
    setError(null)
  }, [updateSessions])

  const deleteChat = useCallback(
    (sessionId) => {
      updateSessions((prev) => {
        const next = prev.filter((s) => s.id !== sessionId)
        if (sessionId === activeSessionId) {
          setActiveSessionId(next[0]?.id || null)
        }
        if (next.length === 0) {
          const session = createSession()
          setActiveSessionId(session.id)
          return [session]
        }
        return next
      })
    },
    [activeSessionId, updateSessions],
  )

  const selectChat = useCallback((sessionId) => {
    setActiveSessionId(sessionId)
    setError(null)
  }, [])

  const sendChatMessage = useCallback(
    async (content) => {
      if (!activeSessionId || !content.trim()) return

      const userMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: content.trim(),
        timestamp: Date.now(),
      }

      updateSessions((prev) =>
        prev.map((s) => {
          if (s.id !== activeSessionId) return s
          const title =
            s.messages.length === 0
              ? content.trim().slice(0, 40) + (content.length > 40 ? '…' : '')
              : s.title
          return {
            ...s,
            title,
            messages: [...s.messages, userMessage],
            updatedAt: Date.now(),
          }
        }),
      )

      setIsLoading(true)
      setError(null)

      try {
        const history =
          activeSession?.messages.map((m) => ({
            role: m.role,
            content: m.content,
          })) || []

        const data = await sendMessage(content, history)

        const assistantMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: data.response,
          confidence: data.confidence,
          isEmergency: data.is_emergency,
          emergencyMessage: data.emergency_message,
          sources: data.sources || [],
          timestamp: Date.now(),
        }

        updateSessions((prev) =>
          prev.map((s) =>
            s.id === activeSessionId
              ? {
                  ...s,
                  messages: [...s.messages, assistantMessage],
                  updatedAt: Date.now(),
                }
              : s,
          ),
        )
      } catch (err) {
        setError(err.message)
      } finally {
        setIsLoading(false)
      }
    },
    [activeSessionId, activeSession, updateSessions],
  )

  return {
    sessions,
    activeSession,
    activeSessionId,
    isLoading,
    error,
    newChat,
    deleteChat,
    selectChat,
    sendChatMessage,
  }
}
