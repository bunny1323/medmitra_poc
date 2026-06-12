import { useEffect, useState } from 'react'
import { AnimatePresence } from 'framer-motion'

import Sidebar from './components/Sidebar'
import Header from './components/Header'
import ChatInterface from './components/ChatInterface'
import { useChat } from './hooks/useChat'
import { checkHealth } from './api/chatApi'

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isConnected, setIsConnected] = useState(false)

  const {
    sessions,
    activeSession,
    activeSessionId,
    isLoading,
    error,
    newChat,
    deleteChat,
    selectChat,
    sendChatMessage,
  } = useChat()

  useEffect(() => {
    const poll = async () => {
      try {
        const health = await checkHealth()

        setIsConnected(
          health.ollama?.status === 'connected' &&
          health.ollama?.model_available
        )
      } catch {
        setIsConnected(false)
      }
    }

    poll()

    const interval = setInterval(poll, 30000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">

      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelect={selectChat}
        onNew={newChat}
        onDelete={deleteChat}
        isOpen={!sidebarCollapsed}
        onClose={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      <main className="flex-1 flex flex-col min-w-0">

        <Header
          isConnected={isConnected}
        />

        <AnimatePresence mode="wait">
          <ChatInterface
            key={activeSessionId}
            messages={activeSession?.messages || []}
            isLoading={isLoading}
            error={error}
            onSend={sendChatMessage}
          />
        </AnimatePresence>

      </main>
    </div>
  )
}