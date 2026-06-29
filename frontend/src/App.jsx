import { useEffect, useState } from 'react'
import { AnimatePresence } from 'framer-motion'

import Sidebar from './components/Sidebar'
import Header from './components/Header'
import ChatInterface from './components/ChatInterface'
import PrescriptionUpload from './components/PrescriptionUpload'
import { useChat } from './hooks/useChat'
import { checkHealth } from './api/chatApi'

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isConnected, setIsConnected] = useState(false)

  // Page mode: chat or prescription upload
  const [activePage, setActivePage] = useState('chat')

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
    <div className="flex h-screen bg-slate-50 overflow-hidden">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        activePage={activePage}
        onSelect={(id) => {
          setActivePage('chat')
          selectChat(id)
        }}
        onNew={() => {
          setActivePage('chat')
          newChat()
        }}
        onDelete={deleteChat}
        onSelectPage={setActivePage}
        isOpen={!sidebarCollapsed}
        onClose={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Right side main area */}
      <main className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
        {/* Fixed header */}
        <Header isConnected={isConnected} />

        {/* Scrollable page content */}
        <div className="flex-1 overflow-y-auto min-h-0">
          <AnimatePresence mode="wait">
            {activePage === 'chat' ? (
              <ChatInterface
                key={`chat-${activeSessionId}`}
                messages={activeSession?.messages || []}
                isLoading={isLoading}
                error={error}
                onSend={sendChatMessage}
              />
            ) : (
              <PrescriptionUpload key="prescription-upload" />
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  )
}