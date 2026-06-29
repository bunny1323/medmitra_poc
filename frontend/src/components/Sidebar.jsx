import { motion, AnimatePresence } from 'framer-motion'
import {
  Plus,
  MessageSquare,
  Trash2,
  X,
  PanelLeftClose,
  PanelLeftOpen,
  FileImage,
} from 'lucide-react'

function formatDate(timestamp) {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now - date

  if (diff < 86400000) return 'Today'
  if (diff < 172800000) return 'Yesterday'

  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
  })
}

export default function Sidebar({
  sessions,
  activeSessionId,
  activePage,
  onSelect,
  onNew,
  onDelete,
  onSelectPage,
  isOpen,
  onClose,
}) {
  return (
    <>
      {/* Desktop Sidebar */}
      <aside
        className={`hidden md:flex bg-white border-r border-slate-200 flex-col transition-all duration-300 ${
          isOpen ? 'w-72' : 'w-20'
        }`}
      >
        {/* Header */}
        <div className="h-16 px-4 border-b border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-3 overflow-hidden">
            <img
              src="/logo.png"
              alt="MedMitra"
              className="w-9 h-9 object-contain flex-shrink-0"
            />

            {isOpen && (
              <div>
                <h1 className="font-bold text-slate-800 text-lg">
                  MedMitra AI
                </h1>
                <p className="text-[10px] text-slate-400">
                  Healthcare Assistant
                </p>
              </div>
            )}
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-slate-100 text-slate-600 transition"
          >
            {isOpen ? (
              <PanelLeftClose className="w-5 h-5" />
            ) : (
              <PanelLeftOpen className="w-5 h-5" />
            )}
          </button>
        </div>

        {/* New Chat */}
        <div className="p-3">
          <button
            onClick={onNew}
            className={`w-full flex items-center ${
              isOpen ? 'justify-center gap-2' : 'justify-center'
            } px-4 py-3 rounded-xl bg-medmitra-600 text-white hover:bg-medmitra-700 transition`}
          >
            <Plus className="w-5 h-5" />

            {isOpen && (
              <span className="text-sm font-medium">
                New Chat
              </span>
            )}
          </button>
        </div>

        {/* MAIN NAV */}
        <div className="px-2 pb-2">
          {isOpen && (
            <p className="px-3 py-2 text-xs text-slate-400 uppercase font-semibold">
              Workspace
            </p>
          )}

          {/* Chat page */}
          <button
            onClick={() => onSelectPage('chat')}
            className={`w-full flex items-center gap-3 rounded-xl px-3 py-3 transition mb-1 ${
              activePage === 'chat'
                ? 'bg-medmitra-50 border border-medmitra-200 text-medmitra-800'
                : 'hover:bg-slate-100 text-slate-700'
            }`}
          >
            <MessageSquare className="w-4 h-4 flex-shrink-0" />

            {isOpen && (
              <span className="text-sm font-medium">
                Chat Assistant
              </span>
            )}
          </button>

          {/* Prescription Upload page */}
          <button
            onClick={() => onSelectPage('prescription')}
            className={`w-full flex items-center gap-3 rounded-xl px-3 py-3 transition ${
              activePage === 'prescription'
                ? 'bg-medmitra-50 border border-medmitra-200 text-medmitra-800'
                : 'hover:bg-slate-100 text-slate-700'
            }`}
          >
            <FileImage className="w-4 h-4 flex-shrink-0" />

            {isOpen && (
              <span className="text-sm font-medium">
                Prescription Upload
              </span>
            )}
          </button>
        </div>

        {/* Chat List */}
        <div className="flex-1 overflow-y-auto px-2">
          {isOpen && (
            <p className="px-3 py-2 text-xs text-slate-400 uppercase font-semibold">
              Chat History
            </p>
          )}

          {sessions.map((session) => (
            <div
              key={session.id}
              className="group relative mb-1"
            >
              <button
                onClick={() => {
                  onSelectPage('chat')
                  onSelect(session.id)
                }}
                className={`w-full flex items-center gap-3 rounded-xl px-3 py-3 transition ${
                  session.id === activeSessionId && activePage === 'chat'
                    ? 'bg-medmitra-50 border border-medmitra-200 text-medmitra-800'
                    : 'hover:bg-slate-100 text-slate-700'
                }`}
              >
                <MessageSquare className="w-4 h-4 flex-shrink-0" />

                {isOpen && (
                  <div className="flex-1 text-left min-w-0">
                    <p className="truncate text-sm font-medium">
                      {session.title}
                    </p>

                    <p className="text-[10px] text-slate-400">
                      {formatDate(session.updatedAt)}
                    </p>
                  </div>
                )}
              </button>

              {isOpen && (
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onDelete(session.id)
                  }}
                  className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition p-1.5 rounded-md hover:bg-red-50 text-slate-400 hover:text-red-500"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="border-t border-slate-200 p-4">
          <div
            className={`flex items-center ${
              isOpen ? 'gap-2' : 'justify-center'
            }`}
          >
            <img
              src="/logo.png"
              alt="logo"
              className="w-5 h-5"
            />

            {isOpen && (
              <span className="text-xs text-slate-500">
                Powered by Llama + RAG + Vision
              </span>
            )}
          </div>
        </div>
      </aside>

      {/* Mobile Sidebar */}
      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={onClose}
              className="md:hidden fixed inset-0 bg-black/40 z-40"
            />

            <motion.aside
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{
                type: 'spring',
                damping: 25,
                stiffness: 300,
              }}
              className="md:hidden fixed left-0 top-0 bottom-0 w-72 bg-white z-50 shadow-xl"
            >
              <div className="h-16 px-4 border-b border-slate-200 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <img
                    src="/logo.png"
                    alt="logo"
                    className="w-8 h-8"
                  />

                  <h1 className="font-bold text-slate-800">
                    MedMitra AI
                  </h1>
                </div>

                <button
                  onClick={onClose}
                  className="p-2 rounded-lg hover:bg-slate-100"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="p-4 space-y-2">
                <button
                  onClick={() => {
                    onSelectPage('chat')
                    onClose()
                  }}
                  className={`w-full flex items-center gap-3 rounded-xl px-3 py-3 transition ${
                    activePage === 'chat'
                      ? 'bg-medmitra-50 border border-medmitra-200 text-medmitra-800'
                      : 'hover:bg-slate-100 text-slate-700'
                  }`}
                >
                  <MessageSquare className="w-4 h-4" />
                  <span className="text-sm font-medium">Chat Assistant</span>
                </button>

                <button
                  onClick={() => {
                    onSelectPage('prescription')
                    onClose()
                  }}
                  className={`w-full flex items-center gap-3 rounded-xl px-3 py-3 transition ${
                    activePage === 'prescription'
                      ? 'bg-medmitra-50 border border-medmitra-200 text-medmitra-800'
                      : 'hover:bg-slate-100 text-slate-700'
                  }`}
                >
                  <FileImage className="w-4 h-4" />
                  <span className="text-sm font-medium">Prescription Upload</span>
                </button>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  )
}