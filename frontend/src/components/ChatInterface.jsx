import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import {
  Stethoscope,
  Pill,
  Heart,
  ShieldAlert,
} from 'lucide-react'

import MessageBubble from './MessageBubble'
import ChatInput from './ChatInput'

const SUGGESTIONS = [
  'What is Paracetamol used for?',
  'What are the side effects of Ibuprofen?',
  'Can I take Ibuprofen with Warfarin?',
  'Tell me about Metformin',
]

function WelcomeScreen({ onSuggestionClick }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex flex-col items-center justify-center flex-1 px-6 py-12"
    >
      <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-medmitra-500 to-medmitra-700 flex items-center justify-center shadow-lg mb-6">
        <Stethoscope className="w-10 h-10 text-white" />
      </div>

      <h1 className="text-3xl font-bold text-slate-800 mb-2 text-center">
        MedMitra AI Assistant
      </h1>

      <p className="text-slate-500 text-center max-w-xl mb-8">
        Healthcare Information & Medicine Guidance Platform
      </p>

      <div className="grid md:grid-cols-3 gap-4 w-full max-w-4xl mb-10">
        <div className="bg-white border border-slate-200 rounded-xl p-4 text-center shadow-sm">
          <Pill className="w-6 h-6 mx-auto text-medmitra-600 mb-2" />
          <h3 className="font-semibold text-slate-800">
            Medicine Information
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Uses, dosage, warnings and side effects
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-4 text-center shadow-sm">
          <Heart className="w-6 h-6 mx-auto text-medmitra-600 mb-2" />
          <h3 className="font-semibold text-slate-800">
            Healthcare Guidance
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Evidence-based healthcare information
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-4 text-center shadow-sm">
          <ShieldAlert className="w-6 h-6 mx-auto text-medmitra-600 mb-2" />
          <h3 className="font-semibold text-slate-800">
            Drug Safety
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Interactions and warning alerts
          </p>
        </div>
      </div>

      <div className="w-full max-w-2xl">
        <p className="text-sm font-medium text-slate-500 mb-4 text-center">
          Try asking:
        </p>

        <div className="grid sm:grid-cols-2 gap-3">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              onClick={() => onSuggestionClick(suggestion)}
              className="text-left p-4 rounded-xl bg-white border border-slate-200 hover:border-medmitra-300 hover:bg-medmitra-50 transition-all"
            >
              <span className="text-sm text-slate-700">
                {suggestion}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="mt-10 text-center">
        <p className="text-xs text-slate-400">
          MedMitra AI provides information only — not medical advice.
        </p>
      </div>
    </motion.div>
  )
}

function TypingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex gap-3"
    >
      <div className="w-8 h-8 rounded-lg bg-medmitra-600 flex items-center justify-center">
        <Stethoscope className="w-4 h-4 text-white" />
      </div>

      <div className="bg-white border border-slate-200 rounded-2xl px-4 py-3">
        <div className="flex gap-1">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className="w-2 h-2 bg-medmitra-500 rounded-full"
              animate={{
                y: [0, -5, 0],
              }}
              transition={{
                duration: 0.6,
                repeat: Infinity,
                delay: i * 0.1,
              }}
            />
          ))}
        </div>
      </div>
    </motion.div>
  )
}

export default function ChatInterface({
  messages,
  isLoading,
  error,
  onSend,
}) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: 'smooth',
    })
  }, [messages, isLoading])

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <WelcomeScreen onSuggestionClick={onSend} />
        ) : (
          <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
              />
            ))}

            {isLoading && <TypingIndicator />}

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {error && (
        <div className="mx-4 mb-2 bg-red-50 border border-red-200 text-red-700 rounded-lg p-3 text-sm">
          {error}
        </div>
      )}

      <div className="border-t border-slate-200 bg-white p-4">
        <div className="max-w-4xl mx-auto">
          <ChatInput
            onSend={onSend}
            isLoading={isLoading}
          />
        </div>

        <p className="text-center text-xs text-slate-400 mt-3">
          MedMitra AI provides information only — not medical advice. Consult a healthcare professional.
        </p>
      </div>
    </div>
  )
}