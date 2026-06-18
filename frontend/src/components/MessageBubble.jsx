import { motion } from 'framer-motion'
import { Bot, User, BookOpen, ShieldCheck } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import ConfidenceScore from './ConfidenceScore'
import EmergencyWarning from './EmergencyWarning'

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
    >
      <div
        className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center shadow-sm ${
          isUser
            ? 'bg-medmitra-600 text-white'
            : 'bg-gradient-to-br from-medmitra-500 to-medmitra-700 text-white'
        }`}
      >
        {isUser ? (
          <User className="w-5 h-5" />
        ) : (
          <Bot className="w-5 h-5" />
        )}
      </div>

      <div
        className={`flex flex-col ${
          isUser ? 'items-end' : 'items-start'
        } flex-1 max-w-[90%] md:max-w-[75%]`}
      >
        {!isUser && (
          <div className="flex items-center gap-1 mb-1 text-xs text-slate-500">
            <ShieldCheck className="w-3 h-3 text-green-600" />
            MedMitra AI Assistant
          </div>
        )}

        <div
          className={`rounded-2xl px-5 py-4 shadow-sm ${
            isUser
              ? 'bg-medmitra-600 text-white rounded-tr-sm'
              : 'bg-white border border-slate-200 text-slate-800 rounded-tl-sm'
          }`}
        >
          {!isUser && message.isEmergency && (
            <EmergencyWarning message={message.emergencyMessage} />
          )}

          <div
            className={`prose prose-sm max-w-none ${
              isUser ? 'prose-invert' : ''
            }`}
          >
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>

          {!isUser && message.confidence != null && (
            <div className="mt-4 pt-3 border-t border-slate-100">
              <ConfidenceScore score={message.confidence} />
            </div>
          )}

          {!isUser && message.sources?.length > 0 && (
            <div className="mt-4 pt-3 border-t border-slate-100">
              <p className="text-xs font-semibold text-slate-500 mb-2 flex items-center gap-1">
                <BookOpen className="w-3 h-3" />
                Verified Medicine Sources
              </p>

              <div className="flex flex-wrap gap-2">
                {message.sources.map((src, i) => (
                  <div
                    key={i}
                    className="px-2.5 py-1 rounded-lg bg-medmitra-50 border border-medmitra-100 text-xs"
                  >
                    <div className="font-medium text-medmitra-700">
                      {src.name}
                    </div>

                    <div className="text-medmitra-500">
                      {Math.round(src.similarity * 100)}% Match
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}