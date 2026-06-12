import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Send, Loader2 } from 'lucide-react'

export default function ChatInput({ onSend, isLoading, disabled }) {
  const [input, setInput] = useState('')
  const textareaRef = useRef(null)

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`
    }
  }, [input])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading || disabled) return
    onSend(input)
    setInput('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="flex items-end gap-2 bg-white border border-slate-200 rounded-2xl shadow-lg shadow-slate-200/50 p-2 focus-within:border-medmitra-400 focus-within:ring-2 focus-within:ring-medmitra-100 transition-all">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about medicines, dosages, interactions..."
          rows={1}
          disabled={isLoading || disabled}
          className="flex-1 resize-none bg-transparent px-3 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none disabled:opacity-50"
        />
        <motion.button
          type="submit"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          disabled={!input.trim() || isLoading || disabled}
          className="flex-shrink-0 w-10 h-10 rounded-xl bg-medmitra-600 text-white flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed hover:bg-medmitra-700 transition-colors"
        >
          {isLoading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </motion.button>
      </div>
      <p className="text-center text-[10px] text-slate-400 mt-2">
        MedMitra AI provides information only — not medical advice. Consult a healthcare professional.
      </p>
    </form>
  )
}
