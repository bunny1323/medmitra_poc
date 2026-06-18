import { motion } from 'framer-motion'
import { AlertTriangle, Phone } from 'lucide-react'

export default function EmergencyWarning({ message }) {
  if (!message) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      className="mb-3 rounded-xl border-2 border-red-300 bg-gradient-to-r from-red-50 to-orange-50 p-4 shadow-sm"
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 w-10 h-10 rounded-full bg-red-100 flex items-center justify-center animate-pulse-soft">
          <AlertTriangle className="w-5 h-5 text-red-600" />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-bold text-red-800 mb-1">
            Medical Emergency Detected
          </h4>
          <p className="text-sm text-red-700 leading-relaxed">{message}</p>
          <div className="mt-3 flex items-center gap-2 text-xs font-semibold text-red-800">
            <Phone className="w-3.5 h-3.5" />
            <span>Call emergency services immediately — do not rely on AI advice</span>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
