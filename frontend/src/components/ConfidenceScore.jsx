import { motion } from 'framer-motion'
import { Shield, ShieldAlert, ShieldCheck } from 'lucide-react'

function getConfidenceLevel(score) {
  if (score >= 0.7) return { label: 'High', color: 'text-health-green', bg: 'bg-emerald-50', border: 'border-emerald-200', Icon: ShieldCheck }
  if (score >= 0.45) return { label: 'Moderate', color: 'text-health-amber', bg: 'bg-amber-50', border: 'border-amber-200', Icon: Shield }
  return { label: 'Low', color: 'text-health-red', bg: 'bg-red-50', border: 'border-red-200', Icon: ShieldAlert }
}

export default function ConfidenceScore({ score }) {
  if (score == null) return null

  const { label, color, bg, border, Icon } = getConfidenceLevel(score)
  const percentage = Math.round(score * 100)

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border ${bg} ${border} ${color}`}
    >
      <Icon className="w-3.5 h-3.5" />
      <span>Confidence: {percentage}% ({label})</span>
      <div className="w-16 h-1.5 bg-white/60 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className={`h-full rounded-full ${
            score >= 0.7 ? 'bg-health-green' : score >= 0.45 ? 'bg-health-amber' : 'bg-health-red'
          }`}
        />
      </div>
    </motion.div>
  )
}
