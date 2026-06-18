import { motion } from "framer-motion";
import { Activity } from "lucide-react";

export default function Header({ isConnected }) {
  return (
    <header className="flex items-center justify-between px-6 py-4 bg-white border-b border-slate-200 shadow-sm sticky top-0 z-30">
      
      {/* Left Side */}
      <div className="flex items-center gap-4">

        <img
          src="/logo.png"
          alt="MedMitra"
          className="w-12 h-12 object-contain"
        />

        <div>
          <h1 className="text-xl font-bold text-[#0F766E]">
            MedMitra AI Assistant
          </h1>

          <p className="text-sm text-slate-500">
            Healthcare Information & Medicine Guidance Platform
          </p>
        </div>
      </div>

      {/* Right Side Status */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium ${
          isConnected
            ? "bg-green-50 text-green-700 border border-green-200"
            : "bg-red-50 text-red-700 border border-red-200"
        }`}
      >
        <Activity className="w-4 h-4" />
        {isConnected ? "System Online" : "System Offline"}
      </motion.div>
    </header>
  );
}