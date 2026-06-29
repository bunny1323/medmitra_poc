import { useRef, useState } from 'react'
import { Upload, FileImage, AlertTriangle, Loader2 } from 'lucide-react'

export default function PrescriptionUpload() {
  const fileInputRef = useRef(null)

  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const handleFileChange = (file) => {
    if (!file) return

    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
    setError('')
  }

  const handleInputChange = (e) => {
    const file = e.target.files?.[0]
    handleFileChange(file)
  }

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select a prescription image first.')
      return
    }

    try {
      setLoading(true)
      setError('')
      setResult(null)

      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch(
        'http://127.0.0.1:8000/api/v1/prescription/upload',
        {
          method: 'POST',
          headers: {
            'X-Internal-API-Key': 'medmitra123',
          },
          body: formData,
        }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data?.detail || 'Failed to analyze prescription.')
      }

      setResult(data)
    } catch (err) {
      setError(err.message || 'Something went wrong while analyzing prescription.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 bg-slate-50">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Title */}
        <div>
          <h1 className="text-2xl md:text-3xl font-bold text-slate-800">
            Prescription Upload
          </h1>
          <p className="text-slate-500 mt-2">
            Upload a prescription image and let MedMitra extract medicines, dosage, frequency, and notes.
          </p>
        </div>

        {/* Upload Card */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
          <div className="grid md:grid-cols-2 gap-6">
            {/* Left: Upload */}
            <div className="space-y-4">
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-300 rounded-2xl p-8 text-center cursor-pointer hover:border-medmitra-400 hover:bg-medmitra-50/30 transition"
              >
                <div className="flex justify-center mb-4">
                  <div className="w-14 h-14 rounded-full bg-medmitra-100 flex items-center justify-center">
                    <Upload className="w-7 h-7 text-medmitra-600" />
                  </div>
                </div>

                <h2 className="text-lg font-semibold text-slate-800">
                  Upload Prescription Image
                </h2>
                <p className="text-sm text-slate-500 mt-2">
                  Supported formats: JPG, PNG, WEBP
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  Max file size: 5MB
                </p>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
                  className="hidden"
                  onChange={handleInputChange}
                />
              </div>

              {selectedFile && (
                <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 flex items-center gap-3">
                  <FileImage className="w-5 h-5 text-medmitra-600" />
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-slate-800 truncate">
                      {selectedFile.name}
                    </p>
                    <p className="text-xs text-slate-500">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
              )}

              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="w-full md:w-auto inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-medmitra-600 text-white hover:bg-medmitra-700 transition disabled:opacity-60"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  'Analyze Prescription'
                )}
              </button>

              {error && (
                <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {error}
                </div>
              )}
            </div>

            {/* Right: Preview */}
            <div className="bg-slate-50 rounded-2xl border border-slate-200 p-4 min-h-[320px] flex items-center justify-center">
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Prescription preview"
                  className="max-h-[420px] w-auto rounded-xl object-contain"
                />
              ) : (
                <div className="text-center text-slate-400">
                  <FileImage className="w-12 h-12 mx-auto mb-3" />
                  <p className="text-sm">Prescription preview will appear here</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Result */}
        {result && (
          <div className="space-y-6">
            {/* Warning */}
            {result.unreadable_text_present && (
              <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4 flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-amber-800">
                    Some prescription text may be unclear
                  </h3>
                  <p className="text-sm text-amber-700 mt-1">
                    Please verify medicine names and dosage with a doctor or pharmacist before use.
                  </p>
                </div>
              </div>
            )}

            {/* Medicines table */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
              <h2 className="text-xl font-bold text-slate-800 mb-4">
                Prescription Analysis Result
              </h2>

              {result.medicines?.length ? (
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-slate-50 text-left">
                        <th className="px-4 py-3 text-sm font-semibold text-slate-700 rounded-l-xl">
                          Medicine
                        </th>
                        <th className="px-4 py-3 text-sm font-semibold text-slate-700">
                          Dosage
                        </th>
                        <th className="px-4 py-3 text-sm font-semibold text-slate-700">
                          Frequency
                        </th>
                        <th className="px-4 py-3 text-sm font-semibold text-slate-700">
                          Duration
                        </th>
                        <th className="px-4 py-3 text-sm font-semibold text-slate-700 rounded-r-xl">
                          Confidence
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.medicines.map((med, index) => (
                        <tr
                          key={`${med.name}-${index}`}
                          className="border-b border-slate-100 last:border-b-0"
                        >
                          <td className="px-4 py-4 text-sm font-medium text-slate-800">
                            {med.name}
                          </td>
                          <td className="px-4 py-4 text-sm text-slate-600">
                            {med.dosage}
                          </td>
                          <td className="px-4 py-4 text-sm text-slate-600">
                            {med.frequency}
                          </td>
                          <td className="px-4 py-4 text-sm text-slate-600">
                            {med.duration}
                          </td>
                          <td className="px-4 py-4">
                            <span
                              className={`inline-flex px-3 py-1 rounded-full text-xs font-semibold ${
                                med.confidence === 'High'
                                  ? 'bg-green-100 text-green-700'
                                  : med.confidence === 'Medium'
                                  ? 'bg-yellow-100 text-yellow-700'
                                  : 'bg-red-100 text-red-700'
                              }`}
                            >
                              {med.confidence}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
                  No medicines were extracted from this prescription.
                </div>
              )}

              {/* Doctor notes */}
              <div className="mt-6">
                <h3 className="text-lg font-semibold text-slate-800 mb-2">
                  Doctor Notes
                </h3>
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700 min-h-[60px]">
                  {result.doctor_notes?.trim()
                    ? result.doctor_notes
                    : 'No additional doctor notes extracted.'}
                </div>
              </div>

              {/* Raw extracted text */}
              <div className="mt-6">
                <h3 className="text-lg font-semibold text-slate-800 mb-2">
                  Raw Extracted Text
                </h3>
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-xs md:text-sm text-slate-700 whitespace-pre-wrap break-words">
                  {result.raw_extracted_text || 'No raw extracted text available.'}
                </div>
              </div>
            </div>

            {/* Safety Note */}
            <div className="rounded-2xl border border-blue-200 bg-blue-50 p-5">
              <h3 className="font-semibold text-blue-800 mb-2">
                Important Medical Safety Note
              </h3>
              <p className="text-sm text-blue-700 leading-6">
                MedMitra extracts prescription text for informational assistance only.
                Always confirm medicine name, dosage, and usage instructions with a
                licensed doctor or pharmacist before taking any medication.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}