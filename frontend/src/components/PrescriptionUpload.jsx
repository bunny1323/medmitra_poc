import { useRef, useState } from 'react'
import {
  Upload,
  AlertCircle,
  Loader2,
  CheckCircle2,
  FileImage,
} from 'lucide-react'

export default function PrescriptionUpload() {
  const fileInputRef = useRef(null)

  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleChooseClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
    setError('')

    // auto analyze immediately after selecting image
    await analyzePrescription(file)
  }

  const analyzePrescription = async (fileToAnalyze) => {
    const file = fileToAnalyze || selectedFile

    if (!file) {
      setError('Please select a prescription image first.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('http://127.0.0.1:8000/api/v1/prescription/upload', {
        method: 'POST',
        headers: {
          'X-Internal-API-Key': 'medmitra123',
        },
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || data.error || 'Failed to analyze prescription')
      }

      setResult(data)
    } catch (err) {
      console.error('Prescription upload error:', err)
      setError(err.message || 'Something went wrong while analyzing the prescription.')
    } finally {
      setLoading(false)
    }
  }

  const medicines = result?.medicines || []
  const hasMedicines = medicines.length > 0

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
        {/* Top Header */}
        <div className="px-6 py-5 border-b border-slate-200 bg-gradient-to-r from-medmitra-50 to-white">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-medmitra-100">
              <Upload className="w-6 h-6 text-medmitra-700" />
            </div>

            <div>
              <h1 className="text-2xl font-bold text-slate-800">
                Prescription Upload
              </h1>
              <p className="text-sm text-slate-500">
                Upload a prescription image and extract medicine details using AI.
              </p>
            </div>
          </div>
        </div>

        <div className="p-6">
          {/* Upload Card */}
          <div className="border-2 border-dashed border-slate-300 rounded-2xl bg-slate-50 p-8 text-center">
            <div className="mx-auto w-16 h-16 rounded-2xl bg-white border border-slate-200 flex items-center justify-center mb-4 shadow-sm">
              <FileImage className="w-8 h-8 text-medmitra-600" />
            </div>

            <h2 className="text-lg font-semibold text-slate-800 mb-2">
              Upload Prescription Image
            </h2>

            <p className="text-sm text-slate-500 mb-5">
              Supported formats: JPG, JPEG, PNG, WEBP (max 5MB)
            </p>

            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp"
              onChange={handleFileChange}
              className="hidden"
            />

            <button
              onClick={handleChooseClick}
              disabled={loading}
              className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-medmitra-600 text-white font-medium hover:bg-medmitra-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing Prescription...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4" />
                  Select Prescription Image
                </>
              )}
            </button>

            {selectedFile && (
              <p className="mt-4 text-sm text-slate-600">
                Selected file: <span className="font-medium">{selectedFile.name}</span>
              </p>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
              <div>
                <h3 className="font-semibold text-red-700">Analysis Error</h3>
                <p className="text-sm text-red-600 mt-1">{error}</p>
              </div>
            </div>
          )}

          {/* Preview + Result */}
          {(previewUrl || result) && (
            <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* LEFT: IMAGE PREVIEW */}
              <div className="bg-slate-50 border border-slate-200 rounded-2xl p-5">
                <h2 className="text-lg font-semibold text-slate-800 mb-4">
                  Prescription Preview
                </h2>

                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Prescription Preview"
                    className="w-full max-h-[520px] object-contain rounded-xl border border-slate-200 bg-white"
                  />
                ) : (
                  <div className="h-[320px] flex items-center justify-center text-slate-400 text-sm">
                    No prescription selected
                  </div>
                )}
              </div>

              {/* RIGHT: ANALYSIS RESULT */}
              <div className="bg-slate-50 border border-slate-200 rounded-2xl p-5">
                <div className="flex items-center gap-2 mb-4">
                  <CheckCircle2 className="w-5 h-5 text-medmitra-600" />
                  <h2 className="text-lg font-semibold text-slate-800">
                    Prescription Analysis Result
                  </h2>
                </div>

                {/* Medicines */}
                <div className="bg-white border border-slate-200 rounded-xl p-4 mb-5">
                  <h3 className="text-base font-semibold text-slate-800 mb-3">
                    Medicines
                  </h3>

                  {hasMedicines ? (
                    <div className="space-y-3">
                      {medicines.map((med, index) => (
                        <div
                          key={index}
                          className="rounded-xl border border-slate-200 p-4 bg-slate-50"
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <p className="font-semibold text-slate-800 text-base">
                                {med.name || 'Unknown medicine'}
                              </p>

                              <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm text-slate-700">
                                <p><span className="font-medium">Dosage:</span> {med.dosage || 'Not specified'}</p>
                                <p><span className="font-medium">Frequency:</span> {med.frequency || 'Not specified'}</p>
                                <p><span className="font-medium">Duration:</span> {med.duration || 'Not specified'}</p>
                                <p><span className="font-medium">Confidence:</span> {med.confidence || 'Medium'}</p>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-slate-500">
                      No medicines extracted.
                    </p>
                  )}
                </div>

                {/* Doctor Notes */}
                <div className="bg-white border border-slate-200 rounded-xl p-4 mb-5">
                  <h3 className="text-base font-semibold text-slate-800 mb-2">
                    Doctor Notes
                  </h3>
                  <p className="text-sm text-slate-700">
                    {result?.doctor_notes?.trim()
                      ? result.doctor_notes
                      : 'No doctor notes found.'}
                  </p>
                </div>

                {/* Unreadable warning */}
                {result?.unreadable_text_present && (
                  <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 mb-5">
                    <h3 className="font-semibold text-amber-700">Warning</h3>
                    <p className="text-sm text-amber-700 mt-1">
                      Some parts of the prescription may be unclear. Please verify extracted details.
                    </p>
                  </div>
                )}

                {/* Backend error */}
                {result?.error && (
                  <div className="rounded-xl border border-red-200 bg-red-50 p-4 mb-5">
                    <h3 className="font-semibold text-red-700">Backend Error</h3>
                    <p className="text-sm text-red-700 mt-1">{result.error}</p>
                  </div>
                )}

                {/* Raw text */}
                {result?.raw_extracted_text && (
                  <div className="bg-white border border-slate-200 rounded-xl p-4 mb-5">
                    <h3 className="text-base font-semibold text-slate-800 mb-2">
                      Raw Extracted Text
                    </h3>
                    <pre className="text-xs text-slate-700 whitespace-pre-wrap break-words bg-slate-50 border border-slate-200 rounded-lg p-3 overflow-x-auto">
                      {result.raw_extracted_text}
                    </pre>
                  </div>
                )}

                {/* Safety note */}
                <div className="rounded-xl border border-blue-200 bg-blue-50 p-4">
                  <h3 className="font-semibold text-blue-700">Medical Safety Note</h3>
                  <p className="text-sm text-blue-700 mt-1">
                    MedMitra extracts prescription text for informational assistance only.
                    Always confirm medicine name, dosage, and usage with a licensed doctor or pharmacist.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}