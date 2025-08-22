// src/app/page.tsx
"use client"
import { useMemo, useState } from "react"
import type React from "react"
import { Upload, AlertTriangle, CheckCircle, Info, Shield, Camera, FileText, TrendingUp } from "lucide-react"

/** CHANGE: Map short labels -> full names (HAM10000-style keys) */
const CLASS_LABELS: Record<string, string> = {
  nv: "Melanocytic Nevus (Mole)",
  bkl: "Benign Keratosis",
  mel: "Melanoma",
  bcc: "Basal Cell Carcinoma",
  akiec: "Actinic Keratosis / Intraepithelial Carcinoma",
  scc: "Squamous Cell Carcinoma",
  vasc: "Vascular Lesion",
  df: "Dermatofibroma",
}

/** CHANGE: helper to show friendly names, fallback to key if unknown */
function pretty(key: string) {
  const k = key.trim().toLowerCase()
  return CLASS_LABELS[k] || key
}

type APIResp = {
  prediction: {
    class_id: number
    class_name: string
    display_name: string
    risk: "Low" | "Medium" | "High" | string
    description: string
    recommendation: string
    confidence: number
  }
  model_predictions: Record<string, { class_id: number; class_name: string; confidence: number }>
  class_probabilities: Record<string, number>
}

function RiskBadge({ risk }: { risk: string }) {
  const riskLower = risk.toLowerCase()
  const baseClasses = "inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border"

  if (riskLower === "high") {
    return (
      <span className={`${baseClasses} risk-high`}>
        <AlertTriangle className="w-4 h-4 mr-1" />
        High Risk
      </span>
    )
  } else if (riskLower === "medium") {
    return (
      <span className={`${baseClasses} risk-medium`}>
        <Info className="w-4 h-4 mr-1" />
        Medium Risk
      </span>
    )
  } else {
    return (
      <span className={`${baseClasses} risk-low`}>
        <CheckCircle className="w-4 h-4 mr-1" />
        Low Risk
      </span>
    )
  }
}

function ConfidenceBar({ confidence }: { confidence: number }) {
  const percentage = Math.round(confidence * 100)
  const getColor = () => {
    if (percentage >= 80) return "bg-green-500"
    if (percentage >= 60) return "bg-yellow-500"
    return "bg-red-500"
  }

  return (
    <div className="w-full">
      <div className="flex justify-between text-sm text-muted-foreground mb-1">
        <span>Confidence Level</span>
        <span>{percentage}%</span>
      </div>
      <div className="w-full bg-muted rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${getColor()}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  )
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<APIResp | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  function onChoose(f: File | null) {
    setFile(f)
    setResult(null)
    setErr(null)
    if (f) setPreview(URL.createObjectURL(f))
    else setPreview(null)
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setErr(null)
    setResult(null)
    if (!file) {
      setErr("Please select an image first.")
      return
    }

    const form = new FormData()
    form.append("file", file)
    setLoading(true)

    try {
      const res = await fetch("/api/predict", { method: "POST", body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.detail || "Analysis failed")
      setResult(data as APIResp)
    } catch (e: any) {
      setErr(e.message || "Unexpected error occurred")
    } finally {
      setLoading(false)
    }
  }

  const sortedProbs = useMemo(() => {
    if (!result) return []
    return Object.entries(result.class_probabilities || {}).sort((a, b) => b[1] - a[1])
  }, [result])

  function fmtPct01(x: number | undefined | null) {
    if (typeof x !== "number" || !isFinite(x)) return "–"
    return `${(Math.max(0, Math.min(1, x)) * 100).toFixed(1)}%`
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center">
                <Shield className="w-5 h-5 text-accent-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-foreground">HealthAware</h1>
                <p className="text-xs text-muted-foreground">Professional Skin Analysis</p>
              </div>
            </div>
            <div className="text-sm text-muted-foreground">Medical-Grade AI Detection</div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-1">
            <div className="medical-card p-6">
              <div className="text-center mb-6">
                <Camera className="w-12 h-12 text-accent mx-auto mb-3" />
                <h2 className="text-lg font-semibold text-card-foreground mb-2">Upload Skin Image</h2>
                <p className="text-sm text-muted-foreground">
                  Upload a clear, well-lit photo of the skin area for analysis
                </p>
              </div>

              <form onSubmit={onSubmit} className="space-y-4">
                <div className="border-2 border-dashed border-border rounded-lg p-6 text-center hover:border-accent/50 transition-colors">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => onChoose(e.target.files?.[0] ?? null)}
                    className="hidden"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <Upload className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground">Click to upload or drag and drop</p>
                    <p className="text-xs text-muted-foreground mt-1">PNG, JPG up to 10MB</p>
                  </label>
                </div>

                {preview && (
                  <div className="relative">
                    <img
                      src={preview || "/placeholder.svg"}
                      alt="Preview of uploaded skin image"
                      className="w-full h-48 object-cover rounded-lg border border-border"
                    />
                    <div className="absolute top-2 right-2 bg-background/80 backdrop-blur-sm rounded-full p-1">
                      <CheckCircle className="w-4 h-4 text-green-600" />
                    </div>
                  </div>
                )}

                <button
                  type="submit"
                  className={`w-full py-3 px-4 rounded-lg font-medium transition-all duration-200 medical-button ${
                    !file || loading ? "opacity-50 cursor-not-allowed" : "hover:shadow-md"
                  }`}
                  disabled={!file || loading}
                >
                  {loading ? (
                    <div className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-foreground mr-2"></div>
                      Analyzing...
                    </div>
                  ) : (
                    <div className="flex items-center justify-center">
                      <TrendingUp className="w-4 h-4 mr-2" />
                      Analyze Image
                    </div>
                  )}
                </button>
              </form>

              {err && (
                <div className="mt-4 p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
                  <div className="flex items-center">
                    <AlertTriangle className="w-4 h-4 text-destructive mr-2" />
                    <p className="text-sm text-destructive">{err}</p>
                  </div>
                </div>
              )}
            </div>

            {/* Disclaimer */}
            <div className="medical-card p-4 mt-6">
              <div className="flex items-start space-x-2">
                <Info className="w-5 h-5 text-accent mt-0.5 flex-shrink-0" />
                <div className="text-xs text-muted-foreground">
                  <p className="font-medium mb-1">Medical Disclaimer</p>
                  <p>
                    This tool is for informational purposes only and should not replace professional medical advice.
                    Always consult a dermatologist for proper diagnosis.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2">
            {result ? (
              <div className="space-y-6">
                {/* Primary Result */}
                <div className="medical-card p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-xl font-semibold text-card-foreground mb-1">
                        {result.prediction.display_name}
                      </h3>
                      <p className="text-muted-foreground text-sm">Primary Classification</p>
                    </div>
                    <RiskBadge risk={result.prediction.risk} />
                  </div>

                  <div className="space-y-4">
                    <div>
                      <h4 className="font-medium text-card-foreground mb-2">Description</h4>
                      <p className="text-sm text-muted-foreground leading-relaxed">{result.prediction.description}</p>
                    </div>

                    <div>
                      <h4 className="font-medium text-card-foreground mb-2">Recommendation</h4>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        {result.prediction.recommendation}
                      </p>
                    </div>

                    <ConfidenceBar confidence={result.prediction.confidence} />
                  </div>
                </div>

                {/* Model Predictions */}
                <div className="medical-card p-6">
                  <div className="flex items-center mb-4">
                    <FileText className="w-5 h-5 text-accent mr-2" />
                    <h3 className="text-lg font-semibold text-card-foreground">Model Analysis</h3>
                  </div>
                  <div className="space-y-3">
                    {Object.entries(result.model_predictions || {}).map(([model, pred]) => (
                      <div key={model} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                        <div>
                          <p className="font-medium text-sm text-card-foreground">{model}</p>
                          {/* CHANGE: show friendly name for each model's class */}
                          <p className="text-xs text-muted-foreground">{pretty(pred.class_name)}</p>
                        </div>
                        <div className="text-right">
                          <p className="font-semibold text-sm text-card-foreground">
                            {(pred.confidence * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Probability Distribution */}
                <div className="medical-card p-6">
                  <h3 className="text-lg font-semibold text-card-foreground mb-4">Classification Probabilities</h3>
                  <div className="space-y-3">
                    {sortedProbs.slice(0, 5).map(([label, p]) => {
                      const pct = p * 100
                      return (
                        <div key={label} className="space-y-1">
                          <div className="flex justify-between text-sm">
                            {/* CHANGE: show friendly label here */}
                            <span className="text-card-foreground font-medium">{pretty(label)}</span>
                            <span className="text-muted-foreground">{pct.toFixed(2)}%</span>
                          </div>
                          <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                            <div
                              className="h-2 bg-accent rounded-full transition-all duration-500"
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            ) : (
              <div className="medical-card p-12 text-center">
                <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
                  <Camera className="w-8 h-8 text-muted-foreground" />
                </div>
                <h3 className="text-lg font-semibold text-card-foreground mb-2">Ready for Analysis</h3>
                <p className="text-muted-foreground">Upload a skin image to begin the AI-powered analysis process</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
