// src/app/api/predict/route.ts
import { NextResponse, type NextRequest } from "next/server"

// CHANGE: ensure we run in Node.js runtime for FormData/file streaming
export const runtime = "nodejs"
// CHANGE: API responses should never be cached
export const dynamic = "force-dynamic"

const BACKEND = process.env.ML_BACKEND_URL // e.g., http://127.0.0.1:8000
const TIMEOUT_MS = Number(process.env.ML_TIMEOUT_MS || 20000)

export async function POST(request: NextRequest) {
  try {
    if (!BACKEND) {
      return NextResponse.json(
        { detail: "Server misconfigured: ML_BACKEND_URL is missing" },
        { status: 500 }
      )
    }

    const inForm = await request.formData()
    const file = inForm.get("file") as File | null
    // CHANGE: optional personalization fields (uncomment on the client to send):
    const meta = (inForm.get("meta") as string) || "" // JSON string if sent

    if (!file) {
      return NextResponse.json({ detail: "No file provided" }, { status: 400 })
    }

    // Rebuild FormData for FastAPI
    const outForm = new FormData()
    outForm.append("file", file, (file as any).name || "upload.jpg")
    if (meta) outForm.append("meta", meta)              // CHANGE: optional JSON string passthrough

    // Timeout wrapper
    const controller = new AbortController()
    const to = setTimeout(() => controller.abort(), TIMEOUT_MS)

    const res = await fetch(`${BACKEND}/predict`, {
      method: "POST",
      body: outForm,
      signal: controller.signal,
    })

    clearTimeout(to)

    const text = await res.text()
    let data: any = text
    try {
      data = JSON.parse(text)
    } catch {
      // non-JSON error body — keep `text` as-is
    }

    if (!res.ok) {
      return NextResponse.json(
        { detail: `ML service error (${res.status}): ${typeof data === "string" ? data : data?.detail || res.statusText}` },
        { status: res.status }
      )
    }

    // Expecting the normalized shape from FastAPI now
    return NextResponse.json(data, { status: 200 })
  } catch (err: any) {
    const msg = err?.name === "AbortError" ? "Upstream ML request timed out" : (err?.message || "Internal server error")
    return NextResponse.json({ detail: msg }, { status: 500 })
  }
}
