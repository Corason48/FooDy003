import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File
    const model = formData.get("model") as string

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"

    // Create FormData for FastAPI backend (file only)
    const backendFormData = new FormData()
    backendFormData.append("file", file)

    const response = await fetch(`${backendUrl}/predict?model=${encodeURIComponent(model)}`, {
      method: "POST",
      body: backendFormData,
    })

    if (!response.ok) {
      const error = await response.json()
      return NextResponse.json({ error: error.detail || "Prediction failed" }, { status: response.status })
    }

    const data = await response.json()

    // Transform FastAPI response to match frontend expectations
    return NextResponse.json({
      class: data.predicted_class,
      confidence: data.confidence,
      confidence_percentage: data.confidence_percentage,
      probabilities: data.all_predictions.reduce((acc: any, pred: any) => {
        acc[pred.class] = pred.confidence
        return acc
      }, {}),
      model: data.model,
      image_name: data.image_name,
    })
  } catch (error) {
    console.error("Prediction error:", error)
    return NextResponse.json(
      { error: "Failed to connect to backend. Make sure FastAPI server is running." },
      { status: 500 },
    )
  }
}
