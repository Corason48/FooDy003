"use client"

import Header from "@/components/header"
import ImageUploader from "@/components/image-uploader"
import ModelSelector from "@/components/model-selector"
import PredictionDisplay from "@/components/prediction-display"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { useState } from "react"


export default function Home() {
  const [selectedModel, setSelectedModel] = useState<"tinyvgg" | "googlenet" | "vit">("tinyvgg")
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [prediction, setPrediction] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleImageUpload = (file: File) => {
    setError(null)
    setPrediction(null)

    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string)
    }
    reader.readAsDataURL(file)

    // Store file for later prediction
    setUploadedFile(file)
  }

  const handlePredict = async () => {
    if (!uploadedFile) return

    setError(null)
    setPrediction(null)
    setLoading(true)

    try {
      const formData = new FormData()
      formData.append("file", uploadedFile)
      formData.append("model", selectedModel)

      console.log("[v0] Sending prediction request with model:", selectedModel)
      const response = await fetch(`http://127.0.0.1:8000/predict?model=${encodeURIComponent(selectedModel)}`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Prediction failed")
      }

      const data = await response.json()
      console.log("[v0] Prediction response:", data)
      setPrediction(data)
    } catch (err) {
      console.error("[v0] Prediction error:", err)
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setUploadedImage(null)
    setUploadedFile(null)
    setPrediction(null)
    setError(null)
  }

  return (
    <main className="min-h-screen bg-background">
      <Header />

      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Controls */}
          <div className="lg:col-span-1 space-y-6">
            <Card className="p-6 border-border bg-card">
              <h2 className="text-lg font-semibold mb-4 text-foreground">Model Selection</h2>
              <ModelSelector selectedModel={selectedModel} onModelChange={setSelectedModel} />
            </Card>

            <Card className="p-6 border-border bg-card">
              <h2 className="text-lg font-semibold mb-4 text-foreground">Upload Image</h2>
              <ImageUploader onImageUpload={handleImageUpload} loading={loading} />
            </Card>

            {uploadedImage && !prediction && (
              <Button
                onClick={handlePredict}
                disabled={loading}
                className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-6 text-lg"
              >
                {loading ? "Analyzing..." : `Predict with ${selectedModel.toUpperCase()}`}
              </Button>
            )}
          </div>

          {/* Right Column - Preview & Results */}
          <div className="lg:col-span-2 space-y-6">
            {uploadedImage && (
              <Card className="p-6 border-border bg-card overflow-hidden">
                <h2 className="text-lg font-semibold mb-4 text-foreground">Image Preview</h2>
                <div className="relative w-full aspect-square rounded-lg overflow-hidden bg-muted">
                  <img
                    src={uploadedImage || "/placeholder.svg"}
                    alt="Uploaded preview"
                    className="w-full h-full object-cover"
                  />
                </div>
              </Card>
            )}

            {error && (
              <Card className="p-6 border-destructive bg-destructive/10">
                <p className="text-destructive font-medium">{error}</p>
              </Card>
            )}

            {loading && (
              <Card className="p-8 border-border bg-card">
                <div className="flex flex-col items-center justify-center space-y-4">
                  <div className="w-12 h-12 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                  <p className="text-muted-foreground">Analyzing image with {selectedModel.toUpperCase()}...</p>
                </div>
              </Card>
            )}

            {prediction && !loading && (
              <>
                <PredictionDisplay prediction={prediction} model={selectedModel} />
                <Button onClick={handleReset} variant="outline" className="w-full bg-transparent">
                  Classify Another Image
                </Button>
              </>
            )}

            {!uploadedImage && !prediction && !loading && (
              <Card className="p-12 border-border bg-card border-dashed flex items-center justify-center min-h-96">
                <div className="text-center">
                  <p className="text-muted-foreground text-lg">Upload an image to get started</p>
                  <p className="text-muted-foreground text-sm mt-2">Supports JPG, PNG, and WebP formats</p>
                </div>
              </Card>
            )}
          </div>
        </div>
      </div>
    </main>
  )
}
