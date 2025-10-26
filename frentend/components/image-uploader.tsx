"use client"

import type React from "react"

import { useRef, useState } from "react"
import { Button } from "@/components/ui/button"

interface ImageUploaderProps {
  onImageUpload: (file: File) => void
  loading: boolean
}

export default function ImageUploader({ onImageUpload, loading }: ImageUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragActive, setDragActive] = useState(false)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    const files = e.dataTransfer.files
    if (files && files[0]) {
      handleFile(files[0])
    }
  }

  const handleFile = (file: File) => {
    if (file.type.startsWith("image/")) {
      onImageUpload(file)
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  return (
    <div
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-all cursor-pointer ${
        dragActive ? "border-primary bg-primary/5" : "border-border bg-muted/20 hover:border-primary/50"
      }`}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="hidden"
        disabled={loading}
      />

      <div className="space-y-3">
        <div className="text-3xl">📸</div>
        <div>
          <p className="font-semibold text-foreground">Drag & drop your image</p>
          <p className="text-sm text-muted-foreground">or click to browse</p>
        </div>
      </div>

      <Button onClick={() => fileInputRef.current?.click()} disabled={loading} className="mt-4 w-full">
        {loading ? "Processing..." : "Select Image"}
      </Button>
    </div>
  )
}
