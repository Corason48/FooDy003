"use client"

interface ModelSelectorProps {
  selectedModel: "tinyvgg" | "googlenet" | "vit"
  onModelChange: (model: "tinyvgg" | "googlenet" | "vit") => void
}

const models = [
  {
    id: "tinyvgg",
    name: "TinyVGG",
    description: "Lightweight CNN",
    icon: "⚡",
  },
  {
    id: "googlenet",
    name: "GoogleNet",
    description: "Transfer Learning ," ,
    icon: "🔍",
  },
  {
    id: "vit",
    name: "Vision Transformer",
    description: "Transformer-based",
    icon: "✨",
  },
]

export default function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  return (
    <div className="space-y-3">
      {models.map((model) => (
        <button
          key={model.id}
          onClick={() => onModelChange(model.id as "tinyvgg" | "googlenet" | "vit")}
          className={`w-full p-4 rounded-lg border-2 transition-all text-left ${
            selectedModel === model.id
              ? "border-primary bg-primary/10"
              : "border-border bg-muted/30 hover:border-primary/50"
          }`}
        >
          <div className="flex items-start gap-3">
            <span className="text-2xl">{model.icon}</span>
            <div className="flex-1">
              <p className="font-semibold text-foreground">{model.name}</p>
              <p className="text-sm text-muted-foreground">{model.description}
               {model.id =="googlenet" && <span className="text-green-700"> Recomanded</span>}
              </p>
            </div>
            {selectedModel === model.id && (
              <div className="w-5 h-5 rounded-full bg-primary flex items-center justify-center">
                <span className="text-primary-foreground text-xs">✓</span>
              </div>
            )}
          </div>
        </button>
      ))}
    </div>
  )
}
