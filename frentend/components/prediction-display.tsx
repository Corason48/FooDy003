import { Card } from "@/components/ui/card"

interface PredictionDisplayProps {
  prediction: {
    predicted_class: string
    confidence: number
    probabilities?: Record<string, number>
  }
  model: string
}

export default function PredictionDisplay({ prediction, model }: PredictionDisplayProps) {
  const confidencePercent = Math.round(prediction.confidence * 100)
  const confidenceColor =
    confidencePercent >= 80 ? "text-green-400" : confidencePercent >= 60 ? "text-yellow-400" : "text-orange-400"

  return (
    <Card className="p-8 border-border bg-card overflow-hidden">
      <div className="space-y-6">
        {/* Main Prediction */}
        <div>
          <p className="text-sm text-muted-foreground mb-2">Prediction Result</p>
          <div className="flex items-end gap-4">
            <div>
              <p className="text-4xl font-bold text-foreground capitalize">{prediction.predicted_class}</p>
              <p className="text-sm text-muted-foreground mt-1">Using {model.toUpperCase()}</p>
            </div>
            <div className={`text-3xl font-bold ${confidenceColor}`}>{confidencePercent}%</div>
          </div>
        </div>

        {/* Confidence Bar */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <p className="text-sm font-medium text-foreground">Confidence Score</p>
            <p className="text-xs text-muted-foreground">{confidencePercent}%</p>
          </div>
          <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-500"
              style={{ width: `${confidencePercent}%` }}
            />
          </div>
        </div>

        {/* Class Probabilities */}
        {prediction.probabilities && (
          <div>
            <p className="text-sm font-medium text-foreground mb-3">Class Probabilities</p>
            <div className="space-y-2">
              {Object.entries(prediction.probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([className, prob]) => (
                  <div key={className} className="flex items-center gap-3">
                    <span className="text-sm text-muted-foreground w-20 capitalize">{className}</span>
                    <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary/60 transition-all"
                        style={{ width: `${Math.round(prob * 100)}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium text-foreground w-12 text-right">
                      {Math.round(prob * 100)}%
                    </span>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}
