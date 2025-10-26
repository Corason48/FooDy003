export default function Header() {
  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center">
              <span className="text-primary-foreground font-bold text-lg">🤖</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">ImageClassify</h1>
              <p className="text-sm text-muted-foreground">Deep Learning Model Comparison</p>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
