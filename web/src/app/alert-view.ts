import { Component, computed, effect, inject, input, signal } from '@angular/core'
import { DecimalPipe, KeyValuePipe } from '@angular/common'
import { AlertDetail, Answer, Api, Claim } from './api'
import { GraphView } from './graph-view'

const CHECK_LABEL: Record<Claim['check'], string> = {
  ok: 'cited',
  number_mismatch: 'number differs from source',
  bad_citation: 'cites something it was not given',
  no_citation: 'no source',
}

@Component({
  selector: 'app-alert-view',
  imports: [GraphView, DecimalPipe, KeyValuePipe],
  templateUrl: './alert-view.html',
  styleUrl: './alert-view.css',
})
export class AlertView {
  tx = input.required<number>()
  private api = inject(Api)
  detail = signal<AlertDetail | null>(null)
  answer = signal<Answer | null>(null)
  asking = signal(false)
  error = signal('')
  hovered = signal<string[]>([])
  checkLabel = CHECK_LABEL

  facts = computed(() => Object.entries(this.detail()?.facts ?? {}))
  unstable = computed(() => (this.detail()?.explanation.stability ?? 1) < 0.5)
  // a hovered citation lights up the fact row and, for an edge, the edge in the graph
  highlightIds = computed(() => {
    const d = this.detail()
    if (!d) return []
    return this.hovered().flatMap((id) => {
      const m = id.match(/^edge_(\d+)$/)
      if (!m) return []
      const e = d.explanation.edges[Number(m[1]) - 1]
      return e ? [`e${e.source}-${e.target}`] : []
    })
  })

  constructor() {
    effect(() => {
      const tx = this.tx()
      this.detail.set(null)
      this.answer.set(null)
      this.error.set('')
      this.api.alert(tx).subscribe((d) => this.detail.set(d))
    })
  }

  ask() {
    this.asking.set(true)
    this.error.set('')
    this.api.ask(this.tx()).subscribe({
      next: (a) => {
        this.answer.set(a)
        this.asking.set(false)
      },
      error: () => {
        this.error.set('The language model is not reachable. Is Ollama running?')
        this.asking.set(false)
      },
    })
  }

  isHovered(id: string) {
    return this.hovered().includes(id)
  }
}
