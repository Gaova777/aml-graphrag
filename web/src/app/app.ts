import { Component, computed, inject, signal } from '@angular/core'
import { DecimalPipe } from '@angular/common'
import { AlertSummary, Api } from './api'
import { AlertView } from './alert-view'
import { CommunitiesView } from './communities-view'

@Component({
  selector: 'app-root',
  imports: [AlertView, CommunitiesView, DecimalPipe],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  private api = inject(Api)
  tab = signal<'alerts' | 'corpus'>('alerts')
  alerts = signal<AlertSummary[]>([])
  selected = signal<number | null>(null)
  filter = signal('')
  shown = computed(() => {
    const f = this.filter()
    return f ? this.alerts().filter((a) => a.shapes.includes(f)) : this.alerts()
  })

  constructor() {
    this.api.alerts().subscribe((a) => {
      this.alerts.set(a)
      if (a.length) this.selected.set(a[0].tx)
    })
  }
}
