import { Component, effect, inject, signal } from '@angular/core'
import { FormsModule } from '@angular/forms'
import { Api, Community, GlobalAnswer } from './api'

@Component({
  selector: 'app-communities-view',
  imports: [FormsModule],
  template: `
    <section class="ask panel">
      <h3>Ask the whole corpus</h3>
      <p class="muted">
        GraphRAG global search: every community report of the chosen level answers in parallel, the useful partial
        answers are merged. Plain RAG can't do this; no single passage holds the answer.
      </p>
      <form (ngSubmit)="ask()">
        <input name="q" [(ngModel)]="question" placeholder="What laundering techniques show up across these cases?" />
        <select name="level" [ngModel]="level()" (ngModelChange)="level.set(+$event)">
          <option [value]="0">C0 (root)</option>
          <option [value]="1">C1</option>
          <option [value]="2">C2</option>
        </select>
        <button [disabled]="asking() || !question">{{ asking() ? 'Searching…' : 'Ask' }}</button>
      </form>
      @if (answer(); as a) {
        <div class="global-answer">{{ a.answer }}</div>
        <p class="muted">{{ a.map_calls }} map calls, {{ a.points }} useful points.</p>
      }
    </section>

    <section class="list">
      <h3>{{ reports().length }} communities at level C{{ level() }}, most important first</h3>
      @for (c of reports(); track c.community) {
        <article class="panel">
          <header>
            <h4>{{ c.title }}</h4>
            <span class="tag">{{ c.size }} entities</span>
            <span class="tag rating">importance {{ c.rating }}/10</span>
          </header>
          <p>{{ c.summary }}</p>
          <details>
            <summary>{{ c.findings.length }} findings</summary>
            <ul>
              @for (f of c.findings; track $index) {
                <li>
                  <b>{{ f.summary }}.</b> {{ f.explanation }}
                </li>
              }
            </ul>
          </details>
        </article>
      }
    </section>
  `,
  styles: `
    :host {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1.3fr);
      gap: 16px;
      padding: 16px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px 14px;
    }
    h3 {
      font-size: 14px;
      margin: 0 0 8px;
    }
    h4 {
      font-size: 14px;
      margin: 0;
      flex: 1;
    }
    .ask {
      position: sticky;
      top: 16px;
    }
    form {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    input {
      flex: 1;
      min-width: 200px;
      padding: 7px 9px;
      border: 1px solid var(--line);
      border-radius: 4px;
      font: 13px var(--sans);
    }
    select {
      padding: 6px;
      border: 1px solid var(--line);
      border-radius: 4px;
      font: 13px var(--sans);
    }
    button {
      background: var(--blue);
      color: white;
      border: 0;
      border-radius: 4px;
      padding: 7px 14px;
      font: 500 13px var(--sans);
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.6;
    }
    .global-answer {
      white-space: pre-wrap;
      font-size: 14px;
      line-height: 1.55;
      margin-top: 12px;
    }
    .list {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    header {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .tag {
      font-size: 11px;
      background: #f1f5f9;
      border-radius: 3px;
      padding: 2px 6px;
      color: var(--muted);
    }
    .rating {
      background: #eff6ff;
      color: var(--blue);
    }
    p {
      font-size: 13px;
      line-height: 1.5;
    }
    li {
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 4px;
    }
    details summary {
      font-size: 12px;
      color: var(--blue);
      cursor: pointer;
    }
    .muted {
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 900px) {
      :host {
        grid-template-columns: 1fr;
      }
      .ask {
        position: static;
      }
    }
  `,
})
export class CommunitiesView {
  private api = inject(Api)
  level = signal(0)
  reports = signal<Community[]>([])
  question = ''
  answer = signal<GlobalAnswer | null>(null)
  asking = signal(false)

  constructor() {
    effect(() => this.api.communities(this.level()).subscribe((r) => this.reports.set(r)))
  }

  ask() {
    this.asking.set(true)
    this.api.global(this.question, this.level()).subscribe({
      next: (a) => {
        this.answer.set(a)
        this.asking.set(false)
      },
      error: () => this.asking.set(false),
    })
  }
}
