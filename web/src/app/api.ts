import { HttpClient } from '@angular/common/http'
import { Injectable, inject } from '@angular/core'

export const API = 'http://localhost:8000'

export interface AlertSummary {
  tx: number
  timestep: number
  rf: number
  gnn: number
  shapes: string[]
  stability: number
  neighbors: { inputs: number; outputs: number; n_1hop: number; risky_1hop: number; n_2hop: number; risky_2hop: number }
}

export interface GraphNode {
  tx: number
  rf: number
  alert: boolean
}

export interface GraphEdge {
  source: number
  target: number
  explanation: boolean
}

export interface AlertDetail extends AlertSummary {
  facts: Record<string, string>
  explanation: {
    stability: number
    neighborhood: number
    p_full: number
    p_without_top: number
    p_only_top: number
    edges: { source: number; target: number; weight: number }[]
  }
  graph: { nodes: GraphNode[]; edges: GraphEdge[] }
}

export interface Claim {
  text: string
  cites: string[]
  check: 'ok' | 'no_citation' | 'bad_citation' | 'number_mismatch'
}

export interface Answer {
  summary: string
  claims: Claim[]
  caveats: string[]
  evidence: Record<string, string>
}

export interface Community {
  community: number
  level: number
  size: number
  title: string
  summary: string
  rating: number
  findings: { summary: string; explanation: string }[]
}

export interface GlobalAnswer {
  answer: string
  map_calls: number
  points: number
}

@Injectable({ providedIn: 'root' })
export class Api {
  private http = inject(HttpClient)

  alerts() {
    return this.http.get<AlertSummary[]>(`${API}/alerts`)
  }

  alert(tx: number) {
    return this.http.get<AlertDetail>(`${API}/alerts/${tx}`)
  }

  ask(tx: number) {
    return this.http.post<Answer>(`${API}/alerts/${tx}/ask`, {})
  }

  communities(level: number) {
    return this.http.get<Community[]>(`${API}/communities`, { params: { level } })
  }

  global(question: string, level: number) {
    return this.http.post<GlobalAnswer>(`${API}/global`, { question, level })
  }
}
