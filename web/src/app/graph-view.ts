import { Component, ElementRef, OnDestroy, effect, input, viewChild } from '@angular/core'
import cytoscape from 'cytoscape'
import { GraphEdge, GraphNode } from './api'

// risk colour of a node from its alert score
function risk(score: number) {
  if (score >= 0.7) return '#dc2626'
  if (score >= 0.4) return '#f59e0b'
  return '#94a3b8'
}

@Component({
  selector: 'app-graph-view',
  template: `<div #box class="box"></div>`,
  styles: `
    :host {
      display: block;
      height: 100%;
    }
    .box {
      width: 100%;
      height: 100%;
    }
  `,
})
export class GraphView implements OnDestroy {
  nodes = input.required<GraphNode[]>()
  edges = input.required<GraphEdge[]>()
  highlight = input<string[]>([])
  private box = viewChild.required<ElementRef<HTMLDivElement>>('box')
  private cy?: cytoscape.Core

  constructor() {
    effect(() => this.draw(this.nodes(), this.edges()))
    effect(() => {
      const ids = new Set(this.highlight())
      this.cy?.elements().removeClass('focus')
      this.cy
        ?.elements()
        .filter((el) => ids.has(el.id()))
        .addClass('focus')
    })
  }

  private draw(nodes: GraphNode[], edges: GraphEdge[]) {
    this.cy?.destroy()
    this.cy = cytoscape({
      container: this.box().nativeElement,
      elements: [
        ...nodes.map((n) => ({
          data: { id: `n${n.tx}`, label: n.alert ? `${n.tx}` : '', color: risk(n.rf), alert: n.alert },
        })),
        ...edges.map((e) => ({
          data: { id: `e${e.source}-${e.target}`, source: `n${e.source}`, target: `n${e.target}` },
          classes: e.explanation ? 'explained' : '',
        })),
      ],
      layout: { name: 'cose', animate: false, nodeRepulsion: () => 9000, idealEdgeLength: () => 40 },
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            width: 12,
            height: 12,
            label: 'data(label)',
            'font-size': 10,
            color: '#0f172a',
            'text-valign': 'top',
            'text-margin-y': -4,
          },
        },
        { selector: 'node[?alert]', style: { width: 22, height: 22, 'border-width': 3, 'border-color': '#0f172a' } },
        {
          selector: 'edge',
          style: {
            width: 1,
            'line-color': '#cbd5e1',
            'target-arrow-color': '#cbd5e1',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 0.6,
            'curve-style': 'bezier',
          },
        },
        { selector: 'edge.explained', style: { width: 3.5, 'line-color': '#2563eb', 'target-arrow-color': '#2563eb' } },
        {
          selector: '.focus',
          style: {
            'line-color': '#7c3aed',
            'target-arrow-color': '#7c3aed',
            width: 6,
            'border-width': 4,
            'border-color': '#7c3aed',
          },
        },
      ],
      wheelSensitivity: 0.2,
    })
  }

  ngOnDestroy() {
    this.cy?.destroy()
  }
}
