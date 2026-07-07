# Fantasy Copilot

## Register

product — authenticated dashboard/tool. Design serves the task: reading recommendations, approving moves, analyzing trades. Users are in a weekly workflow, not browsing a brand.

## Users & Purpose

ESPN fantasy football managers (initially the owner's own league, then league-mates as a small multi-tenant SaaS). They visit 2-4 times a week during the season: Tuesday (waivers), Thursday/Sunday (lineups), anytime (trade ideas, chat). Mostly evening couch sessions, often on a phone. The job: "tell me the best move and let me act on it fast" — the app decides, the user approves.

## Brand personality

Sharp, confident, a little competitive. A quant analyst who talks like a league-mate: numbers first, plain English second, no hedging. Not corporate, not childish-sporty.

## Anti-references

- ESPN's own cluttered fantasy UI (ads, tabs, ten fonts).
- Generic SaaS admin templates (identical KPI card grids, purple gradients).
- Anything that hides the numbers behind decoration — the numbers ARE the product.

## Accessibility

Dark theme is the committed default (evening use). Contrast must hold at 4.5:1 for body text; status must never be color-only (pills carry text). Touch targets ≥44px on mobile — approvals happen on phones.

## Strategic design principles

1. One screen, scannable top to bottom: this week's lineup, then moves to approve, then context (standings, news, chat).
2. Every recommendation shows its "why" (points gained, confidence) inline.
3. State is always visible: building, stale, updated-when, connected/not.
4. Zero-dependency vanilla HTML/CSS/JS stays — no framework, no external assets; the two static files ARE the frontend.
