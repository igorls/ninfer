---
name: NInfer Dual-Purpose Decode
description: A mass-spec printout of closed-set readout and leftover mass.
colors:
  teal: "#1a6b64"
  teal-deep: "#0e433e"
  brick: "#9c3d32"
  paper: "#e4e6df"
  paper-deep: "#d7dacf"
  paper-wash: "#eceee8"
  bench: "#6d736c"
  bench-glow: "#8a9188"
  ink: "#1c1e1a"
  ink-soft: "#3a4038"
  grid: "#c2c7bb"
  hole: "#cfd3c8"
typography:
  display:
    fontFamily: "Overpass Condensed, Overpass, Source Sans 3, sans-serif"
    fontSize: "clamp(1.85rem, 3.6vw, 2.55rem)"
    fontWeight: 700
    lineHeight: 0.95
    letterSpacing: "-0.03em"
  headline:
    fontFamily: "Overpass Condensed, Overpass, sans-serif"
    fontSize: "1.35rem"
    fontWeight: 700
    letterSpacing: "-0.02em"
  title:
    fontFamily: "Overpass Condensed, Overpass, sans-serif"
    fontSize: "1rem"
    fontWeight: 700
    letterSpacing: "0.02em"
  body:
    fontFamily: "Source Sans 3, sans-serif"
    fontSize: "1.0625rem"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "Overpass, sans-serif"
    fontSize: "0.92rem"
    fontWeight: 700
    letterSpacing: "0.02em"
  stamp:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: 1.7
    letterSpacing: "0.04em"
  plate:
    fontFamily: "Overpass Condensed, Overpass, sans-serif"
    fontSize: "22px"
    fontWeight: 700
rounded:
  none: "0"
spacing:
  sprocket: "28px"
  inner: "36px"
  plate: "12px"
  sheet-y: "28px"
  pair: "40px"
  notes: "36px"
components:
  sheet:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "28px 28px 64px"
    width: "920px"
  plate:
    backgroundColor: "{colors.paper-deep}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "12px"
  stamp:
    textColor: "{colors.ink-soft}"
    typography: "{typography.stamp}"
  threshold-thumb:
    backgroundColor: "{colors.brick}"
    rounded: "{rounded.none}"
    height: "22px"
    width: "14px"
  glossary-term:
    textColor: "{colors.teal-deep}"
    typography: "{typography.title}"
  gate-typed:
    textColor: "{colors.teal-deep}"
  gate-think:
    textColor: "{colors.brick}"
  link:
    textColor: "{colors.teal-deep}"
---

# Design System: NInfer Dual-Purpose Decode

## Overview

**Creative North Star: "The Mass-Spec Printout"**

This system is a sheet of thermal plotter stock under cool fluorescent lab light. Grey-green paper, not cream. Typed option-tokens are teal peaks on a shared-prefix baseline; leftover vocabulary is a brick hatch; a second ink trace lights only when that remainder is high. The page is one continuous printout, not a dashboard and not a dark AI explainer.

The NInfer Supervisor workbench is a separate surface. It does not share this palette, type pairing, radius, or layout. Do not mix forest-green Segoe workbench tokens into this explainer world, and do not restyle the workbench as plotter stock.

Personality is instrument-literal: dense notes after the plate, one control, no chrome. The spectrum is the argument. Caption, glossary, and method notes stay on the same perforated sheet.

**Key Characteristics:**
- Cool-fluorescent grey-green paper on an olive lab bench
- Three inks only: teal peaks, brick remainder, ink thinking-trace
- Overpass Condensed on the plate; Source Sans 3 in the notes; JetBrains Mono on the stamp and inline tokens
- Sprocket holes and dashed perforation rule the sheet; square stock, no cards
- One threshold cursor is the only control

## Colors

Cool fluorescent light on grey-green thermal stock: muted paper neutrals, one teal pen, one brick hatch.

### Primary
- **Plotter Teal**: Solid fill for typed peaks, the shared-prefix bar, text selection, and the focus ring. This is the closed-set readout.
- **Teal Deep**: Peak and axis captions on the plate, glossary terms, colophon links, and the typed-gate sentence. Darker so lettering holds on the grid.

### Secondary
- **Remainder Brick**: 45° hatch and stroke for `outside_mass`, the threshold thumb, remainder labels, and the think-gate sentence. Brick is leftover intensity, never a primary fill for typed peaks.

### Neutral
- **Thermal Paper**: Sheet body and the band between the two traces.
- **Paper Deep**: Plate ground under the 28px grid.
- **Paper Wash**: Fluorescent highlight along the top 48px of the sheet.
- **Lab Bench / Bench Glow**: Page surround; a faint cooler wash at the top of the bench, not a hero gradient.
- **Plotter Ink**: Axes, thinking-trace, hairline rules, and body copy.
- **Ink Soft**: Stamp metadata, axis numerals, gate note, colophon.
- **Grid**: 1px plate lattice on 28px cells.
- **Sprocket Hole**: Punched-hole fill in the 28px tractor-feed margins.

### Named Rules
**The Three-Ink Rule.** Teal is typed mass. Brick is remainder. Ink is the thinking trace. Do not add a fourth plot color.

**The Cool Paper Rule.** Stock is grey-green thermal paper on an olive bench. Never cream stationery, never charcoal canvas, never a white marketing card.

## Typography

**Display Font:** Overpass Condensed (with Overpass, then Source Sans 3)
**Body Font:** Source Sans 3 (with generic sans-serif)
**Label/Mono Font:** Overpass for the control label; JetBrains Mono for the instrument stamp, synthetic-run line, and inline `code`

**Character:** Condensed plate lettering sits tight on the grid; Source Sans 3 carries the lab notes at reading width; mono is the instrument’s own stamp, not a code-theme overlay.

### Hierarchy
- **Display** (700, clamp 1.85rem–2.55rem, line-height 0.95, tracking −0.03em): Page title only, capped near 18ch (12ch at the 720px fold).
- **Headline** (700, 1.35rem, tracking −0.02em): Notes section titles on the same sheet.
- **Title** (700, 1rem, tracking 0.02em): Glossary terms in teal-deep.
- **Body** (400, 1.0625rem, line-height 1.55, measure 68ch): Notes, lists, and definition values. Lede is slightly larger (1.12rem / 1.4). Caption sits at 1.05rem / 1.45.
- **Label** (Overpass 700, 0.92rem, tracking 0.02em): The threshold control label. Gate and output numerals stay Source Sans 3 at 0.95rem / 600.
- **Stamp** (JetBrains Mono 400/500, 0.8125rem, tracking 0.04em, line-height 1.7): Instrument header. Stamp `strong` is 500 in ink, not a second color.
- **Plate** (Overpass Condensed 700, 22px): Peak names on the spectrum. Axis and trace labels are the same family at 16px (600 / 700).

### Named Rules
**The Plate vs Notes Rule.** Overpass Condensed letters the plate, titles, and glossary terms. Source Sans 3 writes the notes. JetBrains Mono is stamp, synthetic disclaimer, and token names — never paragraph copy.

## Layout

One centered sheet (max 920px) on the bench, with 36px top / 72px bottom margin. Tractor-feed columns are 28px; inner notes inset another 36px. The plate uses a 28px square grid. Notes measure 68ch. A two-column pair (gap 28×40px) holds The gate / Isolation; the glossary is an 11.5rem / 1fr term-definition grid.

At 720px the sheet goes full-bleed, sprocket columns shrink to 20px, the pair and glossary stack, and document order becomes stamp → spectrum → title → lede → control → notes so the plot still owns the fold. Print drops the bench, the shadow, and the live control.

### Named Rules
**The Continuous Sheet Rule.** Stamp, spectrum, caption, threshold, then notes on one perforated strip. No card grid, no three-up features, no separate hero chrome.

## Elevation & Depth

Depth is physical paper on a bench, not stacked UI. Inks sit in the stock. The plate is a darker tonal well with a 1px ink frame, not a floating card.

### Shadow Vocabulary
- **Sheet on bench** (`box-shadow: 0 18px 40px rgb(20 24 18 / 0.28), 0 0 0 1px rgb(28 30 26 / 0.12)`): Ambient lift of the printout only. Removed at 720px and in print.

### Named Rules
**The Bench, Not the Card Rule.** The only shadow is the sheet on the bench. Do not elevate notes, glossary rows, or the plate as separate cards.

## Shapes

Square thermal stock. No corner radius on the sheet, plate, thumb, or rules. Sprocket holes are 4.5px circular punches on a 28px pitch, flanked by 1px dashed ink perforation. Axes and hairlines are 1–1.2px ink. Peaks are filled gaussian lobes; remainder is a hatched blob, not a bar chart. The threshold thumb is a 14×22px brick rectangle with a 1px ink edge on a 3px ink track.

### Named Rules
**The Square Stock Rule.** Radius is 0. Punches are circular holes. Never 12–14px dashboard rounding.

## Components

Instrument parts on one sheet, not a widget kit. There is no button, chip, or nav.

### Threshold cursor
- **Shape:** Square thumb (14×22px, radius 0) on a 3px ink track; control row max 540px
- **Primary:** Brick thumb, ink border, ink track
- **Hover / Focus:** Pointer on the range; `:focus-visible` is a 3px teal ring, offset 3px, on any focusable control
- **Readout:** Output in teal-deep; gate sentence teal-deep when typed, brick when think would fire

### Cards / Containers
- **Corner Style:** Square (0)
- **Background:** Thermal paper; plate uses paper-deep plus the 28px grid
- **Shadow Strategy:** Sheet-on-bench only (see Elevation)
- **Border:** 1px solid ink around the plate; dashed perforation on sprocket columns; full-width ink rules for stamp, refuse band, and colophon
- **Internal Padding:** 28px / 64px on the sheet; 12px in the plate; 36px inner notes

### Inputs / Fields
- **Style:** The threshold range is the only input — transparent track, brick thumb
- **Focus:** 3px teal ring, offset 3px
- **Error / Disabled:** Not present

### Spectrum plate
Shared-prefix teal bar at the origin, three labeled teal peaks, hatched brick remainder on the right, thinking-trace in ink below a paper band. Trace is a flat baseline when remainder is under the cursor, a 2.2px step-function when it is not. Peak labels sit under the lobes; values sit above.

### Instrument stamp
Mono header, space-between, 1px ink underline. Left: identity. Right: plate number; synthetic-run line in brick at 0.75rem.

### Glossary
Overpass Condensed teal-deep terms in an 11.5rem column; Source Sans 3 definitions. Stacks to one column at 720px.

### Refuse band
Full-width ink rules, 14px vertical padding, 1.4em / 1.6em outer margin. Plain notes type, not a tinted callout.

### Colophon
Ink-soft 0.9rem above a 1px ink rule; links teal-deep with 3px underline offset.

## Do's and Don'ts

### Do:
- **Do** keep typed peaks teal, remainder brick-hatched, and the thinking trace in ink (**The Three-Ink Rule**).
- **Do** letter the plate in Overpass Condensed and the notes in Source Sans 3 (**The Plate vs Notes Rule**).
- **Do** rule the sheet with 28px sprocket holes and dashed perforation, and keep notes on that same strip (**The Continuous Sheet Rule**).
- **Do** use a single brick threshold cursor as the interactive control; caption under the plot is the sentence that must survive a skim.
- **Do** mark synthetic traces as synthetic, in stamp mono at brick.

### Don't:
- **Don't** ship a dark AI explainer, three feature cards, or a rounded workbench panel on this surface.
- **Don't** print cream or warm ivory paper; this stock is grey-green.
- **Don't** mix NInfer Supervisor (Segoe, forest green, 12–14px radius) into this world.
- **Don't** draw hard offset neubrutalist shadows, glyph icons, or kicker/eyebrow labels above the title.
- **Don't** set paragraph copy in JetBrains Mono or treat the stamp as a marketing eyebrow.
