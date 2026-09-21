---
name: NInfer Decision Analyzer
description: A 1980s bench logic analyzer reading closed-set probabilities off live model logits.
colors:
  putty: "#c8c3b4"
  putty-deep: "#b3ad9d"
  putty-shadow: "#8f8979"
  silk: "#2b2822"
  silk-soft: "#3e3a2e"
  frame: "#3a352c"
  frame-edge: "#201c15"
  glass: "#170f06"
  glass-lit: "#24170a"
  phosphor: "#ffb000"
  phosphor-dim: "#ad7a1c"
  phosphor-faint: "#4f3a12"
  key: "#d7d2c3"
  key-lit: "#ffcf5c"
  alarm: "#ff6a3d"
typography:
  display:
    fontFamily: "Barlow Condensed, Barlow, sans-serif"
    fontSize: "44px"
    fontWeight: 700
    lineHeight: 0.95
    letterSpacing: "-0.01em"
  headline:
    fontFamily: "Barlow Condensed, Barlow, sans-serif"
    fontSize: "22px"
    fontWeight: 600
    letterSpacing: "0.08em"
  title:
    fontFamily: "Barlow Condensed, Barlow, sans-serif"
    fontSize: "14px"
    fontWeight: 700
    letterSpacing: "0.16em"
  body:
    fontFamily: "Barlow, system-ui, sans-serif"
    fontSize: "13.5px"
    fontWeight: 400
    lineHeight: 1.35
  label:
    fontFamily: "Barlow Condensed, Barlow, sans-serif"
    fontSize: "12.5px"
    fontWeight: 600
    letterSpacing: "0.08em"
  mono:
    fontFamily: "Kode Mono, ui-monospace, Menlo, monospace"
    fontSize: "13px"
    fontWeight: 400
    letterSpacing: "0.04em"
  readout:
    fontFamily: "Kode Mono, ui-monospace, Menlo, monospace"
    fontSize: "17px"
    fontWeight: 700
    letterSpacing: "0.02em"
rounded:
  panel: "10px"
  frame: "8px"
  crt: "14px / 18px"
  key: "3px"
  dial: "50%"
spacing:
  panel-pad: "32px 36px"
  column-gap: "28px"
  side-gap: "26px"
  block-pad-top: "12px"
  legend-gap: "7px"
  crt-pad: "16px 22px 12px"
  crt-row-gap: "10px"
  channel-gap: "6px"
  channel-cols: "46px 222px 1fr 110px"
  softkey-gap: "12px"
components:
  panel:
    backgroundColor: "{colors.putty}"
    textColor: "{colors.silk}"
    rounded: "{rounded.panel}"
    padding: "{spacing.panel-pad}"
  frame:
    backgroundColor: "{colors.frame}"
    rounded: "{rounded.frame}"
    padding: "14px 14px 10px"
  crt:
    backgroundColor: "{colors.glass}"
    textColor: "{colors.phosphor}"
    typography: "{typography.mono}"
    rounded: "{rounded.crt}"
    padding: "{spacing.crt-pad}"
  softkey-button:
    backgroundColor: "{colors.key}"
    rounded: "{rounded.key}"
    height: "26px"
  softkey-button-lit:
    backgroundColor: "{colors.key-lit}"
    rounded: "{rounded.key}"
    height: "26px"
  channel-fill:
    backgroundColor: "{colors.phosphor-dim}"
  channel-fill-decided:
    backgroundColor: "{colors.phosphor}"
  status-live:
    textColor: "{colors.phosphor}"
    typography: "{typography.mono}"
  fault-line:
    textColor: "{colors.alarm}"
    typography: "{typography.body}"
---

# Design System: NInfer Decision Analyzer

## Overview

**Creative North Star: "The Bench Logic Analyzer"**

This system is a putty-grey 1980s test instrument sitting on a bench, not a dark AI dashboard and
not a chat box beside answer cards. The front panel is painted metal with a silkscreened plate; the
right two-thirds is a recessed CRT behind dark glass, showing amber phosphor text and traces on a
faint raster; five bezel soft keys sit under the screen with their labels silkscreened above them.
The instrument's four channels are the four closed questions; the trace level in each is the model's
own probability for an option, acquired in one pass and read off with a cursor. The page never
invents an answer: on failure the CRT prints a FAULT line in place of a trace.

Three materials carry the whole system: painted metal (the panel, frame, and keys), amber phosphor
(every live value, trace, and cursor on the glass), and silkscreen ink (the plate copy and dial
labels on the metal). Nothing outside those three material logics appears — no neon glow-edge chrome,
no card-and-badge dashboard grammar, no kicker or eyebrow labels. Bloom exists only where the
phosphor lights itself: a soft halo on the leading edge of a filled trace and on the cursor line,
never as a decorative glow on the metal.

**Key Characteristics:**
- Putty-grey painted-metal panel with a darker inset CRT frame, small constant bezel radii
- Amber phosphor (`#ffb000` on `#170f06`) for every on-screen character and trace, with a faint
  scanline raster over the glass
- Barlow Condensed silkscreens the metal plate; Kode Mono is the phosphor's own type; Barlow is the
  panel's body copy
- Four channel rows with decoded bus segments whose fill height is the probability, a triangular
  cursor, and a P/Δt readout column
- Five soft keys under the screen, only RUN and REPLAY ever actuate; the POWER LED and CURSOR
  soft key light without being buttons the user presses

## Colors

Putty-grey metal under a single warm accent: amber phosphor is the only color that reads as "alive."

### Primary
- **Phosphor Amber** (`#ffb000`): The decided trace's top edge and fill, the cursor line and its
  triangle, the live status word, the P readout value and its option key, and the STATE window text
  while capturing. This is the instrument's one voice; nothing else on the glass competes with it.
- **Phosphor Dim** (`#ad7a1c`): Un-decided trace fill, channel IDs before decision, bus segment
  labels, status separators, and the P value before a channel decides. The same hue at lower
  intensity, never a second color.

### Secondary
- **Alarm Orange** (`#ff6a3d`): The FAULT line and the status word's color when a request fails.
  Reserved for the one failure state; never used decoratively.

### Neutral
- **Panel Putty** (`#c8c3b4` deepening to `#b3ad9d`): The painted-metal front panel, radial-lit from
  the top edge.
- **Putty Shadow** (`#8f8979`): Panel screw heads, the block dividers on the silkscreen column, and
  the soft key's resting drop shadow.
- **Silkscreen Ink** (`#2b2822`) / **Silkscreen Soft** (`#3e3a2e`): Plate copy, block headings, and
  dt/dd labels stenciled onto the putty panel.
- **Frame** (`#3a352c` to `#201c15`): The recessed bezel that holds the CRT, a tonal well one step
  darker than the panel.
- **Glass** (`#170f06`, lit `#24170a`): The CRT substrate itself, radially lit from center.
- **Phosphor Faint** (`#4f3a12`): Hairline rules under the status line, the STATE window border, and
  the bus baseline — the glass's own darkest visible ink, not a UI grey.
- **Key / Key-Lit** (`#d7d2c3` / `#ffcf5c`): Soft key resting and lit-armed face.

### Named Rules
**The Three-Material Rule.** Every surface is painted metal, amber phosphor, or silkscreen ink.
There is no fourth material system (no neon glass, no flat UI grey) anywhere on the page.

**The Phosphor's Own Bloom Rule.** The only glow in the system is `0 0 5px`/`0 0 8px` amber halo on
a trace's lit top edge and on the cursor. Bloom never appears on the metal panel or the keys.

## Typography

**Display Font:** Barlow Condensed (with Barlow, sans-serif) — the metal plate
**Body Font:** Barlow (with system-ui, sans-serif) — panel copy off the glass
**Label/Mono Font:** Kode Mono (with ui-monospace, Menlo, monospace) — every character on the CRT

**Character:** Barlow Condensed is the silkscreen stencil voice for the plate and block headers;
Barlow carries the panel's short prose (the legend, the repo line); Kode Mono is the phosphor's own
face — status, state text, channel data, and readouts never leave it.

### Hierarchy
- **Display** (700, 44px, line-height 0.95, tracking −0.01em, Barlow Condensed): The `NInfer` brand
  line on the plate, the largest mark on the page.
- **Headline** (600, 22px, tracking 0.08em, Barlow Condensed): `DECISION ANALYZER`, the model-plate
  subtitle.
- **Title** (700, 14px, tracking 0.16em, Barlow Condensed, uppercase by content): Block headers
  `INPUT` / `READOUT` on the silkscreen column.
- **Body** (400, 13.5px, line-height 1.35, Barlow): Legend rows and short panel prose.
- **Label** (600, 12.5px, tracking 0.08em, Barlow Condensed): `dt` field labels (`SOURCE`, `MODEL`,
  `MODE`) beside their mono values.
- **Mono** (400, 13–15px, tracking 0.02–0.04em, Kode Mono): Every CRT surface — status line, STATE
  text, channel questions and IDs, bus labels, soft-key labels, and the series/repo stamp on the
  plate.
- **Readout** (700, 17px, tracking 0.02em, Kode Mono): The bold P value in each channel's readout
  column, the single largest live number on the glass.

### Named Rules
**The Stencil vs Glass Rule.** Barlow Condensed and Barlow are silkscreened onto the putty metal;
Kode Mono exists only behind the glass. Neither face crosses onto the other's material.

## Layout

A single fixed 1280×720 instrument face, centered on a darker bench background (`#5b574c`), scaled
down uniformly (`transform: scale()`) rather than reflowed below that viewport; there is no
responsive breakpoint restructuring inside the panel itself. The panel is a two-column CSS grid: a
fixed 300px silkscreen column (brand plate, INPUT block, READOUT legend, repo line pinned to the
bottom with `margin-top: auto`) and a flexible CRT column, with a 28px column gap. Inside the CRT,
a row-gap-10px stack holds the status line, the STATE window, and the four-row channel grid
(row-gap 6px); each channel row is a fixed four-column track (46px id / 222px question / flexible
bus / 110px readout). The soft-key strip below the screen is five equal columns at 12px gaps.

## Elevation & Depth

Depth is a physical enclosure, not floating UI cards: a radial-lit metal panel, a darker recessed
frame, and a glass well lit from its own center. The only shadows are a panel drop shadow onto the
bench, an inset frame shadow that reads as the bezel's depth, an inset CRT shadow that reads as
glass curvature, and a soft key's resting shadow that flattens to an inset press shadow when armed
or clicked. No card in the system elevates on hover; the only "lift" state is a soft key's own
physical press.

### Shadow Vocabulary
- **Panel on bench** (`0 30px 60px -20px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.35)`):
  Ambient lift of the whole instrument off the desk background.
- **Frame well** (`inset 0 2px 6px rgba(0,0,0,0.55), 0 1px 0 rgba(255,255,255,0.25)`): The bezel
  reads as a recess cut into the panel.
- **CRT curvature** (`inset 0 0 60px rgba(0,0,0,0.65)`): Vignette that reads the glass as curved and
  lit from its own center, not a flat rectangle.
- **Soft key rest / press** (`0 2px 0 var(--putty-shadow), inset 0 1px 0 rgba(255,255,255,0.7)` at
  rest; `0 0 0 var(--putty-shadow), inset 0 1px 2px rgba(0,0,0,0.25)` pressed): A 2px physical
  key travel, not a hover lift.

### Named Rules
**The Enclosure, Not the Card Rule.** Depth comes from one physical enclosure (panel → frame →
glass) and key travel. No component elevates as an independent floating card.

## Shapes

Small, constant bezel radii throughout: 10px on the outer panel, 8px on the inner CRT frame, an
elliptical 14px/18px on the CRT glass itself (wider than tall, reading as tube curvature), and 3px
on the soft keys — the smallest radius in the system, appropriate to a physical button rather than a
UI chip. Screws and the POWER LED are perfect circles (50%). The cursor is a small solid triangle
(a 6px CSS border wedge) rather than a glyph icon. There is no large-radius, card-style rounding
anywhere; radius scales down as elements get physically smaller (panel > frame > CRT bezel > keys).

### Named Rules
**The Bezel Radius Rule.** Radius is always the enclosure's own: 10/8/14–18/3px, largest at the
panel and smallest at the keys. Never a uniform dashboard 12–16px card radius.

## Components

Instrument parts on one panel, not a widget kit. There is no nav, no chip, no generic card.

### CRT screen
- **Shape:** Elliptical 14px/18px radius glass in an 8px-radius metal frame, radially lit
  (`glass-lit` center to `glass` edge)
- **Overlay:** A repeating 3px scanline raster (`repeating-linear-gradient`, `mix-blend-mode:
  multiply`) sits over the whole screen at all times
- **Power state:** Opacity fades 0 → 1 over 0.7s on power-on; a `FAULT` line in alarm orange
  overlays the lower screen on request failure, and the status word turns alarm orange with it

### Status line
Space-between mono row above a phosphor-faint hairline: left is the bold blinking `ACQ` glow plus
the current mode word (`STANDBY` / `CAPTURE` / `ARM` / `ACQUIRING` / `DECIDED` / `FAULT`); right is
channel count, token count (with cached count when present), and Δt in milliseconds. `ACQUIRING`
blinks via `steps(2, end)` opacity.

### State window
A bordered phosphor-faint box with a raised tab label (`STATE · SAMPLE INPUT`) cut into its top
edge; text types in at 3-character increments every 9ms behind a blinking block caret, then the
caret is removed. Reduced motion sets the full text at once with no caret.

### Bus segments (channel trace)
Each of the four channel rows is: a dim channel ID, the question with a small dim sublabel (`YES /
NO` or `N OPTIONS`), a bus of equal-width segments each holding a `fill` div whose height animates
0→probability% (0.9s ease) with a 2px phosphor-halo edge on top, and a 110px readout column. The
decided option's segment turns full phosphor (`.top`); its label inverts to glass-dark text when the
fill is tall, or stays phosphor when the fill is short (`.lowfill`). Rows sit at 0.18 opacity until
armed, then fade to 1 (0.35s).

### Cursor
A 2px phosphor line with a halo, sliding to the decided segment's center over 1.1s ease and fading
in with it; a 6–7px CSS triangle caps its top edge, pointing down into the trace.

### P readout
A 110px-wide column per channel: `P` label, the bold 17px phosphor value (probability to 3 decimals),
and the decided option's key below it. Values are invisible (`opacity: 0`) until the channel decides,
then fade in over 0.3s — never present before the real response arrives.

### Soft keys
Five equal 1fr columns under the screen, each a mono label above a 3px-radius putty button with a
2px physical press. Only **RUN** and **REPLAY** are wired to actuate the sequence; **SINGLE** and
**STATE** are permanently `disabled` (silkscreened but inert); **CURSOR** starts disabled and gains
the `lit` label color once a decision lands, but is never clickable. RUN itself becomes `lit` and
visually `pressed` for the duration of the live request.

### POWER LED
A 10px circle beside the `POWER` label on the plate; unlit is a dark rust (`#5a3a2a`), lit is a
warm orange (`#ff8a2a`) with a 0.4s background transition — the plate's own indicator, not a phosphor
element.

## Motion

The sequence is scripted, not user-paced: power-on fade (0.7s) → LED on → STATE text types in
character-chunks at 9ms/step → a 300ms pause → each channel arms in turn at 220ms intervals → RUN
lights and depresses → the live `/v1/score` request runs (`ACQUIRING`, blinking) → on success, bus
fills animate to their probabilities (0.9s), the cursor slides to the decided segment (1.1s), then
the P readout fades in (0.3s) → CURSOR key lights. `R`/`Space` and the REPLAY key re-run the whole
sequence; `?loop=1` re-triggers it automatically 5s after completion. `prefers-reduced-motion:
reduce` removes every transition and the blink/caret animations, collapsing the sequence to instant
state changes while preserving the same order of events.

### Named Rules
**The One Real Sweep Rule.** The only thing the script waits on is the actual `/v1/score` round
trip; every other timing (typing, arming, fills, cursor) is a fixed choreography around that one
real measurement, and Δt on screen is the real request time, never an authored number.

## Do's and Don'ts

### Do:
- **Do** keep phosphor amber (`#ffb000`) as the only "alive" color on the glass, with phosphor-dim
  for the un-decided state and alarm orange reserved for the one failure state (**The Three-Material
  Rule**).
- **Do** letter the metal plate in Barlow Condensed/Barlow and keep Kode Mono exclusive to the CRT
  (**The Stencil vs Glass Rule**).
- **Do** scale radius down with the enclosure: 10px panel, 8px frame, 14–18px CRT, 3px keys, never a
  flat 12–16px dashboard radius (**The Bezel Radius Rule**).
- **Do** let bloom appear only on a lit trace edge and the cursor; keep the panel and keys matte.
- **Do** tie every on-screen number to the real `/v1/score` response and its measured latency; a
  request failure prints `FAULT` in alarm orange, never a synthetic answer.

### Don't:
- **Don't** add a neon-glow "AI dashboard" treatment, chat-box-beside-answer-cards layout, or bar
  charts with drop shadows to this world.
- **Don't** draw hard offset neobrutalist shadows, glyph icons, or kicker/eyebrow labels above any
  heading; the cursor's triangle is a CSS border wedge, not an icon-font glyph.
- **Don't** mix the NInfer Supervisor workbench (`DESIGN.md`, Segoe, forest green, 12–14px radius)
  or the Dual-Purpose Decode plotter-stock world (`docs/dual-purpose.DESIGN.md`, teal/brick thermal
  paper, square 0-radius stock) into this instrument's palette or shapes.
- **Don't** wire SINGLE or STATE as functional controls, or treat CURSOR's lit state as a clickable
  affordance; only RUN and REPLAY actuate, and that is this build's own behavior, not a rule that
  future soft keys must always ship half-inert.
- **Don't** animate a bus fill, cursor position, or P value ahead of the response; every fade-in is
  gated on `decided`, driven by the real request.
