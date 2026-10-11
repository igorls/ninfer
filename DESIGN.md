---
name: NInfer Supervisor
description: An avionics glass cockpit for one local inference engine.
colors:
  panel-gray: "#343a3f"
  glareshield-gray: "#1d2124"
  panel-seam: "#121517"
  bezel-ring: "#262b2f"
  bezel-ring-top: "#2e3439"
  bezel-ring-low: "#1f2326"
  key-face: "#1e2225"
  key-top-edge: "#4a5157"
  key-face-hover: "#262b2f"
  key-legend: "#e9edef"
  key-legend-dim: "#aab4ba"
  display-glass: "#040708"
  display-glass-inset: "#0a0f12"
  glass-rule: "rgba(233,240,244,.15)"
  glass-rule-faint: "rgba(233,240,244,.075)"
  measured-white: "#f1f4f5"
  secondary-gray: "#a8b2b8"
  legend-gray: "#8592a0"
  inactive-gray: "#76828a"
  engaged-green: "#4fd98d"
  selectable-cyan: "#41cdea"
  target-magenta: "#ee78de"
  caution-amber: "#f5b53d"
  warning-red: "#ff6257"
  warning-red-text: "#ffb2ab"
  reading-ice: "#98bfe0"
  caution-glass: "#1c1406"
  warning-glass: "#200807"
  log-text: "#c9d3d8"
  bezel-glint: "#4a5157"
  pole-dark: "#2a0a08"
  day-panel-gray: "#cdd2d6"
  day-glareshield-gray: "#b7bec3"
  day-panel-seam: "#98a1a8"
  day-bezel-ring: "#aeb6bc"
  day-bezel-ring-top: "#bcc3c8"
  day-bezel-ring-low: "#a1aab0"
  day-bezel-glint: "#e4e8eb"
  day-display-glass: "#fbfbfa"
  day-display-glass-inset: "#eef1f2"
  day-glass-rule: "rgba(18,24,28,.17)"
  day-glass-rule-faint: "rgba(18,24,28,.085)"
  day-measured-ink: "#11171a"
  day-secondary-gray: "#44515a"
  day-legend-gray: "#56636c"
  day-inactive-gray: "#86919a"
  day-engaged-green: "#0a7a43"
  day-selectable-cyan: "#006f99"
  day-target-magenta: "#9e2189"
  day-caution-amber: "#8f5400"
  day-warning-red: "#b3241c"
  day-warning-red-text: "#a01d16"
  day-reading-ice: "#2b5fa3"
  day-caution-glass: "#fff1d4"
  day-warning-glass: "#fde6e3"
  day-log-text: "#1f282d"
  day-pole-light: "#ffffff"
typography:
  display:
    fontFamily: "Bahnschrift, 'DIN Alternate', 'Barlow Semi Condensed', 'Arial Narrow', sans-serif"
    fontSize: "24px"
    fontWeight: 600
    lineHeight: 1.1
    letterSpacing: "-0.01em"
    fontVariation: "'wdth' 87.5"
  headline:
    fontFamily: "Bahnschrift, 'DIN Alternate', 'Barlow Semi Condensed', 'Arial Narrow', sans-serif"
    fontSize: "18px"
    fontWeight: 400
    lineHeight: 1.2
    letterSpacing: "0.04em"
    fontVariation: "'wdth' 87.5"
  title:
    fontFamily: "Bahnschrift, 'DIN Alternate', 'Barlow Semi Condensed', 'Arial Narrow', sans-serif"
    fontSize: "17px"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "0.06em"
    fontVariation: "'wdth' 87.5"
  value:
    fontFamily: "Bahnschrift, 'DIN Alternate', 'Barlow Semi Condensed', 'Arial Narrow', sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.2
    letterSpacing: "0.03em"
    fontVariation: "'wdth' 87.5"
  readout:
    fontFamily: "Bahnschrift, 'DIN Alternate', 'Barlow Semi Condensed', 'Arial Narrow', sans-serif"
    fontSize: "17px"
    fontWeight: 600
    lineHeight: 1.2
    fontFeature: "'tnum' 1"
    fontVariation: "'wdth' 87.5"
  key:
    fontFamily: "Bahnschrift, 'DIN Alternate', 'Barlow Semi Condensed', 'Arial Narrow', sans-serif"
    fontSize: "12.5px"
    fontWeight: 600
    letterSpacing: "0.06em"
    fontVariation: "'wdth' 87.5"
  label:
    fontFamily: "Bahnschrift, 'DIN Alternate', 'Barlow Semi Condensed', 'Arial Narrow', sans-serif"
    fontSize: "11px"
    fontWeight: 400
    letterSpacing: "0.06em"
    fontVariation: "'wdth' 87.5"
  body:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.5
  body-small:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "12.5px"
    fontWeight: 400
    lineHeight: 1.55
  mono:
    fontFamily: "'Cascadia Mono', 'Cascadia Code', Consolas, monospace"
    fontSize: "11.5px"
    fontWeight: 400
    lineHeight: 1.7
rounded:
  hairline: "1px"
  lamp: "2px"
  tag: "3px"
  key: "4px"
  pad: "5px"
  glass: "7px"
  bezel: "10px"
spacing:
  key-gap: "6px"
  gap: "8px"
  display-pad-x: "11px"
  glareshield-gap: "14px"
  cdu-pad-x: "18px"
  column-gap: "28px"
components:
  function-key:
    backgroundColor: "{colors.key-face}"
    textColor: "{colors.key-legend}"
    typography: "{typography.key}"
    rounded: "{rounded.key}"
    padding: "0 13px"
    height: "32px"
  function-key-hover:
    backgroundColor: "{colors.key-face-hover}"
    textColor: "{colors.measured-white}"
  key-disabled:
    backgroundColor: "{colors.key-face}"
    textColor: "{colors.inactive-gray}"
  key-go:
    backgroundColor: "{colors.key-face}"
    textColor: "{colors.engaged-green}"
  key-stop:
    backgroundColor: "{colors.key-face}"
    textColor: "{colors.warning-red-text}"
  key-insert:
    backgroundColor: "{colors.key-face}"
    textColor: "{colors.selectable-cyan}"
  display-mode-key:
    backgroundColor: "{colors.key-face}"
    textColor: "{colors.key-legend}"
    typography: "{typography.key}"
    rounded: "{rounded.key}"
    padding: "0 10px"
    height: "28px"
  prompt:
    textColor: "{colors.selectable-cyan}"
    typography: "{typography.key}"
  master-caution-dark:
    backgroundColor: "#241b0b"
    textColor: "#6e5726"
    rounded: "{rounded.key}"
    padding: "2px 8px"
    height: "32px"
  master-caution-lit:
    backgroundColor: "{colors.caution-amber}"
    textColor: "#1b1305"
  master-warning-lit:
    backgroundColor: "{colors.warning-red}"
    textColor: "{colors.warning-glass}"
  display-unit:
    backgroundColor: "{colors.display-glass}"
    textColor: "{colors.measured-white}"
    rounded: "{rounded.bezel}"
    padding: "7px 11px 9px"
  display-unit-day:
    backgroundColor: "{colors.day-display-glass}"
    textColor: "{colors.day-measured-ink}"
    rounded: "{rounded.bezel}"
    padding: "7px 11px 9px"
  cdu-page:
    backgroundColor: "{colors.display-glass}"
    textColor: "{colors.measured-white}"
    rounded: "{rounded.bezel}"
    padding: "12px 18px 18px"
  cdu-temporary-tag:
    textColor: "{colors.caution-amber}"
    typography: "{typography.label}"
    rounded: "{rounded.tag}"
    padding: "1px 7px"
  input-field:
    backgroundColor: "{colors.display-glass-inset}"
    textColor: "{colors.selectable-cyan}"
    typography: "{typography.value}"
    rounded: "{rounded.key}"
    padding: "6px 10px"
    height: "34px"
  input-field-day:
    backgroundColor: "{colors.day-display-glass-inset}"
    textColor: "{colors.day-selectable-cyan}"
  input-field-staged:
    backgroundColor: "{colors.caution-glass}"
    textColor: "{colors.caution-amber}"
  input-field-readonly:
    textColor: "{colors.secondary-gray}"
  scratchpad-caution:
    backgroundColor: "{colors.caution-glass}"
    textColor: "{colors.caution-amber}"
    rounded: "{rounded.pad}"
    padding: "6px 12px"
  scratchpad-warning:
    backgroundColor: "{colors.warning-glass}"
    textColor: "{colors.warning-red-text}"
    rounded: "{rounded.pad}"
    padding: "6px 12px"
  log-pane:
    backgroundColor: "{colors.display-glass-inset}"
    textColor: "{colors.log-text}"
    typography: "{typography.mono}"
    rounded: "{rounded.key}"
    padding: "8px 10px"
  bench-method:
    textColor: "{colors.measured-white}"
    typography: "{typography.key}"
    rounded: "{rounded.key}"
    padding: "6px 8px"
    width: "62px"
  bench-endpoint:
    textColor: "{colors.selectable-cyan}"
    typography: "{typography.mono}"
    rounded: "{rounded.tag}"
    padding: "4px 6px"
  bench-response:
    backgroundColor: "{colors.display-glass-inset}"
    textColor: "{colors.log-text}"
    typography: "{typography.mono}"
    rounded: "{rounded.key}"
    padding: "8px 10px"
---

# Design System: NInfer Supervisor

## Overview

**Creative North Star: "The Glass Cockpit"**

The Supervisor is a flight deck of electronic displays, not a web dashboard and not glassmorphism. A panel-gray instrument panel carries black display glass set in raised bezel rings; every label, key legend and annunciation is lettered in condensed DIN capitals; colour is a code with fixed meanings, never decoration. The layout copies the cockpit split: a primary display band that is always in view for the glance, and one paged control-display unit (CDU) for everything visited rarely: connecting, testing the API, diagnosing, changing settings.

The panel has two lighting modes. Night is black glass in a dark-gray bezel; Day is white glass in a cool light-gray bezel, the way cockpit displays read in daylight. Auto, the default, follows the operating system's colour scheme. Only the displays and the panel change between modes; the keys are hardware and stay dark-capped with their night legends.

It is dense on purpose. A narrow side window has to show readiness, memory, throughput and the calling apps with no scrolling. Plain-language explanation still exists but is called up on demand (the Explain key) instead of taking space by default. The pull is toward the real-avionics reading: tapes, readout windows, bugs, barber poles, function keys and line-select keys. The pull is never toward frosted blur, gradients on panels, or a card grid.

**Key Characteristics:**
- Night: panel-gray bezel, black display glass, a faint cover-glass sheen on the glass only. Day: white glass in a light-gray bezel, no sheen. Auto follows the OS.
- Bahnschrift at 87.5% width in caps for all panel lettering; Segoe UI Variable Text for sentences; Cascadia Mono for machine text.
- Semantic avionics colour: white measured, green engaged, cyan selectable, magenta target, amber caution or staged, red warning, ice blue prompt reading.
- A glareshield row and a mode-annunciator row on every page; Monitor is four displays; every other page (Connect, API, Diagnose, Settings) is a CDU page.
- Changes are announced, not animated: a white box for ten seconds, an amber TEMPORARY revision, a Master Caution light.

## Colors

Two neutral materials (panel and glass) and seven signal colours, each with exactly one meaning, in two lighting modes that share one code. Every colour is a custom property on the root; Day overrides the same properties, and the canvases read their palette from them, so a mode change repaints the instruments too.

### Primary
- **Engaged Green** (engaged-green): the engine is ready, a mode is as configured, the generation channel (its trace, tape scale and readout), the current page lamp on a function key, the supervisor's own process in the GPU app list, a saved status.
- **Selectable Cyan** (selectable-cyan): anything the user can act on or edit. `<PROMPT` and `PROMPT>` lines, links, editable input values, toggled keys, INSERT, the armed line under the engine mode, and the focus ring.

### Secondary
- **Target Magenta** (target-magenta): configured targets, never measurements. The desktop-reserve bug on the memory tape, the reserve readout, and the capacity sliders' accent.
- **Caution Amber** (caution-amber): caution findings and anything staged but not saved. The TEMPORARY tag, the amber CDU title and ring, staged fields, the UNSAVED tag on the Settings key, the alert count, a lit Master Caution, a supervisor-reduced plan. Caution Glass (caution-glass) is its background tint for caution rows and lines.
- **Warning Red** (warning-red): warnings and the desktop-reserve limit, drawn as a red-and-dark barber pole on the memory tape and in the reserve preview. Warning Red Text (warning-red-text) is the readable tint for red copy on Warning Glass (warning-glass) and for the Stop key legend.

### Tertiary
- **Reading Ice** (reading-ice): the prompt-reading channel only: its trace, its right-hand scale and its readout. It exists so two rates can share a display on separate scales without being confused.

### Neutral
- **Measured White** (measured-white): measured values and primary text. Memory used, card total, readout figures, the memory trace.
- **Secondary Gray** (secondary-gray), **Legend Gray** (legend-gray), **Inactive Gray** (inactive-gray): sources and units; field legends and axis labels; disabled keys and unavailable presets, in that order.
- **Panel Gray** (panel-gray) and **Glareshield Gray** (glareshield-gray): the instrument panel the displays sit in, and the darker strip that carries the function keys. Panel Seam (panel-seam) is the 1px cut between panel parts.
- **Bezel Ring** (bezel-ring, bezel-ring-top, bezel-ring-low): the 4px ring around each display, lit from above. Bezel Glint (bezel-glint) is the 1px lower highlight that cuts the ring from the panel.
- **Display Glass** (display-glass) and **Glass Inset** (display-glass-inset): the black screen, and the slightly lifted wells for inputs, code and logs. Glass Rule and Glass Rule Faint separate columns and lines on the glass.
- **Key Face**, **Key Top Edge**, **Key Legend** (key-face, key-top-edge, key-legend): physical keys on the panel.

### Day mode
The Day set (`day-` tokens) replaces the panel and glass materials and darkens every signal colour so it holds contrast on white glass; the meanings do not move.
- **Day Display Glass** (day-display-glass) and **Day Glass Inset** (day-display-glass-inset): the near-white screen and its cool wells. Day Glass Rule and Day Glass Rule Faint are dark hairlines at the same two weights as at night.
- **Day Panel Gray**, **Day Glareshield Gray**, **Day Bezel Ring** (day-panel-gray, day-glareshield-gray, day-bezel-ring with its top and low tones): a cool panel-gray bezel around the white glass, with a light glint (day-bezel-glint) under the ring and a mid-gray seam (day-panel-seam).
- **Day Measured Ink** (day-measured-ink) takes Measured White's role; Day Secondary, Legend and Inactive Gray keep the same three-step order.
- **Day signal colours** (day-engaged-green, day-selectable-cyan, day-target-magenta, day-caution-amber, day-warning-red, day-reading-ice): the night hues, darkened for white glass. Day Warning Red Text, Day Caution Glass and Day Warning Glass are the readable red copy and the pale caution and warning tints. Day Log Text (day-log-text) is the ink of wells and logs.
- **Barber pole:** red and dark (pole-dark) at night, red and white (day-pole-light) by day.
- **Physical keys** (function keys, panel keys, the Master lights) keep the night key face, legends and signal legends in Day.

### Named Rules
**The One Meaning Rule.** Each signal colour carries one meaning across every page. A colour is never chosen because it looks good next to another; if an element's meaning has no colour in the code, it is white or gray.

**The Configured Is Not Running Rule.** Configured targets are magenta; measured state is white or green. A setting and the engine's running value are never drawn in the same colour.

**The Amber Revision Rule.** Anything staged but not inserted is amber, from the field to the page title to the function key, until INSERT or ERASE.

**The One Code In Daylight Rule.** Day and Night are the same colour code under different light. A mode changes a signal colour's lightness, never its meaning, and no colour exists in only one mode.

**The Keys Are Hardware Rule.** Keys do not change with the display. Function keys, panel keys and the Master lights keep their dark caps and night legends in every mode; only glass and panel follow the display mode.

## Typography

**Display Font:** Bahnschrift (Windows' built-in DIN), condensed to 87.5% width, with DIN Alternate, Barlow Semi Condensed and Arial Narrow as fallbacks
**Body Font:** Segoe UI Variable Text (with Segoe UI, system-ui)
**Label/Mono Font:** Cascadia Mono (with Cascadia Code, Consolas)

**Character:** DIN is the lettering of panels and placards, so it does all the instrument talking. Segoe is the voice for sentences, so it sounds like a person explaining. The build sets the width with `font-stretch: 87.5%`; the `wdth` variation in the tokens is the same thing.

### Hierarchy
- **Display** (600, 24px, 1.1): the major readout, memory used.
- **Headline** (400, 18px, 1.2, caps, 0.04em): the engine mode (READY, STOPPED).
- **Title** (600, 17px, 1.2, caps, 0.06em): the centred CDU page title.
- **Value** (400, 16px, 1.2, caps, 0.03em): annunciator values (model, plan) and the bench's HTTP status (600); 14px for facts, inputs and table client names.
- **Readout** (600, 17px, tabular figures): display readouts beside the traces; 15px in the calling-apps table and for memory's secondary lines.
- **Key** (600, 12.5px, caps, 0.06em): function keys, keys, prompts (prompts are 11.5-14px).
- **Label** (400-600, 11px, caps, 0.06em, legend gray): field legends above values, table headers, bench group legends, display titles at 12px.
- **Body** (400, 13px/1.5 Segoe): prose; help text at 12-12.5px, kept to 88-95ch.
- **Mono** (11.5px/1.7 Cascadia): logs, the launch command, file names, flags and addresses; 12-13px for API paths, request bodies and responses.

### Named Rules
**The Lettering Rule.** Panel lettering is DIN in caps; sentences are Segoe in sentence case. A sentence is never set in caps, and a label is never set in Segoe.

**The Machine Text Rule.** Monospace is only for text a machine reads or writes: logs, commands, paths, flags, addresses, request and response bodies. Numbers use DIN with tabular figures, not mono.

## Layout

The page is a fixed-height column (`100dvh`, body never scrolls): the glareshield strip, the mode-annunciator row, an optional scratchpad line, then the workspace. The glareshield and annunciator rows appear on every page.

- **Monitor** is a grid of four display units on an 8px gap. At 1180px and wider: memory over rates on the left (memory column 1.55fr), and one flex column on the right (at least 380px) where calling apps size to their rows and alerts take the remaining height with their full text wrapped. Narrower, the four stack.
- **Memory and rates bodies** are size containers in the fixed-height layout (821px and wider, Explain off), so the readout column compacts as a display gets short (steps at 190px and 150px body height) instead of clipping. Both traces keep a 40px right gutter that holds the prompt-reading scale, clear of the plot.
- **CDU pages** centre one unit (max 980px; 1240px wide variant) with 34px side margins, and keep the one-line live band (memory, generating, reading, running, `<MONITOR`) in view above them.
- **API test bench** fills the CDU height with two panes: the endpoint index (220-280px) behind a faint rule, and the request over the response. At 880px and narrower the panes stack, the index list caps at 30vh and the page scrolls.
- **No page scroll** on Monitor or any CDU page at 1440x900, 960x1000 and 1280x720. Tables, logs and finding lists scroll inside their glass. Explain mode deliberately grows the page and scrolls.
- **Short screens** (max-height 780px, wider than 820px) tighten readouts and drop readout sub-lines.
- **820px and narrower:** function keys wrap to a full-width row, the annunciator row becomes two columns with the engine block full width, readouts move above the trace, two-column CDU grids collapse, and line-select keys leave the bezel.
- **560px and narrower (phones):** displays size to their content and the page scrolls; the calling-apps table turns into stacked rows with inline labels.
- Rhythm: 8px between units, 6px between keys, 10-12px inside display columns, 28px between CDU columns, 7-10px line padding with a faint rule under each line.

## Elevation & Depth

Depth is physical, not floating. The panel is flat; each display is recessed glass inside a raised bezel ring. The ring is lit from above (lighter top edge, darker bottom), cut from the panel by a 1px seam and given a 1px lower highlight; the glass inside carries a deep inset shadow. Keys get a lighter top edge and press down 1px. A cover-glass sheen, a faint diagonal highlight, sits only on display glass, and only at night; Day glass has no sheen and a much lighter inset. The bezel shadow is one token per mode, so every display unit changes together. Nothing hovers over the panel except the confirmation dialog.

### Shadow Vocabulary
- **Bezel ring** (`box-shadow: 0 0 0 1px #121517, 0 1px 0 1px #4a5157, inset 0 0 0 1px #000, inset 0 2px 9px rgba(0,0,0,.9)` with a 4px bezel border): every display unit, the annunciator row, the live band, the CDU.
- **Day bezel ring** (`box-shadow: 0 0 0 1px #98a1a8, 0 1px 0 1px #e4e8eb, inset 0 0 0 1px rgba(40,48,54,.3), inset 0 2px 6px rgba(40,48,54,.16)`): the same ring by day.
- **Cover-glass sheen** (`linear-gradient(162deg, rgba(255,255,255,.045) 0, rgba(255,255,255,.012) 28%, rgba(255,255,255,0) 46%)` overlay): display glass only, Night only.
- **Temporary ring** (`box-shadow: 0 0 0 1px #f5b53d, 0 1px 0 1px #3a4045`): replaces the bezel ring on a CDU holding a staged revision.
- **Dialog** (`box-shadow: 0 0 0 1px rgba(233,240,244,.15), 0 18px 50px #000000aa`; Day `0 0 0 1px rgba(18,24,28,.17), 0 18px 50px #2a333a55`): the confirmation dialog, the only floating surface.

### Named Rules
**The Cover Glass Rule.** The sheen and inset shadow belong to display glass, and the sheen only at night. Panels, keys and the glareshield never get gradients, blur or glow.

## Shapes

Hard-edged hardware with softened corners. Display units have 10px outer corners on a 4px bezel; keys and inputs 4px; tags and the cursor readout 3px; lamps 2px; bars, slot strips and tape bodies 1px. Instrument marks are drawn, not iconified: the readout window is a box with a pointer notch and a taller drum slot for the last digit, the reserve bug is a magenta pentagon on the tape edge, and the reserve limit is a 45-degree barber pole. The few icons (copy, open, explain, display mode) are 14px inline SVG strokes at 1.4px.

## Components

### Buttons (keys)
Physical panel keys, never flat web buttons.
- **Shape:** 4px corners, 1px near-black border with a lighter top edge, 32px tall (30px engine keys, 28px small, 24px tiny).
- **Function keys:** Monitor, Connect, API, Diagnose, Settings on the glareshield, in that order (API amended 2026-10-11). A 2px lamp bar at the top lights green on the current page. A count chip (amber) or UNSAVED tag (amber outline) can sit inside.
- **Hover / Focus:** face darkens one step and the legend goes white (0.15s ease-out); pressed keys drop 1px; focus is the 2px cyan ring offset 2px.
- **Variants:** Start in green, Stop in red text, INSERT and SEND in cyan, toggles pressed show a cyan legend and cyan top edge, disabled keys go inactive gray.
- **Display-mode key:** a small glareshield key (28px) beside Master Caution that cycles Auto, Day, Night and annunciates the resolved mode (`AUTO · NIGHT`, `DAY`). The choice is stored per browser and resolved before first paint, so the page never flashes the other mode (amended 2026-10-11).

### Prompts
CDU prompt lines in cyan DIN caps, `<INDEX` pointing back and `PRESETS>` pointing on. The angle is part of the legend and hidden from screen readers. Hover underlines.

### Mode-annunciator row
Four columns on glass, present on every page: engine (mode, uptime, armed line, with Start/Restart/Stop beside the mode), model (with `PRESETS>`), running plan (As configured in green, Reduced by supervisor in amber, plus a one-line brief), API address (cyan, click to copy). Engine keys sit beside the engine mode at the row's left end: controls go next to the state they change, and the right end holds the API address (amended 2026-10-11 at finish review). The cyan armed line under the engine mode reads `Loading preset`, `Restart to apply` or `Settings staged`.

### Display units
- **Corner Style:** 10px over a 4px bezel ring.
- **Background:** display glass with the cover-glass sheen (Night), plain white glass (Day).
- **Head:** caps title in white, source in gray, range at the right in legend gray.
- **Memory and rates:** a vertical scale tape (66px) at left, a trace in the middle, readouts at right (156px). The rates display carries generation (green, left scale) and prompt reading (ice, scale in the trace's right gutter) on separate scales. Memory's secondary readouts are single label-value lines under the major readout. An empty rates trace says `Collecting rate samples…` while the engine is busy and `No rate samples in this window` otherwise.
- **Calling apps:** above phone widths an app name stays on one line with an ellipsis, while its counts and sub-lines wrap.

### Inputs / Fields
- **Style:** inset glass well, 1px `#2a343a` stroke, 4px corners, 34px tall, cyan DIN value; path fields use 13px mono.
- **Focus:** cyan ring. **Staged:** amber stroke, amber value, caution-glass well. **Read-only:** gray value on transparent, dashed stroke. **Invalid:** red stroke. **Disabled:** inactive gray.
- **Sliders:** native range with magenta accent (configured targets), a DIN tick scale under them.

### Navigation
Function keys are the top level. Settings opens a CDU index of line-select prompts in two columns, left prompts `<NAME` and right prompts `NAME>`, each with a one-line summary; at 821px and wider, small blank line-select keys sit in the bezel beside each line and light a cyan top edge on hover or focus. Diagnose uses outlined CDU tabs (cyan; current tab white).

### API test bench
A CDU page for calling every exposed route by hand (amended 2026-10-11).
- **Endpoint index:** a filter field, then caps group legends (Status, OpenAI, Anthropic, NInfer, Admin, Supervisor) over rows of method (DIN caps, gray) and path (cyan mono, wrapping after each `/`) with a one-line title. The current endpoint gets a hover-glass fill and a 1px cyan inset ring; disruptive routes add an amber `· disruptive`. A Recent list under a faint rule replays past requests.
- **Composer:** an outlined method box (amber for a disruptive route), the mono path field, SEND (cyan) and STOP (red) keys; example presets as outlined CDU tabs; a Stream check, Format and Copy as curl keys; the JSON body in a mono glass well whose stroke turns red when invalid, with a one-line validation status under it.
- **Response:** the HTTP status annunciated in DIN caps (green 2xx, amber 4xx, red 5xx or no answer), a timings line, and Text / Body / Events / Headers tabs over a mono glass well. JSON keys are legend gray and strings white. Events read as narrow time and name columns with the payload wrapped at its separators.
- **Confirmation:** a disruptive route confirms in the dialog, with the path in case-preserving mono.

### Signature behaviours
- **Annunciation box:** an annunciator value that changes is outlined in white for 10 seconds, so a glance catches the change rather than the state.
- **Temporary revision:** a staged edit turns the CDU amber: TEMPORARY tag at the title's left, amber title and ring, amber status line in the sticky INSERT/ERASE bar, UNSAVED on the Settings key, `Settings staged` under the engine mode. INSERT saves, ERASE discards.
- **Shared time cursor:** one cursor reads both traces at once, with a small readout box listing the time and every value.
- **Rolling drum:** the memory readout window's last digit rolls on a drum over 250ms (ease-out cubic) and always rests on a whole digit. Reduced motion jumps straight to the value.
- **Master Caution / Warning:** dark amber at rest; lights amber for unacknowledged caution findings and red for warnings. Pressing it acknowledges, and acknowledgement is stored per browser.
- **Scratchpad:** one-line messages under the annunciator row (amber caution, red-tinted error) cleared with CLR.

## Do's and Don'ts

### Do:
- **Do** put every display inside the bezel ring with the 4px ring border and 10px corners, on display glass (#040708 at night, #fbfbfa by day).
- **Do** take every colour from the root tokens, in markup and in canvases, so Day and Night stay one system.
- **Do** keep keys dark-capped with night legends in every display mode.
- **Do** letter every label, key and annunciation in Bahnschrift caps at 87.5% width, and keep sentences in Segoe UI Variable Text.
- **Do** colour by meaning: white measured, green engaged, cyan selectable, magenta configured target, amber caution or staged, red warning, ice blue prompt reading.
- **Do** put controls beside the state they change, as the engine keys sit beside the engine mode.
- **Do** stage settings as an amber temporary revision that ends only in INSERT or ERASE.
- **Do** keep Monitor and every CDU page free of page scroll at 1440x900, 960x1000 and 1280x720; scroll inside the glass instead.
- **Do** show unavailable readings as a dash, never as zero.

### Don't:
- **Don't** read "glass" as glassmorphism: no frosted blur, translucent panels or glow.
- **Don't** put the cover-glass sheen, gradients or inset shadows on the panel, keys or glareshield.
- **Don't** use cyan for anything that cannot be selected or edited, or magenta for a measured value.
- **Don't** set prose in caps or in DIN, or set numbers in monospace.
- **Don't** add a colour outside the signal set to tell elements apart; use position, rules and gray levels.
- **Don't** hide measured numbers behind a click; explanation may be on demand, readings may not.
