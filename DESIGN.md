---
name: NInfer Supervisor
description: A quiet monochrome control surface for one local inference engine, where every costly choice sits beside a live proof of what it buys.
colors:
  s-bg: "#111111"
  s-surface: "#191919"
  s-soft: "#202020"
  s-line: "#2b2b2b"
  s-line-strong: "#3b3b3b"
  s-ink: "#f2f2f2"
  s-ink-2: "#c8c8c8"
  s-gray: "#a1a1a1"
  s-gray-2: "#7c7c7c"
  s-gray-3: "#3e3e3e"
  s-accent: "#9b91ff"
  s-accent-ink: "#111111"
  s-ring: "rgba(155,145,255,.35)"
  s-bad: "#ff8378"
  s-wash: "rgba(242,242,242,.035)"
  backdrop: "rgba(0,0,0,.6)"
  day-s-bg: "#fafafa"
  day-s-surface: "#ffffff"
  day-s-soft: "#f2f2f3"
  day-s-line: "#e5e5e7"
  day-s-line-strong: "#c9c9ce"
  day-s-ink: "#111111"
  day-s-ink-2: "#3a3a3c"
  day-s-gray: "#6e6e73"
  day-s-gray-2: "#8e8e93"
  day-s-gray-3: "#d2d2d6"
  day-s-accent: "#5b4cdb"
  day-s-accent-ink: "#ffffff"
  day-s-ring: "rgba(91,76,219,.25)"
  day-s-bad: "#c4322b"
  day-s-wash: "rgba(17,17,17,.03)"
  day-backdrop: "rgba(40,40,40,.35)"
typography:
  s-page-title:
    fontFamily: "'Segoe UI Variable Display', 'Segoe UI', system-ui, sans-serif"
    fontSize: "28px"
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: "-0.02em"
  s-sheet-title:
    fontFamily: "'Segoe UI Variable Display', 'Segoe UI', system-ui, sans-serif"
    fontSize: "22px"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "-0.015em"
  s-major-reading:
    fontFamily: "'Segoe UI Variable Display', 'Segoe UI', system-ui, sans-serif"
    fontSize: "25px"
    fontWeight: 600
    lineHeight: 1.1
    letterSpacing: "-0.02em"
    fontFeature: "'tnum' 1"
  s-brand-mark:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "24px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "-0.04em"
  s-brand:
    fontFamily: "'Segoe UI Variable Display', 'Segoe UI', system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 600
    letterSpacing: "-0.01em"
  s-figure-input:
    fontFamily: "'Segoe UI Variable Display', 'Segoe UI', system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 600
    letterSpacing: "-0.01em"
    fontFeature: "'tnum' 1"
  s-claim:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "'tnum' 1"
  s-claim-figure:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 600
    fontFeature: "'tnum' 1"
  s-label:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 600
  s-lede:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
  s-body:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "'tnum' 1"
  s-message:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.6
  s-control:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 500
  s-button:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "13.5px"
    fontWeight: 600
  s-key:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 600
  s-help:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.5
  s-caption:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "12.5px"
    fontWeight: 400
    lineHeight: 1.4
  s-chart-label:
    fontFamily: "'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif"
    fontSize: "11.5px"
    fontWeight: 600
  s-flag:
    fontFamily: "'Cascadia Mono', 'Cascadia Code', Consolas, monospace"
    fontSize: "12px"
    fontWeight: 400
  s-well:
    fontFamily: "'Cascadia Mono', 'Cascadia Code', Consolas, monospace"
    fontSize: "11.5px"
    fontWeight: 400
    lineHeight: 1.7
rounded:
  s-mark: "2px"
  s-bar: "3px"
  s-reading: "5px"
  s-control: "8px"
  s-tag: "10px"
  s-pill: "11px"
  s-panel: "12px"
spacing:
  s-hair: "2px"
  s-inline: "8px"
  s-gap: "12px"
  s-index-pad: "14px"
  s-workspace-y: "16px"
  s-proof-pad-x: "18px"
  s-chrome-x: "20px"
  s-sheet-pad-x: "22px"
  s-column-gap: "28px"
  s-page-x: "40px"
  s-page-x-narrow: "16px"
components:
  s-button:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-ink}"
    typography: "{typography.s-button}"
    rounded: "{rounded.s-control}"
    padding: "0 16px"
    height: "36px"
  s-button-disabled:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-gray-2}"
  s-button-small:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-ink}"
    rounded: "{rounded.s-control}"
    padding: "0 12px"
    height: "30px"
  s-button-primary:
    backgroundColor: "{colors.s-accent}"
    textColor: "{colors.s-accent-ink}"
    typography: "{typography.s-button}"
    rounded: "{rounded.s-control}"
    padding: "0 16px"
    height: "36px"
  s-button-primary-disabled:
    backgroundColor: "{colors.s-soft}"
    textColor: "{colors.s-gray-2}"
  s-key:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-ink}"
    typography: "{typography.s-key}"
    rounded: "{rounded.s-control}"
    padding: "0 13px"
    height: "32px"
  s-key-stop:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-bad}"
  s-key-commit:
    backgroundColor: "{colors.s-accent}"
    textColor: "{colors.s-accent-ink}"
    typography: "{typography.s-key}"
    rounded: "{rounded.s-control}"
    padding: "0 13px"
    height: "32px"
  s-alerts-button:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-gray}"
    rounded: "{rounded.s-control}"
    padding: "0 12px"
    height: "32px"
  s-alerts-button-lit:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-ink}"
  s-alerts-button-problem:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-bad}"
  s-input:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-ink}"
    typography: "{typography.s-body}"
    rounded: "{rounded.s-control}"
    padding: "6px 11px"
    height: "36px"
  s-input-readonly:
    backgroundColor: "{colors.s-soft}"
    textColor: "{colors.s-ink-2}"
  s-input-disabled:
    backgroundColor: "{colors.s-soft}"
    textColor: "{colors.s-gray-2}"
  s-search:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-ink}"
    rounded: "{rounded.s-control}"
    padding: "7px 12px 7px 34px"
    height: "38px"
  s-composer:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-ink}"
    typography: "{typography.s-message}"
    rounded: "{rounded.s-tag}"
    padding: "10px 12px"
    height: "62px"
  s-segment:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-gray}"
    typography: "{typography.s-control}"
    rounded: "{rounded.s-control}"
    padding: "0 15px"
    height: "34px"
  s-segment-selected:
    backgroundColor: "{colors.s-ink}"
    textColor: "{colors.s-bg}"
  s-segment-selected-unsaved:
    backgroundColor: "{colors.s-accent}"
    textColor: "{colors.s-accent-ink}"
  s-tab:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-gray}"
    rounded: "{rounded.s-control}"
    padding: "0 13px"
    height: "32px"
  s-tab-selected:
    backgroundColor: "{colors.s-ink}"
    textColor: "{colors.s-bg}"
  s-tag:
    textColor: "{colors.s-ink-2}"
    rounded: "{rounded.s-tag}"
    padding: "0 8px"
  s-switch-on:
    backgroundColor: "{colors.s-ink}"
    rounded: "{rounded.s-pill}"
    width: "38px"
    height: "22px"
  s-switch-off:
    backgroundColor: "{colors.s-gray-3}"
  s-switch-on-unsaved:
    backgroundColor: "{colors.s-accent}"
  s-page-tab:
    textColor: "{colors.s-gray}"
    typography: "{typography.s-control}"
    padding: "0 12px"
    height: "52px"
  s-page-tab-current:
    textColor: "{colors.s-ink}"
  s-engine-line:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-ink}"
    padding: "5px 20px"
    height: "44px"
  s-live-band:
    backgroundColor: "{colors.s-bg}"
    textColor: "{colors.s-gray}"
    typography: "{typography.s-caption}"
    padding: "8px 20px"
  s-pad-line:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-ink}"
    typography: "{typography.s-help}"
    rounded: "{rounded.s-control}"
    padding: "9px 12px"
  s-pad-line-notice:
    backgroundColor: "{colors.s-soft}"
    textColor: "{colors.s-ink}"
  s-pad-line-bad:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-bad}"
  s-card:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-ink}"
    rounded: "{rounded.s-panel}"
    padding: "12px 16px 14px"
  s-sheet:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-ink}"
    rounded: "{rounded.s-panel}"
    padding: "16px 22px 22px"
  s-reading-label:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-ink}"
    typography: "{typography.s-caption}"
    rounded: "{rounded.s-reading}"
    height: "20px"
  s-index-link:
    textColor: "{colors.s-ink}"
    typography: "{typography.s-label}"
    rounded: "{rounded.s-control}"
    padding: "9px 12px 10px"
  s-index-link-current:
    backgroundColor: "{colors.s-soft}"
    textColor: "{colors.s-ink}"
  s-proof-panel:
    backgroundColor: "{colors.s-soft}"
    textColor: "{colors.s-ink}"
    typography: "{typography.s-claim}"
    rounded: "{rounded.s-panel}"
    padding: "14px 18px 13px"
  s-proof-bar:
    backgroundColor: "{colors.s-ink-2}"
    textColor: "{colors.s-soft}"
    typography: "{typography.s-chart-label}"
    rounded: "{rounded.s-bar}"
    height: "18px"
  s-proof-bar-rest:
    backgroundColor: "{colors.s-gray-3}"
    textColor: "{colors.s-ink}"
  s-save-bar:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-gray}"
    typography: "{typography.s-help}"
    padding: "12px 40px"
  s-alert:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-ink}"
    typography: "{typography.s-help}"
    rounded: "{rounded.s-control}"
    padding: "9px 12px"
  s-alert-bad:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-bad}"
  s-well:
    backgroundColor: "{colors.s-soft}"
    textColor: "{colors.s-ink-2}"
    typography: "{typography.s-well}"
    rounded: "{rounded.s-control}"
    padding: "10px 12px"
  s-user-message:
    backgroundColor: "{colors.s-soft}"
    textColor: "{colors.s-ink}"
    typography: "{typography.s-message}"
    rounded: "{rounded.s-tag}"
    padding: "9px 13px 11px"
  s-dialog:
    backgroundColor: "{colors.s-surface}"
    textColor: "{colors.s-ink}"
    rounded: "{rounded.s-panel}"
    padding: "16px 18px"
    width: "440px"
---

# Design System: NInfer Supervisor

## Overview

**Creative North Star: "Claim and Proof"**

The Supervisor is one quiet monochrome world on every page: Monitor, Chat, Connect, API, Diagnose, Settings and the chrome above them. A page is a neutral sheet of hairlines and sentences; every control that costs something sits beside a live proof of what it buys, so the cost of a choice is read before saving, never learned by restarting. Colour is almost absent: one neutral scale carries all structure, one violet accent marks the user's hand (what they changed, what they are about to commit, where focus is, and their own configured limits), and red appears only when something would fail or is failing.

Readings are drawn the same way: charts are ink and gray on a hairline scale, the current value sits in a small rounded label at the scale, and the desktop reserve is always the same diagonal hatch, on the memory bar, in a proof chart or across the memory history. Display modes are Auto (follows the OS, resolved before first paint), Day and Night; every token has both values.

It is calm on purpose. Lettering is Segoe UI Variable in sentence case, never condensed and never in capitals; numbers are tabular; approximate figures carry a `~`. Density comes from short rows, one-line bands and index lines that already show the current values, not from small type.

**Key Characteristics:**
- One neutral scale, near-white to near-black, mirrored between Day and Night; one violet accent; red only for failure. No amber, green or blue anywhere.
- Segoe UI Variable in sentence case at 400 and 600 (500 for unselected segments, tabs and page tabs); Cascadia Mono only for flags, paths, addresses, logs and request bodies.
- Hairline borders, 8px control radii, 12px cards, sheets and proof panels, 3px bar segments; flat except focus rings, the switch knob and the one floating dialog.
- Claim and proof: a control on the left, a soft panel on the right with one sentence and one small bar chart that redraws as the control moves.
- Unsaved changes are marked in violet at every level: the field, the row dot, the index line, the page tab, the save bar, the reserve marker on Monitor.

## Colors

A neutral scale in two mirrored modes with one violet accent and one red. Every value is a `--s-*` custom property on the root; Day overrides the same names, so a mode change repaints everything, including canvases, which read the same properties.

### Primary
- **Violet** (s-accent; day-s-accent): the user's hand, and nothing else. It may mean: keyboard focus (2px outline, or the accent border plus the Violet Ring on fields; a focused or dragged Monitor seam); a value the user changed but has not saved (the field's 1px inner accent border, the 7px dot after a row's label, the selected segment or switch turning violet, the context pool segment of the memory bar, `N unsaved` on an index line, `Unsaved` beside the Settings page tab, the 2px line along the top of the save bar, the `Unsaved settings` line beside the engine state, the next-start reserve marker on the Monitor memory scale); the filled part of a slider track; the commit action (Save, Load and restart, Send, Confirm); a model preset selected but not yet loaded; the user's own budget cuts on the Experts map; a token's hover outline; and the dot of the `n.` brand mark. Violet Ink (s-accent-ink) is the text on a violet fill.
- **Violet Ring** (s-ring; day-s-ring): the translucent violet around a focused field (3px), behind a focused slider thumb (4px), around the chat composer while a file is dragged over it, behind search hits, and as the text-selection colour.

### Neutral
- **Ground** (s-bg): the page, the top bar, the live band and the scratchpad area.
- **Surface** (s-surface): Monitor cards, page sheets, fields, buttons, segmented groups and tabs, the engine line, scratchpad lines, the save bar, alerts and the dialog.
- **Soft** (s-soft): proof panels, the current index line and hovered list rows, read-only and disabled fields, wells (code, logs, responses, launch commands), user chat messages, notice scratchpad lines, highlighted traffic rows, and the low end of the Experts ramp.
- **Line** (s-line) and **Line Strong** (s-line-strong): hairlines between rows, sections, cards and panes; the stroke of fields, buttons, tabs and segments; the chart value axis (Line Strong) and dotted grid.
- **Ink** (s-ink): text; the selected segment, tab or toggle fill; a switch that is on; the slider thumb ring; the context pool on the memory bar; the reserve hatch; the live dot; generation and memory lines on charts; the band and chat meters; the high end of the Experts ramp.
- **Ink 2** (s-ink-2): secondary text in wells and file names, the engine segment, proof bars, the saved next-start reserve marker.
- **Gray** (s-gray), **Gray 2** (s-gray-2), **Gray 3** (s-gray-3): help, captions, labels and the prompt-reading line on charts; disabled text, the weights segment and memory-range ticks; the off switch, the other-apps segment and the rest of a proof bar, in that order.
- **Wash** (s-wash): the faint fill under the generation line and the memory line.
- **Backdrop** (backdrop): the dimmed page behind the open dialog.

### Failure
- **Red** (s-bad; day-s-bad): something will fail or is wrong. A plan that would not start (the memory bar status, a proof claim, the red reserve hatch and the free-memory outline); the engine inside the desktop reserve (the scale hatch, dashed line, band label and reading label on Monitor turn red); an invalid field or request body; the error list above the save bar; a red alert or scratchpad line; the alerts button when a finding is a problem; an offline live dot; a Problem finding; error log lines; the Stop button. Never decoration and never a warning-in-advance tint.

### Named Rules
**The One Hand Rule.** Violet marks only what the user is doing or has set: focus, an unsaved change, the chosen slider value, the commit action, the user's own budget cut. A resting reading is ink and gray. If an element is not focusable, not changed, not a commit and not the user's own setting, it is not violet.

**The Ink Is Chosen Rule.** A saved choice is solid ink (the selected segment, tab or toggle, the switch that is on); the same choice turns violet only while it is unsaved. The two never mean the same thing.

**The Red Means It Fails Rule.** Red is spent only on a state that would fail, is failing or has failed. There is no amber and no caution tint; severity below a problem is a word in ink or gray.

**The No Signal Hues Rule.** There is no amber, green or blue in the system. Script state classes still carry historical names (green, amber, cyan, magenta, ice); they resolve to Ink, Ink, Violet, Violet and Gray, and red to Red. A new state uses those four values, never a new hue.

**The Mirror Rule.** Day and Night are the same scale reversed, not two palettes; every `--s-*` token has both values, and nothing exists in only one mode.

## Typography

**Display Font:** Segoe UI Variable Display (with Segoe UI, system-ui)
**Body Font:** Segoe UI Variable Text (with Segoe UI, system-ui)
**Label/Mono Font:** Cascadia Mono (with Cascadia Code, Consolas)

**Character:** One family speaking in sentences. The Display optical size takes only titles, the brand name, the major Monitor readings and the large figure inputs; everything else is Text. Numbers use tabular figures across the whole page.

### Hierarchy
- **Page title** (600, 28px, 1.15, -0.02em, Display): Settings section headings (Model, Memory, Requests); 26px when the content column is 760px or narrower.
- **Sheet title** (600, 22px, 1.2, -0.015em, Display): the title of the Chat, Connect, API and Diagnose sheets; 20px at 560px and narrower. The empty chat greeting uses 20px Display.
- **Major reading** (600, 25px, 1.1, -0.02em, Display): Used memory and the two rates on Monitor; 22px at 560px wide, 20px or 17px when the card is short.
- **Brand mark** (700, 24px, -0.04em) `n.` with its violet dot, beside **Brand** (600, 18px, Display) `NInfer`; the name hides at 820px and narrower.
- **Figure input** (600, 17px, Display, tabular): the context capacity field and number fields in a Settings control row.
- **Claim** (400, 16px, 1.45): the sentence in a proof panel; its figures 600 at 18px, its connecting words in Gray.
- **Label** (600, 15px): a setting's name, an index section name, a preset name, a Monitor card heading, a subhead on a sheet, the engine model name. **Lede** (400, 15px, Gray, 72ch): one line under a page title.
- **Body** (400, 14px, 1.45): fields, segments and tabs (500), page tabs (500), finding titles (600). **Message** (400, 14px, 1.6): chat messages and the composer, up to 100ch.
- **Button** (600, 13.5px): Settings buttons; 12.5px on small buttons. **Key** (600, 13px): chrome and sheet buttons.
- **Help** (400, 13px, 1.5, Gray, 72-90ch): help under a control, finding bodies, notes, scratchpad lines, alerts. The page base is 13px/1.5.
- **Caption** (400, 12.5px, Gray): index summaries, readout labels, band labels, table heads, slider ends; chart captions at 12px; chart axis figures at 11px 500.
- **Chart label** (600, 11.5px): the token count printed inside a proof bar when it fits.
- **Flag** (Cascadia Mono 12px, Gray): the `--flag` after a label, file names, preset paths, the model badge; 13px in path fields and the API address. **Well** (Cascadia Mono 11.5-12.5px, 1.5-1.75): logs, launch commands, request and response bodies, code blocks.

### Named Rules
**The Sentence Case Rule.** Every word on the page is in sentence case at normal width. No capitals, no condensed faces, no letter-spacing on labels; the only tracking is the slight negative tightening on Display sizes.

**The Machine Text Rule.** Monospace is only for what a machine reads: flags, file names, paths, addresses, logs, launch commands, request and response bodies, code. Figures stay in Segoe with tabular numerals.

## Layout

- **Chrome.** A 52px top bar on Ground with 20px sides: the brand, text page tabs (Monitor, Chat, Connect, API, Diagnose, Settings), then at the right the alerts button (`No alerts`, `N alerts`, or `N problems` in red), the display-mode button, the Explain button and the live dot with its word. Under it the **engine line** on Surface with a hairline below, split into columns by hairlines: engine state with Up time and its buttons (Start or Restart and Stop), the model with Presets, the running plan in plain words (`Running: pool 12.00 GiB · 256K context · 4 at once · KV fp8 · dflash2, 7 drafts`, hidden while the launch arguments are unknown), and the API address with a copy icon pushed right. Scratchpad messages follow as bordered lines.
- **Live band.** On Chat, Connect, API and Diagnose a one-line band on Ground with a hairline below shows Memory (used of total, an 84 by 6px ink meter, free), Generating, Reading and Running, with a `Monitor` link at the right. It is hidden on Monitor and Settings.
- **Workspace.** The area under the chrome scrolls on its own with 16px by 20px padding (none on Settings). Monitor and the one-screen sheets (Chat, API, Diagnose log and Experts) do not page-scroll at desktop sizes; their logs, lists and maps scroll inside.
- **Monitor.** Four Surface cards with a 12px gap: GPU memory, Rates, Calling apps and Alerts. Below 1180px they stack. From 1180px memory and rates fill a left column (1.55fr) beside a right column of calling apps over alerts (at least 380px); the gaps between cards are draggable seams that show a short gray grip on hover and a 2px violet line while focused or dragged, and the split is remembered. A memory or rates card is a three-column body: a 66px value scale, the history, and a 172px readout column behind a hairline. Explain mode lets the cards grow and drops the seams. At 820px and narrower the readouts move above the chart in four columns, two at 560px.
- **Sheets.** Chat, Connect, API and Diagnose are a single Surface sheet, up to 1040px wide (1240px for wide sheets; full width for Chat), centered, with a title line over a hairline. Chat is the message log beside a 350-430px Reply and Parameters column behind a hairline (one pane with a Parameters toggle at 980px and narrower). API is an endpoint index (220-280px) beside the request and response (stacked at 880px). Diagnose findings run in two columns from 1280px.
- **Settings.** Two columns: the section index (232-288px, a hairline to its right) and the work column, a scrolling body with 40px sides over a fixed save bar. The index has a search field, then one link per section (Model, Memory, Requests, Access, Web frontends, Logs and launch), each a name over a one-line summary of the current draft values, and the config file name at the foot in mono. Sections stack on hairlines: title and lede, then rows. A **row** is a two-column grid (control up to 520px, proof the rest) with 12px row gap, 28px column gap and a hairline above; at a 760px work column (container query) the proof drops under its control. At 880px the page stacks, the body flows with 16px sides and the save bar sticks to the bottom of the window.
- **Search** filters Settings rows by label, flag, help, keywords and current value; a status line counts matches, index summaries become match counts, unmatched index lines fade to 45%, and matched words are marked with the Violet Ring. `/` focuses search; Escape clears it.
- **Responsive chrome.** At 1240px and narrower the running plan takes its own row of the engine line under a hairline. At 820px the top bar wraps: page tabs take a full-width row at 40px and 13px, engine-line columns shrink to 40px, and the engine column takes the full width.
- **Rhythm.** 2px between bar segments and index links, 8px inline, 12px between rows, cards and blocks, 20px chrome sides, 22px sheet sides, 28px between control and proof, 40px Settings sides (16px narrow).

## Elevation & Depth

Flat. Depth is a fill step, not a shadow: Ground, then Surface for cards, sheets and things you touch, then Soft for things you read about (proof panels, wells, the current index line, user messages). Borders are 1px hairlines. The shadows that exist are functional: focus rings, unsaved and invalid inner borders, the save bar's top line, a hollow ring for the paused live dot, the switch knob, and the dialog, the only floating surface.

### Shadow Vocabulary
- **Focus ring** (`box-shadow: 0 0 0 3px var(--s-ring)` with an accent border): fields, search, the composer while a file is dragged over it; `0 0 0 4px` on a slider thumb. Other focusable elements take a 2px accent outline offset 2px.
- **Unsaved field** (`box-shadow: inset 0 0 0 1px var(--s-accent)`): a changed field. Invalid uses the same with Red.
- **Unsaved bar** (`box-shadow: inset 0 2px 0 var(--s-accent)`): the save bar while anything is unsaved.
- **Current list row** (`box-shadow: inset 0 0 0 1px var(--s-line)`; Line Strong in the API index): the current index line or endpoint on Soft.
- **Paused dot** (`box-shadow: inset 0 0 0 1.5px var(--s-gray)` on a transparent dot).
- **Switch knob** (`box-shadow: 0 1px 2px rgba(0,0,0,.25)`).
- **Dialog** (`box-shadow: 0 0 0 1px var(--s-line-strong), 0 18px 50px var(--dialog-shadow)`, with `--dialog-shadow` at 50% black by Night and 18% by Day) over the Backdrop.

### Named Rules
**The Fill Step Rule.** Depth is Ground, Surface, Soft. A new layer gets a fill step and a hairline, never a drop shadow; the confirmation dialog is the single exception.

## Shapes

Gently rounded and thin-lined. Fields, buttons, tabs, segmented groups, index links, alerts, scratchpad lines and wells take 8px corners; Monitor cards, sheets, proof panels and the dialog 12px; user messages and the chat composer 10px; tags (client, preset and sheet tags) are 10px outlined pills; the switch is a full pill (38 by 22px); bar segments and meters 3px; legend swatches, slider ticks and the reserve hatch 2px; the reading label on a chart scale has 5px corners. The reserve is drawn, not iconified: a 135-degree hatch of 2px ink lines on 5px repeats, outlined in ink on bars, and a 1px ink hatch on chart scales and across the memory history; the same hatch in red means the plan is short or the engine is inside it. Icons are few and stroked: a 16px magnifier in Settings search and 14px inline line icons (1.4px stroke) for display mode, Explain and copy.

## Components

### Buttons
Quiet outlined buttons; one filled violet button per action group.
- **Shape:** 8px corners, 1px Line Strong stroke; 36px tall in Settings (30px small), 32px in the chrome and sheets (30px in the engine line, 26px tiny).
- **Default:** Surface fill, Ink text at 600. **Commit:** Violet fill and stroke, Violet Ink text: Save, Load and restart, Send, Confirm. Nothing else is a violet fill.
- **Hover / Active:** the stroke darkens to Gray 2; pressed moves 1px down. **Disabled:** Gray 2 text, Line stroke, weight 500; a disabled commit falls back to Soft.
- **Stop:** an outlined button with red text at rest, because it ends every running request.
- **Chrome:** display mode and Explain are 32px outlined buttons with an icon (Explain takes an Ink stroke while on). The alerts button is a 70px-minimum outlined box in Gray at 500 reading `No alerts`; with new findings it reads `N alerts` with an Ink stroke at 600, or `N problems` with a red stroke and text; pressing it acknowledges them and opens Diagnose.
- **Text links** (Presets, Evidence, Set, Monitor): Ink at 600 with a Line Strong underline that darkens to Ink on hover.

### Tabs, choices and segments
Sheet tabs and choices (Diagnose sections, Thinking, API examples and response views, log filters) are outlined 8px buttons, 30-32px tall, Gray at 500; the chosen one is solid Ink with Ground text at 600, with any count in the same colour at 72%. Settings' segmented control is one 8px outlined group, 34px tall; its selected option is solid Ink, Violet while the row holds an unsaved change. Concurrent requests is a fixed 1-8 strip of 40px cells.

### Inputs / Fields
- **Style:** Surface fill, 1px Line Strong stroke, 8px corners, 36px tall, 14px Ink text with tabular figures; selects carry a drawn chevron in Gray. Every page uses this field.
- **Hover:** stroke Gray 2. **Focus:** accent stroke plus the 3px Violet Ring. **Unsaved:** accent stroke plus a 1px inner accent line. **Invalid:** the same in Red. **Read-only / Disabled:** Soft fill, Ink 2 / Gray 2 text.
- **Sliders:** a 4px track filled Violet up to the value and Line beyond; an 18px Surface thumb with a 2px Ink ring; a 2px by 12px Gray tick marks the running value.
- **Switch:** 38 by 22px pill, Gray 3 off, Ink on, Violet on while unsaved; a 16px knob slides 150ms.
- **Composer:** the chat input is a 10px-cornered field at 14px/1.5, 62px tall and resizable up to 30vh.

### Navigation
- **Page tabs:** text tabs 52px tall, Gray at 500, Ink on hover; the current page is Ink at 600 over a 2px Ink underline. The Diagnose count is an Ink pill with Ground text; `Unsaved` beside Settings is plain Violet text.
- **Live dot:** a 9px Ink dot beside `Live`; Red while offline; a hollow Gray ring while paused.
- **Section index and endpoint index:** a name over a summary; hover and current on Soft, current with an inner hairline.

### Cards and sheets
- **Monitor card:** Surface, 1px Line, 12px corners, 12px 16px 14px padding; a heading line of a 15px title, a Gray source and a Gray range at the right.
- **Sheet:** Surface, 1px Line, 12px corners, 16px 22px 22px padding; a 22px Display title over a hairline with the page name in Gray at the right.
- **Wells:** Soft, 1px Line, 8px corners, Ink 2 mono text; logs, launch commands, responses, code blocks and message details.

### Scratchpad
One bordered line per message under the engine line: Surface, 1px Line, 8px corners, 13px Ink text, an optional small clear button. A notice (disconnected, memory guidance) takes Soft; an error takes a red stroke and red text.

### Signature: the monochrome chart
Memory and rates draw on one grammar. A hairline value axis stands at the right of a 66px scale column with Gray figures; the current reading sits in a small rounded label (Surface, 1px stroke, 5px corners, 600 12.5px) joined to the axis. The history runs on a dotted Line Strong grid over a relative time axis (`−10m 0s` to `now`). Memory is a 1.5px Ink line over the Wash, with Gray 2 ticks for the range inside each pixel and an Ink trend stroke on the scale when it is moving. Generation is the heavier Ink line (1.75px) over the Wash, read off the left scale; prompt reading is a thin Gray line (1.25px) read off Gray figures at the right; the legend names each scale (`left scale`, `right scale`) beside a matching swatch. Engine events are dashed Ink verticals with a small triangle. The desktop reserve is an Ink hatch on the scale and a faint hatched band above a dashed Gray line labelled `Desktop reserve` across the memory history; all of it, and the reading label, turns red only while the engine is inside the reserve. The next start's reserve is a small Ink 2 marker on the scale, Violet while Settings holds an unsaved value. Figures and bars travel 380ms on an ease-out cubic.

### Signature: claim and proof row
Each memory control (Context capacity, Max context with KV format, Concurrent requests, Desktop reserve) sits beside a proof panel on Soft with 12px corners: one claim sentence with its figures in 600 at 18px, one small horizontal bar chart (18px; 28px column cells for requests), and a two-ended caption. Dragging any control redraws every proof and the memory bar in the same frame; bars ease 150ms, and reduced motion jumps. If the plan would not start the claim turns Red, other charts dim to 40% and claims prefix `If it fit:`. With no running engine every proof says `Start the engine to see what this buys.`

### Signature: the whole-card memory bar
At the top of Settings Memory: a head line with the fit verdict at the right, a 14px bar of the whole card split by 2px gaps into model weights (Gray 2), engine (Ink 2), context pool (Ink; Violet while capacity is unsaved), other apps (Gray 3) and free (an outlined Surface box holding the desktop reserve hatch at its right end), a legend of 10px swatches with values, and a basis note listing what is approximate until restart.

### Findings and severity
Diagnose findings are rows on hairlines: a 96px severity word, a 14px title at 600, a Gray body and an Ink action, with `Evidence` as a text link. Severity is a word, not a hue: **Problem** in Red, **Review** and **Note** in Ink, **Info** in Gray. Log lines follow the same code: errors Red, warnings Ink at 600, requests Ink, info Ink 2, detail Gray.

### Experts map
One row per text layer and one column per expert, drawn as a single ramp from Soft to Ink for both decode and prefill; the bars at its right are Ink. The user's budget cuts are Violet (solid per layer, dashed for the even split), because they are the user's own setting. A stale map fades to 40%.

### Save bar
A Surface strip under the Settings body with a hairline on top: Discard, a status in help type, Save (commit) and Save and restart (outlined, shown only when the Supervisor manages the engine). While anything is unsaved a 2px Violet line runs along its top. Validation errors gather in a Red outlined box above it.

## Do's and Don'ts

### Do:
- **Do** build every surface from the `--s-*` tokens, in Day and Night, and read canvas colours from the same properties.
- **Do** put a live proof beside any control whose cost the user cannot otherwise see, with one sentence and one small bar chart from the same reading.
- **Do** mark an unsaved change in Violet at the field, the row, the index line, the page tab, the save bar and any marker that shows it, and return it to Ink once saved.
- **Do** keep the current value visible: in the index summary, beside each control (`Running 12.00 GiB`), and as the running plan on the engine line.
- **Do** draw charts in Ink and Gray on a hairline scale, name each scale in the legend, and show the current reading in the rounded label.
- **Do** draw the desktop reserve as the hatch everywhere it appears.
- **Do** state severity as a word: Problem, Review, Note, Info.
- **Do** mark approximate figures with `~` and say why a figure is unavailable instead of showing zero.
- **Do** separate with hairlines and fill steps; use 8px corners on controls and wells and 12px on cards, sheets and proof panels.
- **Do** let bars ease 150ms and readings travel 380ms, and jump under reduced motion.

### Don't:
- **Don't** use Violet for a resting reading, a heading, a status or decoration; the only violet fills are Save, Load and restart, Send and Confirm.
- **Don't** set anything in capitals, condensed type or letter-spaced labels.
- **Don't** add amber, green, blue or any second hue; red is only for a plan, input or state that fails.
- **Don't** give a state class a new colour; historical class names resolve to Ink, Gray, Violet or Red.
- **Don't** use monospace for numbers or sentences.
- **Don't** add drop shadows, gradients, bezels, sheen or blur to cards and panels; the dialog is the only floating surface and the reserve hatch is the only pattern.
- **Don't** hide a cost behind a restart: if a choice changes memory, its proof updates before Save.
