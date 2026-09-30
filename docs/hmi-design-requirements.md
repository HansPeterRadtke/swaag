# Human-machine interface design requirements

This document is the stable, implementation-neutral contract for Swaag user
interfaces and protocol clients. It does not prescribe a particular web or
native toolkit. It turns the GUI recordings into testable requirements so a
future adapter does not replace complete durable state with an attractive but
incomplete display.

The recording evidence includes every GUI/UI transcript through 2026-09-16, including the voice-interface recording that broadens this scope from graphical UI to user interfaces generally. The
meaningful opening of `Recording 2026-08-27 18-49-13-262` explicitly requires
readable control text and physical, real-world size research; its remaining
repeated "I don't know" text is an STT hallucination and is not design input.
Earlier GUI recordings additionally require complete information, explicit
validity/freshness, action-oriented controls, careful destructive actions,
touch and pointer support, and measured rather than assumed ergonomics.

## Information integrity

- The durable task, event, result, attachment, and artifact records remain the
  source of truth. A screen is a derived view, never a replacement source.
- Never hide meaningful information with blind clipping, ellipses, fixed row
  counts, or a permanently shortened status. Wrap and reflow ordinary text.
- Density may justify a clearly labeled summary, but it must disclose omitted
  item counts and offer an action such as **Show complete result**, paging, or
  raw artifact access. The complete view must remain navigable and copyable.
- Use horizontal scrolling only for intrinsically two-dimensional content such
  as a data table or diagram. Do not force ordinary prose into two-dimensional
  scrolling. This follows WCAG 2.2 reflow guidance:
  <https://www.w3.org/WAI/WCAG22/Understanding/reflow.html>.
- Show freshness, validity, uncertainty, source, and failure state next to the
  value they qualify. A stale value must not look current. De-emphasize routine
  healthy detail instead of removing it, and make anomalies actionable.
- Semantic importance and user relevance come from the model-backed status or
  presentation stage. Exact timestamps, liveness, queue state, identifiers,
  counts, and integrity facts come from canonical mechanical state.

## Typography and physical size

Pixels alone are not a physical-size specification. Record the target device's
physical display dimensions, native resolution, effective scaling, viewing
distance, and input modes. Measure rendered controls on the actual hardware;
do not infer millimeters from nominal CSS pixels or device-independent units.

- Respect platform text settings and support at least 200 percent text resize
  without loss of content or functionality. At a 320 CSS-pixel viewport,
  content must reflow without two-dimensional scrolling except for genuine 2D
  data. See WCAG 2.2 Resize Text and Reflow:
  <https://www.w3.org/WAI/WCAG22/Understanding/resize-text.html> and
  <https://www.w3.org/WAI/WCAG22/Understanding/reflow.html>.
- Avoid light font weights and decorative faces for operational data. Use a
  readable text face, clear hierarchy, sufficient line spacing and contrast,
  and platform-supported dynamic type. Test every supported accessibility size
  rather than treating the default preview as evidence. Apple gives the same
  adaptive typography guidance:
  <https://developer.apple.com/design/human-interface-guidelines/typography>.
- WCAG 2.2 AA requires pointer targets at least 24 by 24 CSS pixels, subject to
  defined exceptions. Treat that as a conformance floor, not the preferred
  touch design. The enhanced criterion is 44 by 44 CSS pixels:
  <https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html>.
- For touch-first surfaces, start from a 48 by 48 dp hit area, approximately
  9 mm, with spacing between targets, then test the real device. Android cites
  a recommended 7 to 10 mm range:
  <https://support.google.com/accessibility/android/answer/7101858>.
  Windows independently recommends about 7.5 mm and says frequency, error
  consequence, position, form factor, and finger posture can require more:
  <https://learn.microsoft.com/en-us/windows/apps/develop/input/guidelines-for-targeting>.
- A visible icon may be smaller than its hit area. Enlarging invisible hit
  bounds must not create overlapping targets or misleading focus indicators.

No one point size, control width, or thumb-zone rule is universal. Phone,
tablet, desktop, kiosk, vehicle, and distant-display use have different viewing
and input constraints. Placement hypotheses must be measured with intended
users, including left- and right-handed use, rather than encoding a presumed
right-thumb layout.

## Controls and labels

- Use sentence-case, action-oriented labels that state the outcome: **Resume
  task**, **Show complete history**, or **Delete task**, not vague labels such
  as **OK**, **Submit**, or an unexplained symbol. The GOV.UK button guidance
  provides tested examples: <https://design-system.service.gov.uk/components/button/>.
- Keep the accessible name equal to or beginning with the visible label so
  speech-input users can name the control. Expose the correct name, role, value,
  state, and focus semantics. See WCAG Label in Name and Name, Role, Value:
  <https://www.w3.org/WAI/WCAG22/Understanding/label-in-name.html> and
  <https://www.w3.org/WAI/WCAG22/Understanding/name-role-value.html>.
- Use icon-only controls only for established, unambiguous actions and still
  provide an accessible name. Pair novel, irreversible, or safety-relevant
  symbols with visible text. Shape, location, or color must not be the only cue.
- Present one clear primary action per decision context. Base prominence and
  placement on purpose, frequency, urgency, accidental-activation risk, and
  observed workflows, not decoration or a universal toolbar convention.
- Separate destructive controls from frequent controls. State the consequence
  in the label and confirmation, and make irreversible legal, financial, or
  data-changing actions reversible, checked, or explicitly confirmed. Do not
  use red as the sole warning signal:
  <https://www.w3.org/WAI/WCAG22/Understanding/error-prevention-legal-financial-data.html>
  and <https://www.w3.org/WAI/WCAG22/Understanding/use-of-color.html>.
- Support touch, keyboard, pointer, switch, and assistive-technology operation.
  Never make hover the only route to an action or required information.

## Tooltips and contextual help

Tooltips supplement a control; they never contain its sole identity, essential
instructions, or an action. The trigger must work on keyboard focus as well as
pointer hover. Hover/focus content must be dismissible, hoverable, and remain
available until the user dismisses it or the trigger becomes invalid:
<https://www.w3.org/WAI/WCAG22/Understanding/content-on-hover-or-focus.html>.

Use `role="tooltip"` and associate the description with its trigger where the
platform supports that pattern. A tooltip itself does not receive focus. If the
surface contains links, buttons, inputs, or other interaction, use a popover or
dialog pattern instead. On touch-only devices, provide an explicit tap-accessible
help affordance; a hover-only tooltip does not exist. The ARIA Authoring
Practices tooltip pattern is still marked work in progress, so validate actual
browser and assistive-technology combinations rather than treating the example
as proof: <https://www.w3.org/WAI/ARIA/apg/patterns/tooltip/>.

## Operational data and status

- Display units explicitly. Keep precision stable and justified by the source
  measurement; do not imply accuracy that the source does not have.
- Align comparable values and units, and use tabular numerals where supported,
  so changes can be scanned without destroying the semantic reading order.
- Couple progress, waiting, input-required, cancellation, failure, and stale
  states to exact worker evidence. Programmatically expose status changes
  without stealing focus, following WCAG Status Messages:
  <https://www.w3.org/WAI/WCAG22/Understanding/status-messages.html>.
- Preserve the user's primary question in the visual hierarchy. Diagnostic
  detail may expand below it, but must not displace or obscure the answer.

## Required design and validation process

Before designing a screen, record:

1. The user's purpose and the decisions the screen must support.
2. Inputs, outputs, sources, freshness, uncertainty, and failure modes.
3. Information importance, update frequency, interaction frequency, and the
   consequence of omission or accidental activation.
4. Target hardware dimensions, resolution, scaling, viewing distance,
   orientation, lighting, and supported input/assistive modalities.
5. Hypotheses that require observation rather than assumption, such as control
   placement, one-handed reach, information density, or alert prominence.

Prototype and test on representative physical hardware. Cover minimum and
maximum supported text scale, narrow/reflowed and wide layouts, portrait and
landscape where applicable, keyboard-only use, screen-reader semantics,
pointer and touch operation, left- and right-handed use, high zoom, long and
localized labels, stale/error states, and maximum realistic data volume. Use
automated accessibility checks as diagnostics, then perform manual checks and
task-based user tests. Record discrepancies between design-system claims and
observed behavior. Usage telemetry may validate hypotheses only when it follows
the project's privacy and retention policy.

## Acceptance checklist

An HMI change is not complete until evidence shows all applicable checks pass:

- Complete meaningful information is reachable without guessing that clipped
  content exists; every summary or page states its scope and recovery action.
- Text reaches 200 percent and the 320 CSS-pixel reflow test without overlap,
  clipping, hidden functionality, or ordinary two-dimensional scrolling.
- Pointer targets meet the WCAG 24 by 24 CSS-pixel floor; touch-first targets
  are physically measured near the platform's 7 to 10 mm guidance or have a
  documented, tested reason for differing.
- Every control has an action-specific visible label or a justified standard
  icon, a matching accessible name, correct role/state, visible focus, and
  keyboard and touch operation.
- Destructive actions are separated, consequence-labeled, non-color-dependent,
  and reversible, checked, or confirmed as appropriate.
- Required help is available without hover; tooltip and interactive-popover
  behavior is tested with the supported browser/platform combinations.
- Units, precision, freshness, uncertainty, source, and failure state remain
  understandable at every supported layout and text scale.
- The implementation is tested on the recorded target hardware, not only a
  desktop emulator or screenshot.

WCAG 2.2 is the baseline web standard, not proof that every user need is met:
<https://www.w3.org/TR/WCAG22/>. Platform guidance and actual user/device tests
may justify stricter requirements; any exception must preserve information and
be documented with measured evidence.

## Voice-interface requirements

These requirements are conditional on a SWAAG client actually offering voice/audio interaction; they are not a requirement that the core agent runtime itself ship a microphone UI. When a voice-capable client is in scope, the September 16 recording and current Infra `gui/voice_interfaces.md` make the following behaviors mandatory:

- Preserve a manual fallback. Automatic transcription, turn detection and automatic sending must be disableable; the user can type from scratch, explicitly transcribe buffered audio, edit the resulting text and manually send it.
- Treat user edits as high-confidence transcription evidence. Track the final edit delta rather than raw keystrokes, preserve typed exact spellings such as paths/model names, and keep any later vocabulary-learning analysis semantic rather than deterministic.
- Show the current unsent transcript in a clearly editable text field and distinguish it from received agent messages. Received agent answers are not editable by default.
- Show operational state the user needs to judge whether interaction will work: microphone activity, server reachability, speech-to-text/model availability, and clear failure state. Manual text entry/sending remains usable when speech-to-text fails.
- Do not send partial transcription accidentally. A voice UI is an intermediate control layer that determines when captured speech/text is ready for the working agent.
- Spoken-response interruption is semantic. Voice activity alone does not prove interruption intent; acknowledgements, laughter, coughs, background voices and short commands differ. Retain explicit player controls as the reliable fallback.
- Every exposed audio recording/player supports play/pause/resume, stop, seeking/timeline, configurable short jumps, and replay of both received audio and the user's own unsent recording. Show total duration, current position and remaining time in stable positions, updating duration as streaming media grows.
- Voice automation is progressively degradable: fully manual record/transcribe/edit/send, live transcription with manual send, automatic voice-activity transcription with configurable send, then continuous conversational mode. Wake-word detection is separate and optional.
- Re-research current STT/TTS/audio-LLM capabilities before implementation. Benchmark vocabulary/hotword prompting, speaker handling, edit-derived adaptation and interruption-intent classification on the actual target devices and environments.

A native/browser/mobile/watch voice UI is not currently part of the core SWAAG daemon. Therefore these requirements are an acceptance contract for future or external clients, not evidence that the current core runtime is missing agent semantics.

## Domain-specific guideline scope

Trading/finance recordings are conditional domain authority. They do not require the general SWAAG harness to implement trading. If a SWAAG product later claims that capability, its client/worker package must additionally satisfy the current Infra trading rules for point-in-time data, leakage prevention, train/validation/test separation, fees/costs, formal claim evaluation and domain-specific evidence. The benchmark discipline in those recordings remains useful generically, but their market semantics are not core-agent requirements.
