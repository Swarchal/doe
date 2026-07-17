/* DoE planner — a thin, dependency-free UI over the doe-service API mounted at /api.
 * State is a single design document (the Design.to_dict wire format); every step
 * round-trips it through the service unchanged. */

"use strict";

const state = {
  design: null,           // current design document
  responseNames: ["response"],  // ordered measurement names (cap MAX_RESPONSES)
  responseName: "response",     // the *active* response driving the step-3 results view
  planType: null,         // the plan type the user picked ("ccd"/"bb"/"ff2"/"dopt"); null once loaded
};

/* Cap on simultaneously-tracked measurements — matches doe-service's own
 * `Limits.max_goals` (POST /v1/optimize/desirability caps goals at 8), so a design with
 * more measurements than the balance view could ever use is refused early and clearly. */
const MAX_RESPONSES = 8;

/* Non-factor run columns that are bookkeeping, not measured responses: `std_order` is the
 * standard-order index that `randomize` preserves so results can be re-joined. Excluded when
 * detecting response columns in a loaded design. */
const RESERVED_RUN_COLUMNS = new Set(["std_order"]);

/* Perceptually uniform colormap for the fitted-surface contour. Marker overlays are
 * white-filled with a dark outline so they stay visible on both the dark (low) and
 * bright (high) ends of the scale. */
const CONTOUR_COLORSCALE = "Viridis";

/* Shown in a plot slot when Plotly failed to load (offline / CDN blocked) — the numbers on the
 * page still come from the API, so only the visualisation is missing. */
const OFFLINE_PLOT_MSG =
  "<p class=\"plot-fallback\">Plot unavailable offline (Plotly.js loads from a CDN) — " +
  "the numbers on the page are unaffected.</p>";

const $ = (id) => document.getElementById(id);

/* Draw (or redraw) a Plotly figure sized to its slot. Plotly measures the container only at
 * draw time and its `responsive: true` handler only reacts to *window* resizes — a figure
 * drawn while its slot is hidden or still settling gets Plotly's 700px default and stays
 * wrong until the window happens to resize. Watching the slot itself with a ResizeObserver
 * closes that gap: the figure is resized the moment its slot ends up with a different size,
 * whatever caused the change. */
const plotResizeObserver = typeof ResizeObserver === "undefined" ? null :
  new ResizeObserver((entries) => {
    for (const { target } of entries) {
      // Only live Plotly figures in a measurable (not display:none) slot; the width check
      // makes the resize a no-op-free fixpoint rather than an observe/resize loop.
      if (target._fullLayout && target.offsetWidth > 0 &&
          Math.abs(target._fullLayout.width - target.offsetWidth) > 1) {
        Plotly.Plots.resize(target);
      }
    }
  });

function drawPlot(el, traces, layout) {
  Plotly.react(el, traces, layout, { responsive: true, displaylogo: false });
  if (plotResizeObserver) plotResizeObserver.observe(el); // observing twice is a no-op
}

async function api(path, body) {
  const res = await fetch(`/api/v1${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON error body */ }
  if (!res.ok) throw new Error(apiErrorMessage(data, res.status));
  return data;
}

function apiErrorMessage(data, status) {
  if (data && data.error) {
    const extra = (data.error.errors || []).join("\n");
    return data.error.message + (extra ? `\n${extra}` : "");
  }
  if (data && data.detail) {
    return Array.isArray(data.detail)
      ? data.detail.map((d) => d.msg || JSON.stringify(d)).join("\n")
      : String(data.detail);
  }
  return `Request failed (HTTP ${status})`;
}

/* Toggle a button into a disabled "busy" state with a spinner and temporary label,
 * restoring its original text when done. Reusable across the async action buttons.
 *
 * `escalations` is an optional list of `{ after, label }` — while the button stays busy the
 * label is swapped to each `label` after `after` ms, so a slow request reassures the user
 * ("Creating plan…" -> "Still working…") instead of looking hung. */
function setBusyText(button, label) {
  button.innerHTML = `<span class="btn-spinner" aria-hidden="true"></span>${label}`;
}

function setButtonBusy(button, busy, busyLabel, escalations) {
  // Clear any pending escalation timers from a previous busy period.
  (button._busyTimers ?? []).forEach(clearTimeout);
  button._busyTimers = null;

  if (busy) {
    if (button.dataset.label === undefined) button.dataset.label = button.textContent;
    button.disabled = true;
    button.classList.add("busy");
    setBusyText(button, busyLabel ?? "Working…");
    if (escalations && escalations.length) {
      button._busyTimers = escalations.map(({ after, label }) =>
        setTimeout(() => setBusyText(button, label), after));
    }
  } else {
    button.disabled = false;
    button.classList.remove("busy");
    if (button.dataset.label !== undefined) {
      button.textContent = button.dataset.label;
      delete button.dataset.label;
    }
  }
}

function showError(id, err) {
  const el = $(id);
  el.textContent = err instanceof Error ? err.message : String(err);
  el.hidden = false;
}
function clearError(id) { $(id).hidden = true; }

const fmt = (x, digits = 3) => (x === null || x === undefined || Number.isNaN(x))
  ? "–"
  : Number(x).toLocaleString("en", { maximumSignificantDigits: digits });

/* ---------- Step 1: factors ---------- */

const EXAMPLE_FACTORS = [
  { name: "temperature", low: 20, high: 80, units: "C" },
  { name: "time", low: 2, high: 10, units: "min" },
];

function escapeAttr(s) {
  return String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

function addFactorRow(preset = {}) {
  const type = preset.type === "categorical" ? "categorical" : "continuous";
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input type="text" class="f-name" value="${escapeAttr(preset.name ?? "")}" placeholder="e.g. pH"></td>
    <td><select class="f-type">
      <option value="continuous"${type === "continuous" ? " selected" : ""}>Continuous</option>
      <option value="categorical"${type === "categorical" ? " selected" : ""}>Categorical</option>
    </select></td>
    <td>
      <span class="f-continuous"${type === "continuous" ? "" : " hidden"}>
        <input type="number" class="f-low" value="${preset.low ?? ""}" step="any" placeholder="low"> –
        <input type="number" class="f-high" value="${preset.high ?? ""}" step="any" placeholder="high">
      </span>
      <span class="f-categorical"${type === "categorical" ? "" : " hidden"}>
        <input type="text" class="f-levels" value="${escapeAttr((preset.levels ?? []).join(", "))}"
          placeholder="options, comma-separated (e.g. A, B, C)">
      </span>
    </td>
    <td><input type="text" class="f-units" value="${escapeAttr(preset.units ?? "")}"></td>
    <td><button type="button" class="remove-row" title="Remove factor">✕</button></td>`;
  tr.querySelector(".remove-row").addEventListener("click", () => { tr.remove(); onFactorsChanged(); });
  tr.querySelector(".f-type").addEventListener("change", (e) => {
    const cat = e.target.value === "categorical";
    tr.querySelector(".f-continuous").hidden = cat;
    tr.querySelector(".f-categorical").hidden = !cat;
    onFactorsChanged();
  });
  $("factor-rows").appendChild(tr);
}

function readFactors() {
  const factors = [];
  for (const tr of $("factor-rows").querySelectorAll("tr")) {
    const name = tr.querySelector(".f-name").value.trim();
    const type = tr.querySelector(".f-type").value;
    const units = tr.querySelector(".f-units").value.trim() || null;

    if (type === "categorical") {
      const raw = tr.querySelector(".f-levels").value.trim();
      if (!name && !raw) continue; // blank row
      if (!name) throw new Error("Every factor needs a name.");
      const levels = raw.split(",").map((s) => s.trim()).filter((s) => s.length);
      if (levels.length < 2) {
        throw new Error(`Factor "${name}" needs at least two options (comma-separated).`);
      }
      if (new Set(levels).size !== levels.length) {
        throw new Error(`Factor "${name}": its options must be unique.`);
      }
      factors.push({ type: "categorical", name, levels, units });
      continue;
    }

    const low = parseFloat(tr.querySelector(".f-low").value);
    const high = parseFloat(tr.querySelector(".f-high").value);
    if (!name && Number.isNaN(low) && Number.isNaN(high)) continue; // blank row
    if (!name) throw new Error("Every factor needs a name.");
    if (Number.isNaN(low) || Number.isNaN(high)) {
      throw new Error(`Factor "${name}" needs both a low and a high value.`);
    }
    if (low >= high) throw new Error(`Factor "${name}": low must be less than high.`);
    factors.push({ type: "continuous", name, low, high, units });
  }
  if (factors.length < 2) throw new Error("Add at least two factors.");
  const names = factors.map((f) => f.name);
  if (new Set(names).size !== names.length) throw new Error("Factor names must be unique.");
  return factors;
}

/* A factor's number of encoded model columns: 1 for a continuous factor, levels-1 for a
 * categorical (deviation coding). Drives the D-optimal run-count suggestion. */
function encodedColumns(f) {
  return f.type === "categorical" ? f.levels.length - 1 : 1;
}

/* Parameter count of the quadratic model the analysis fits — intercept + main effects +
 * continuous squares + all pairwise interactions. The D-optimal design needs at least this
 * many runs to be estimable; we suggest a few extra for residual degrees of freedom. */
function quadraticTermCount(factors) {
  let p = 1; // intercept
  for (const f of factors) {
    p += encodedColumns(f);
    if (f.type !== "categorical") p += 1; // squared term
  }
  for (let i = 0; i < factors.length; i++) {
    for (let j = i + 1; j < factors.length; j++) {
      p += encodedColumns(factors[i]) * encodedColumns(factors[j]);
    }
  }
  return p;
}

/* Quick screen ("ff2") run-count math, mirroring doe.generators.factorial exactly so the
 * live hint (and the reroute decision in generatePlan()) agree with what the service will
 * actually build. A 2-level full factorial explodes as 2^k; Plackett-Burman instead needs
 * only the smallest constructible multiple of 4 that is >= k + 1. */
const PB_BASES = [1, 12, 20];

function isPowerOfTwo(n) { return n >= 1 && (n & (n - 1)) === 0; }

function pbConstructible(n) {
  return PB_BASES.some((base) => n % base === 0 && isPowerOfTwo(n / base));
}

function pbRunCount(k) {
  let n = 4 * Math.ceil((k + 1) / 4);
  while (!pbConstructible(n)) n += 4;
  return n;
}

function fullFactorialRunCount(factors) {
  return factors.reduce((p, f) => p * (f.type === "categorical" ? f.levels.length : 2), 1);
}

/* Plackett-Burman (see doe.generators.factorial.plackett_burman) is a 2-level design: it
 * accepts continuous factors and categorical factors with exactly two levels, but rejects
 * a categorical factor with three or more options. Only reroute the quick screen when every
 * factor fits that shape — otherwise the full factorial (which already handles any
 * categorical factor) stays the generator, so a wider factor list never turns a working
 * plan into a 422. */
function pbCompatible(factors) {
  return factors.every((f) => f.type !== "categorical" || f.levels.length === 2);
}

/* More than 4 factors is where the 2-level full factorial's run count (2^k) starts to bite
 * (32 runs at 5 factors, 64 at 6); above that, a compact Plackett-Burman screen is used
 * instead — see generatePlan(). */
const PB_REROUTE_MIN_FACTORS = 5;

function screeningWillUsePlackettBurman(factors) {
  return factors.length >= PB_REROUTE_MIN_FACTORS && pbCompatible(factors);
}

/* Central composite run count (see doe.generators.rsm.central_composite): the UI never passes
 * a `fraction`, so the core is always the full 2^k factorial; add the 2k axial runs and the
 * service's default center = 4. Anchored against tests/test_app.py's central-composite cases. */
function centralCompositeRunCount(k) {
  return 2 ** k + 2 * k + 4;
}

/* Box-Behnken supports only 3 <= k <= 5 factors (see doe.generators.rsm.box_behnken); outside
 * that range the library itself raises rather than returning a design. */
const BOX_BEHNKEN_MAX_FACTORS = 5;

/* Box-Behnken run count: one edge run per +/-1 x +/-1 combination of each factor pair
 * (4 * C(k, 2)) plus the service's default center = 3. Only meaningful for
 * 3 <= k <= BOX_BEHNKEN_MAX_FACTORS. */
function boxBehnkenRunCount(k) {
  return 4 * ((k * (k - 1)) / 2) + 3;
}

/* Keep the plan controls in step with the current factor rows: the categorical note, the
 * quick-screen Plackett-Burman reroute hint, and the D-optimal run-count field / its
 * suggested default. Called whenever a row or the plan type changes; failures (a
 * half-typed row) are swallowed — the notes just stay as-is. */
function onFactorsChanged() {
  let factors = [];
  try { factors = readFactors(); } catch { /* incomplete rows — leave hints unchanged */ }
  const anyCategorical = factors.some((f) => f.type === "categorical");
  const type = $("design-type").value;

  $("design-cat-note").hidden = !(anyCategorical && (type === "ccd" || type === "bb"));

  const pbHint = $("ff2-pb-hint");
  if (type === "ff2" && factors.length >= PB_REROUTE_MIN_FACTORS) {
    pbHint.hidden = false;
    if (pbCompatible(factors)) {
      pbHint.textContent = `With ${factors.length} factors this uses a compact ` +
        `Plackett–Burman screen (${pbRunCount(factors.length)} runs) instead of a ` +
        `full factorial (${fullFactorialRunCount(factors)} runs).`;
    } else {
      pbHint.textContent = `With ${factors.length} factors this would normally switch to a ` +
        "compact Plackett–Burman screen, but a categorical factor here has more than two " +
        "options, so it stays a full factorial — expect a large number of runs.";
    }
  } else {
    pbHint.hidden = true;
  }

  const isDopt = type === "dopt";
  $("dopt-runs-wrap").hidden = !isDopt;
  if (isDopt && factors.length >= 2) {
    const terms = quadraticTermCount(factors);
    const suggested = terms + 4;
    const input = $("dopt-runs");
    if (!input.value || input.dataset.auto === "1") {
      input.value = suggested;
      input.dataset.auto = "1";
    }
    $("dopt-runs-hint").textContent = `≥ ${terms} runs needed to fit the model (${suggested} suggested).`;
  }

  // Live run-count preview: the bench cost of the plan as it stands right now, before the
  // user commits by clicking "Create experimental plan". Hidden whenever the rows don't
  // parse, there aren't yet 2 factors, or the plan/factor combination is invalid (the
  // categorical-with-ccd/bb case #design-cat-note already covers); the D-optimal run count
  // already has its own field + hint above, so this stays quiet for "dopt".
  const preview = $("run-count-preview");
  preview.hidden = true;
  const planInvalid = anyCategorical && (type === "ccd" || type === "bb");
  if (factors.length >= 2 && !planInvalid) {
    let text = null;
    if (type === "ccd") {
      text = `This plan will need ${centralCompositeRunCount(factors.length)} runs.`;
    } else if (type === "bb") {
      if (factors.length < 3) {
        text = "Box–Behnken needs at least 3 factors.";
      } else if (factors.length > BOX_BEHNKEN_MAX_FACTORS) {
        text = `Box–Behnken supports at most ${BOX_BEHNKEN_MAX_FACTORS} factors.`;
      } else {
        text = `This plan will need ${boxBehnkenRunCount(factors.length)} runs.`;
      }
    } else if (type === "ff2") {
      const n = screeningWillUsePlackettBurman(factors)
        ? pbRunCount(factors.length)
        : fullFactorialRunCount(factors);
      text = `This plan will need ${n} runs.`;
    }
    if (text !== null) {
      preview.textContent = text;
      preview.hidden = false;
    }
  }
}

/* The debug seed (behind the ⚙ toggle), or null when unset. Only two calls in the whole
 * flow are stochastic — the D-optimal search and the run-order randomization — so seeding
 * those makes a generated plan fully reproducible. */
function debugSeed() {
  const v = parseInt($("seed").value, 10);
  return Number.isInteger(v) && v >= 0 ? v : null;
}

async function generatePlan() {
  clearError("error-factors");
  try {
    const factors = readFactors();
    const type = $("design-type").value;
    const anyCategorical = factors.some((f) => f.type === "categorical");
    if (anyCategorical && (type === "ccd" || type === "bb")) {
      throw new Error(
        "Central composite and Box–Behnken designs need all-continuous factors. " +
        "For a categorical factor, pick the quick screen or the D-optimal custom design.");
    }
    // A quick screen with more than 4 factors reroutes to a Plackett-Burman design: the
    // 2-level full factorial's 2^k runs get impractical fast (32 at 5 factors, 64 at 6),
    // while Plackett-Burman needs only ~k+1 runs. Only when the factor list is a shape
    // Plackett-Burman accepts (see pbCompatible) — otherwise the full factorial stays, so
    // this never turns a working flow into a 422.
    const useScreeningPB = type === "ff2" && screeningWillUsePlackettBurman(factors);
    const endpoint = useScreeningPB ? "/designs/plackett-burman" : {
      ccd: "/designs/central-composite",
      bb: "/designs/box-behnken",
      ff2: "/designs/full-factorial",
      dopt: "/designs/optimal",
    }[type];
    const body = { factors };
    if (type === "ff2" && !useScreeningPB) body.levels = 2;
    if (type === "dopt") {
      const nRuns = parseInt($("dopt-runs").value, 10);
      const minRuns = quadraticTermCount(factors);
      if (Number.isNaN(nRuns) || nRuns < 2) throw new Error("Enter how many runs the design should have.");
      if (nRuns < minRuns) {
        throw new Error(`This model needs at least ${minRuns} runs to be estimable; you asked for ${nRuns}.`);
      }
      body.n_runs = nRuns;
      body.model = "quadratic";
    }
    const seed = debugSeed();
    if (seed !== null && type === "dopt") body.seed = seed;
    // Generation can take a moment (the D-optimal coordinate-exchange search especially),
    // so show a spinner once the request is actually in flight — after all validation above.
    // If it runs long, escalate the label so it clearly hasn't hung.
    setButtonBusy($("generate"), true, "Creating plan…", [
      { after: 4000, label: "Still working…" },
      { after: 12000, label: "Hang tight — optimising…" },
    ]);
    const { design } = await api(endpoint, body);
    // Remember which plan the user chose — it selects the screening vs. response-surface
    // results view (see isScreeningPlan / renderResults).
    state.planType = type;
    // Randomize the run order so the sheet is bench-ready.
    state.design = (await api("/designs/randomize",
      seed !== null ? { design, seed } : { design })).design;
    state.responseNames = ["response"];
    state.responseName = "response";
    renderPlan();
    $("step-plan").hidden = false;
    $("step-results").hidden = true;
    $("step-balance").hidden = true;
    $("step-plan").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    showError("error-factors", err);
  } finally {
    setButtonBusy($("generate"), false);
  }
}

/* ---------- Step 2: run sheet ---------- */

function factorNames() { return state.design.factors.map((f) => f.name); }
function continuousFactors() { return state.design.factors.filter((f) => f.type !== "categorical"); }
function continuousFactorNames() { return continuousFactors().map((f) => f.name); }
function hasCategorical() { return state.design.factors.some((f) => f.type === "categorical"); }

/* Screening generators (loaded plans carry the generator that built them): a plan that ranks
 * factor effects rather than locating an optimum. The UI only offers the 2-level full
 * factorial, but a fractional-factorial or Plackett–Burman screen loaded from JSON is
 * recognised too. A full factorial is screening only at 2 levels (3+ can fit curvature). */
const SCREENING_GENERATORS = new Set(["fractional_factorial", "plackett_burman"]);

function isScreeningPlan() {
  // A freshly generated plan knows exactly which type the user picked.
  if (state.planType) return state.planType === "ff2";
  // A loaded plan doesn't, so infer it from the recorded generator.
  const gen = state.design && state.design.meta && state.design.meta.generator;
  if (!gen) return false;
  if (gen.name === "full_factorial") {
    const levels = gen.parameters && gen.parameters.levels;
    return levels === 2 || (Array.isArray(levels) && levels.every((l) => l === 2));
  }
  return SCREENING_GENERATORS.has(gen.name);
}

/* Rebuild the "Measurements:" row of small text inputs (one per `state.responseNames`
 * entry) plus its remove buttons — the response-side counterpart of `addFactorRow`'s
 * per-row wiring. Renaming or removing a measurement re-renders the whole run sheet
 * (via `renderPlan`), so it always calls this too; called directly whenever only the
 * controls (not the table) need refreshing. */
function renderResponseControls() {
  const box = $("response-list");
  box.innerHTML = state.responseNames.map((name, i) => `
    <span class="resp-row">
      <input type="text" class="resp-name" data-idx="${i}" value="${escapeAttr(name)}" size="12">
      ${state.responseNames.length > 1
        ? '<button type="button" class="remove-row" title="Remove measurement">✕</button>'
        : ""}
    </span>`).join("");
  [...box.querySelectorAll(".resp-row")].forEach((row, i) => {
    const input = row.querySelector(".resp-name");
    input.addEventListener("change", () => renameResponse(i, input.value));
    const removeBtn = row.querySelector(".remove-row");
    if (removeBtn) removeBtn.addEventListener("click", () => removeResponse(i));
  });
  $("add-response").disabled = state.responseNames.length >= MAX_RESPONSES;
}

/* Read every measurement input's current value, keyed by response name then run index
 * (name-keyed, not index-keyed, so values survive a response being added/removed/renamed
 * and the table being rebuilt with a different column set). Blank cells are omitted. */
function snapshotResponseValues() {
  const snap = {};
  for (const name of state.responseNames) snap[name] = {};
  for (const input of document.querySelectorAll("input.resp")) {
    const name = state.responseNames[Number(input.dataset.resp)];
    if (name === undefined || input.value === "") continue;
    snap[name][Number(input.dataset.run)] = input.value;
  }
  return snap;
}

function renameResponse(i, rawValue) {
  const name = rawValue.trim();
  if (!name || state.responseNames.some((n, j) => j !== i && n === name)) {
    if (!name) showError("error-plan", new Error("A measurement needs a name."));
    else showError("error-plan", new Error(`"${name}" is already used for another measurement.`));
    renderResponseControls(); // restore the old name in the input
    return;
  }
  clearError("error-plan");
  const snap = snapshotResponseValues();
  const old = state.responseNames[i];
  if (name !== old) { snap[name] = snap[old]; delete snap[old]; }
  state.responseNames[i] = name;
  if (state.responseName === old) state.responseName = name;
  renderPlan(snap);
}

function removeResponse(i) {
  if (state.responseNames.length <= 1) return;
  const snap = snapshotResponseValues();
  const removed = state.responseNames[i];
  delete snap[removed];
  state.responseNames.splice(i, 1);
  if (state.responseName === removed) state.responseName = state.responseNames[0];
  renderPlan(snap);
}

function addResponse() {
  if (state.responseNames.length >= MAX_RESPONSES) return;
  const snap = snapshotResponseValues();
  let n = state.responseNames.length + 1;
  let name = `response${n}`;
  while (state.responseNames.includes(name)) { n += 1; name = `response${n}`; }
  state.responseNames.push(name);
  renderPlan(snap);
}

/* `responseSnapshot` (from `snapshotResponseValues`) re-fills measurement cells after a
 * rebuild triggered by adding/removing/renaming a response — otherwise every re-render
 * would start the sheet blank again. Omit it for a fresh plan (nothing to preserve). */
function renderPlan(responseSnapshot) {
  const d = state.design;
  const names = factorNames();
  $("plan-summary").textContent =
    `${d.runs.length} runs covering ${names.length} factors (plan: ${d.name}).`;

  const byName = new Map(d.factors.map((f) => [f.name, f]));
  // A categorical value is a level string — render it verbatim; only continuous values
  // are numbers to be formatted (and right-aligned).
  const cellFor = (n, v) => byName.get(n).type === "categorical"
    ? `<td>${escapeAttr(v)}</td>`
    : `<td class="num">${fmt(v, 5)}</td>`;

  // Some generators (e.g. the D-optimal custom design) don't tag runs with a point type;
  // drop the column entirely rather than show an empty one.
  const showPointType = Boolean(d.point_types);
  const head = ["Run", ...d.factors.map((f) => f.units ? `${f.name} (${f.units})` : f.name),
    ...(showPointType ? ["Point type"] : []), ...state.responseNames.map((n) => `Measurement: ${n}`)];
  const rows = d.runs.map((run, i) => {
    const cells = names.map((n) => cellFor(n, run[n])).join("");
    const pt = showPointType ? `<td class="point-type">${d.point_types[i]}</td>` : "";
    const respCells = state.responseNames.map((name, k) => {
      const v = responseSnapshot && responseSnapshot[name] && responseSnapshot[name][i] !== undefined
        ? responseSnapshot[name][i] : "";
      return `<td><input type="number" class="resp" step="any" data-run="${i}" data-resp="${k}" ` +
        `value="${escapeAttr(v)}"></td>`;
    }).join("");
    return `<tr><td class="num">${i + 1}</td>${cells}${pt}${respCells}</tr>`;
  });
  $("run-table").innerHTML =
    `<thead><tr>${head.map((h) => `<th${h === "Run" ? ' class="num"' : ""}>${h}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.join("")}</tbody>`;
  $("import-status").hidden = true; // a fresh table has no imported measurements
  renderResponseControls();
}

/* Quote a CSV field when it contains a comma, quote or newline (factor names, levels and
 * units are free-text, so this keeps the sheet well-formed). */
function csvField(value) {
  const s = value === null || value === undefined ? "" : String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function downloadCsv() {
  const d = state.design;
  const showPointType = Boolean(d.point_types);

  // Each factor gets its value column; a factor with units gets an adjacent "<name>_units"
  // column carrying that unit on every row, so downstream tools read the units explicitly.
  const header = ["run"];
  for (const f of d.factors) {
    header.push(f.name);
    if (f.units) header.push(`${f.name}_units`);
  }
  if (showPointType) header.push("point_type");
  for (const name of state.responseNames) header.push(name);

  const lines = d.runs.map((run, i) => {
    const cells = [i + 1];
    for (const f of d.factors) {
      cells.push(run[f.name]);
      if (f.units) cells.push(f.units);
    }
    if (showPointType) cells.push(d.point_types[i]);
    for (const _name of state.responseNames) cells.push(""); // empty response cells to fill in at the bench
    return cells.map(csvField).join(",");
  });

  const blob = new Blob([[header.map(csvField).join(","), ...lines].join("\n") + "\n"],
    { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${d.name || "design"}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ---------- Step 2: results import (CSV upload + spreadsheet paste) ---------- */

/* Pick the field separator by counting candidates in the header line (outside quotes):
 * Excel in comma-decimal locales writes semicolon-separated CSV, and a spreadsheet
 * "save as text" gives tabs. Comma wins ties, so a plain sheet stays a plain sheet. */
function sniffDelimiter(text) {
  const end = text.indexOf("\n");
  const line = end === -1 ? text : text.slice(0, end);
  let best = ",";
  let bestCount = 0;
  for (const d of [",", ";", "\t"]) {
    let count = 0;
    let inQuotes = false;
    for (const c of line) {
      if (c === '"') inQuotes = !inQuotes;
      else if (!inQuotes && c === d) count += 1;
    }
    if (count > bestCount) { best = d; bestCount = count; }
  }
  return best;
}

/* Minimal RFC-4180 reader: quoted fields, doubled quotes, CR/LF/CRLF row ends.
 * Returns rows of string cells, with all-blank rows dropped. */
function parseCsv(text, delimiter = ",") {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1); // Excel "CSV UTF-8" BOM
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  let i = 0;
  while (i < text.length) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
        inQuotes = false; i += 1; continue;
      }
      field += c; i += 1; continue;
    }
    if (c === '"') { inQuotes = true; i += 1; continue; }
    if (c === delimiter) { row.push(field); field = ""; i += 1; continue; }
    if (c === "\n" || c === "\r") {
      row.push(field); field = ""; rows.push(row); row = [];
      i += c === "\r" && text[i + 1] === "\n" ? 2 : 1;
      continue;
    }
    field += c; i += 1;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  return rows.filter((r) => r.some((cell) => cell.trim() !== ""));
}

/* Strict numeric parse — the whole cell must be a number, unlike parseFloat's
 * prefix parsing ("12abc" → 12, and worse, "0,5" → 0). With `commaDecimal`,
 * a single-comma-no-dot token is read as a comma-decimal ("0,5" → 0.5), for
 * semicolon CSVs and clipboard pastes from comma-decimal locales. */
function parseNumber(s, commaDecimal = false) {
  const t = s.trim();
  if (/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(t)) return parseFloat(t);
  if (commaDecimal && /^[+-]?\d+,\d+([eE][+-]?\d+)?$/.test(t)) {
    return parseFloat(t.replace(",", "."));
  }
  return NaN;
}

/* Match an uploaded results sheet against the current plan. Pure (no DOM), so it is
 * directly testable. Rows are matched by the sheet's `run` column when present — a
 * sheet re-sorted in a spreadsheet still lands on the right runs — falling back to row
 * order. Factor columns present in the sheet are cross-checked against the plan
 * (tolerantly for continuous values), so results from a different or re-generated plan
 * are refused instead of silently attached to the wrong runs. Every column that is
 * neither `run`, a factor, a `<factor>_units` carrier, `point_type`, nor bookkeeping is a
 * measurement candidate — `preferredNames` (the plan's existing measurement names, in
 * order) are tried first so a re-imported sheet lines back up with its columns, then any
 * others. Returns { columns: [{ name, values (number|null per run), imported, blank }] },
 * one entry per candidate column that has at least one number; throws on any structural
 * problem (naming the offending rows) or on a column whose non-blank cells aren't all
 * numbers — same strictness as before, just per-column. */
function matchResultsCsv(design, rows, preferredNames, commaDecimal) {
  if (rows.length < 2) throw new Error("That file has a header row but no data rows.");
  const header = rows[0].map((h) => h.trim());
  const data = rows.slice(1);
  const n = design.runs.length;
  const colOf = (name) => header.indexOf(name);

  const known = new Set(["", "run", "point_type"]);
  for (const c of RESERVED_RUN_COLUMNS) known.add(c.toLowerCase());
  for (const f of design.factors) {
    known.add(f.name.toLowerCase());
    known.add(`${f.name}_units`.toLowerCase());
  }
  const candidates = header.filter((h) => !known.has(h.toLowerCase()));
  if (!candidates.length) {
    throw new Error(
      "Couldn't find a measurement column — the sheet needs one extra column " +
      `(e.g. "${(preferredNames && preferredNames[0]) || "response"}") alongside the factor columns.`);
  }

  // Which design run each data row belongs to.
  const assignments = [];
  const runCol = header.findIndex((h) => h.toLowerCase() === "run");
  if (runCol !== -1) {
    const seen = new Set();
    data.forEach((r, rowNo) => {
      const cell = (r[runCol] ?? "").trim();
      const k = Number(cell);
      if (!Number.isInteger(k) || k < 1 || k > n) {
        throw new Error(`Row ${rowNo + 2}: run number "${cell}" isn't between 1 and ${n}.`);
      }
      if (seen.has(k)) throw new Error(`Run ${k} appears more than once in the file.`);
      seen.add(k);
      assignments.push([k - 1, r]);
    });
  } else {
    if (data.length !== n) {
      throw new Error(`The file has ${data.length} data rows but the plan has ${n} runs, ` +
        'and there is no "run" column to match them by.');
    }
    data.forEach((r, i) => assignments.push([i, r]));
  }

  const mismatched = [];
  for (const [idx, r] of assignments) {
    for (const f of design.factors) {
      const col = colOf(f.name);
      if (col === -1) continue; // factor column dropped from the sheet — nothing to check
      const cell = (r[col] ?? "").trim();
      if (cell === "") continue;
      const expected = design.runs[idx][f.name];
      if (f.type === "categorical") {
        if (cell !== String(expected)) { mismatched.push(idx + 1); break; }
      } else {
        const got = parseNumber(cell, commaDecimal);
        if (Number.isNaN(got) ||
            Math.abs(got - expected) > 1e-6 * Math.max(1, Math.abs(expected))) {
          mismatched.push(idx + 1); break;
        }
      }
    }
  }
  if (mismatched.length) {
    const shown = mismatched.slice(0, 8).join(", ") + (mismatched.length > 8 ? ", …" : "");
    throw new Error(
      `This file doesn't match the current plan — the factor settings differ on run${mismatched.length > 1 ? "s" : ""} ` +
      `${shown}. Import the sheet downloaded from this plan (or load the matching saved plan first).`);
  }

  // Preferred (existing measurement) names first, in their own order, then any other
  // candidate columns — so a re-imported sheet's known columns always line up first.
  const ordered = [
    ...(preferredNames || []).filter((h) => candidates.includes(h)),
    ...candidates.filter((h) => !(preferredNames || []).includes(h)),
  ];

  const columns = [];
  for (const name of ordered) {
    const respCol = colOf(name);
    const values = new Array(n).fill(null);
    const bad = [];
    for (const [idx, r] of assignments) {
      const cell = (r[respCol] ?? "").trim();
      if (cell === "") continue;
      const v = parseNumber(cell, commaDecimal);
      if (Number.isNaN(v)) { bad.push(`run ${idx + 1}: "${cell}"`); continue; }
      values[idx] = v;
    }
    if (bad.length) {
      throw new Error(`Some "${name}" values aren't numbers — ` +
        bad.slice(0, 5).join("; ") + (bad.length > 5 ? "; …" : "") + ".");
    }
    const imported = values.filter((v) => v !== null).length;
    if (imported) columns.push({ name, values, imported, blank: n - imported });
  }
  if (!columns.length) {
    throw new Error(`The "${ordered[0]}" column has no numbers yet — ` +
      "fill in your measurements and import the sheet again.");
  }
  return { columns };
}

function showImportStatus(message) {
  const el = $("import-status");
  el.textContent = `✓ ${message}`;
  el.hidden = false;
}

/* Import a filled-in run sheet: re-upload of the downloaded CSV (or a spreadsheet
 * re-save of it — semicolon/tab separated variants included). Fills the measurement
 * inputs to mirror the file, so blank sheet cells clear stale typed values. */
async function importResults(file) {
  clearError("error-plan");
  $("import-status").hidden = true;
  try {
    const text = await file.text();
    const delimiter = sniffDelimiter(text);
    const rows = parseCsv(text, delimiter);
    const { columns } = matchResultsCsv(state.design, rows, state.responseNames, delimiter !== ",");

    const snap = snapshotResponseValues();
    const parts = [];
    let skipped = 0;
    for (const { name, values, imported, blank } of columns) {
      if (!state.responseNames.includes(name)) {
        if (state.responseNames.length >= MAX_RESPONSES) { skipped += 1; continue; }
        state.responseNames.push(name);
      }
      snap[name] = {};
      values.forEach((v, i) => { if (v !== null) snap[name][i] = v; });
      parts.push(`${imported} measurement${imported === 1 ? "" : "s"} from “${name}”` +
        (blank ? ` (${blank} run${blank === 1 ? " is" : "s are"} still blank)` : ""));
    }
    renderPlan(snap);
    showImportStatus(`Imported ${parts.join("; ")}.` +
      (skipped ? ` ${skipped} extra column${skipped === 1 ? "" : "s"} skipped ` +
        `(${MAX_RESPONSES}-measurement limit).` : ""));
  } catch (err) {
    showError("error-plan", err);
  }
}

/* Paste a column of numbers copied from a spreadsheet into any measurement cell:
 * the values spill down the following rows instead of landing in one input. A
 * single-value paste keeps the browser's normal behaviour, and a copied header
 * cell riding above the numbers is skipped. */
function pasteResults(e) {
  const target = e.target;
  if (!target.classList || !target.classList.contains("resp")) return;
  const text = e.clipboardData ? e.clipboardData.getData("text") : "";
  const tokens = text.split(/[\n\r\t]+/).map((s) => s.trim()).filter((s) => s !== "");
  if (tokens.length < 2) return;
  e.preventDefault();
  if (Number.isNaN(parseNumber(tokens[0], true)) && !Number.isNaN(parseNumber(tokens[1], true))) {
    tokens.shift();
  }
  // Spill down within the target's own measurement column only — a multi-response sheet
  // has one `input.resp` per response per run, sharing `data-resp`.
  const inputs = [...document.querySelectorAll(`input.resp[data-resp="${target.dataset.resp}"]`)];
  let i = inputs.indexOf(target);
  let filled = 0;
  for (const t of tokens) {
    if (i >= inputs.length) break;
    const v = parseNumber(t, true);
    if (!Number.isNaN(v)) { inputs[i].value = v; filled += 1; }
    i += 1;
  }
  showImportStatus(`Pasted ${filled} value${filled === 1 ? "" : "s"} from the clipboard.`);
}

/* Save the current design as a JSON file — the same Design.to_dict() document the service
 * speaks, so it round-trips through Load (and any other doe tooling). Any measurements typed
 * but not yet analysed are folded in first, so a work-in-progress sheet is saved intact. */
function saveDesign() {
  if (!state.design) return;
  let design = state.design;
  const inputs = document.querySelectorAll("input.resp");
  if (inputs.length) {
    const runs = state.design.runs.map((r) => ({ ...r }));
    let any = false;
    inputs.forEach((input) => {
      const name = state.responseNames[Number(input.dataset.resp)];
      if (!name) return;
      const v = parseFloat(input.value);
      if (!Number.isNaN(v)) { runs[Number(input.dataset.run)][name] = v; any = true; }
    });
    if (any) design = { ...state.design, runs };
  }
  const blob = new Blob([JSON.stringify(design, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${design.name || "design"}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* Load a previously saved design JSON: validate it through the service, restore it as the
 * current design, rebuild the factor table, and re-open the run sheet with any saved
 * measurements pre-filled. */
async function loadDesign(file) {
  clearError("error-factors");
  try {
    const text = await file.text();
    let design;
    try {
      design = JSON.parse(text);
    } catch {
      throw new Error("That file isn't valid JSON.");
    }
    const { valid, errors } = await api("/designs/validate", { design });
    if (!valid) {
      throw new Error("This file isn't a valid design document:\n" + (errors || []).join("\n"));
    }
    state.design = design;
    // A loaded plan didn't come from the picker; isScreeningPlan() falls back to its generator.
    state.planType = null;

    // Rebuild step 1 so the factor table reflects the loaded design.
    $("factor-rows").innerHTML = "";
    for (const f of design.factors) addFactorRow(f);
    onFactorsChanged();

    // A measured response is a run column that is neither a factor nor bookkeeping
    // (`std_order` et al.); pre-fill every one so the results can be analysed.
    const factorSet = new Set(design.factors.map((f) => f.name));
    const responseCols = design.runs.length
      ? Object.keys(design.runs[0]).filter((k) => !factorSet.has(k) && !RESERVED_RUN_COLUMNS.has(k))
      : [];

    state.responseNames = responseCols.length ? responseCols.slice(0, MAX_RESPONSES) : ["response"];
    state.responseName = state.responseNames[0];

    const snap = {};
    for (const name of state.responseNames) {
      snap[name] = {};
      design.runs.forEach((run, i) => {
        const v = run[name];
        if (v !== null && v !== undefined) snap[name][i] = v;
      });
    }
    renderPlan(snap);
    $("step-plan").hidden = false;
    $("step-results").hidden = true;
    $("step-balance").hidden = true;
    $("step-plan").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    showError("error-factors", err);
  }
}

/* Deterministic synthetic response(s) so the whole flow can be demoed without a lab:
 * a smooth quadratic over the first two factors (in coded units) plus a small
 * run-dependent wiggle standing in for noise. The first measurement (`data-resp="0"`)
 * gets that surface; any additional measurement gets a shifted, negated variant, so a
 * demo with 2+ measurements has a genuine trade-off between them (see the balance view
 * below) rather than two copies of the same optimum. */
function demoFill() {
  const d = state.design;
  const coded = (f, v) => (2 * (v - f.low)) / (f.high - f.low) - 1;
  const [fa, fb] = continuousFactors();
  const cats = d.factors.filter((f) => f.type === "categorical");
  document.querySelectorAll("input.resp").forEach((input) => {
    const run = d.runs[Number(input.dataset.run)];
    const a = fa ? coded(fa, run[fa.name]) : 0;
    const b = fb ? coded(fb, run[fb.name]) : 0;
    // Each categorical factor contributes an offset tied to which level the run uses.
    let catBump = 0;
    for (const f of cats) {
      const idx = f.levels.indexOf(run[f.name]);
      catBump += 2 * (idx - (f.levels.length - 1) / 2);
    }
    const k = Number(input.dataset.resp);
    const wiggle = 0.3 * Math.sin(37 * (a + 2 * b) + Number(input.dataset.run) + k);
    const value = k === 0
      ? 70 + 4 * a + 3 * b - 3 * a * a - 2 * b * b + 1.5 * a * b + catBump + wiggle
      : 30 - 3 * a - 2 * b + 2 * a * a + 1.5 * b * b - a * b - catBump + wiggle;
    input.value = value.toFixed(2);
  });
}

/* ---------- Step 3: analysis ---------- */

/* Read one measurement column's values, erroring with the run and measurement name of
 * the first blank cell found. `idx` is the column's `data-resp` index. */
function readResponseValues(idx, name) {
  const values = [];
  for (const input of document.querySelectorAll(`input.resp[data-resp="${idx}"]`)) {
    const v = parseFloat(input.value);
    if (Number.isNaN(v)) {
      throw new Error(`Run ${Number(input.dataset.run) + 1} ("${name}") has no measurement yet — ` +
        "fill in every row (or use the demo button).");
    }
    values[Number(input.dataset.run)] = v;
  }
  return values;
}

/* Validate the measurement names before attaching them: unique, non-empty, and not
 * clashing with a factor name or a reserved bookkeeping column — mirrors readFactors()'s
 * validation style. */
function validateResponseNames() {
  const names = state.responseNames;
  if (!names.length) throw new Error("Add at least one measurement.");
  if (new Set(names).size !== names.length) throw new Error("Measurement names must be unique.");
  const factorSet = new Set(factorNames());
  for (const n of names) {
    if (!n) throw new Error("Every measurement needs a name.");
    if (factorSet.has(n)) throw new Error(`Measurement "${n}" clashes with a factor name — rename it.`);
    if (RESERVED_RUN_COLUMNS.has(n)) throw new Error(`"${n}" is a reserved column name — rename it.`);
  }
}

/* A copy of the design document with a given run column removed (no-op if absent). Used to
 * drop a stale response column before re-attaching, so re-analysing — or analysing a plan
 * loaded with results already on it — doesn't clash with the existing column. */
function designWithoutColumn(design, column) {
  if (!design.runs.some((run) => column in run)) return design;
  const runs = design.runs.map((run) => {
    const copy = { ...run };
    delete copy[column];
    return copy;
  });
  return { ...design, runs };
}

async function analyze() {
  clearError("error-plan");
  clearError("error-results");
  try {
    validateResponseNames();
    // Drop any existing column of each name first: `with_response` refuses to overwrite,
    // so a loaded-with-results plan (or a re-analysis with edited numbers) would otherwise
    // clash. Attach every measurement in one call.
    let base = state.design;
    const responses = {};
    state.responseNames.forEach((name, idx) => {
      responses[name] = readResponseValues(idx, name);
      base = designWithoutColumn(base, name);
    });
    const attached = await api("/designs/responses", { design: base, responses });
    state.design = attached.design;
    if (!state.responseNames.includes(state.responseName)) {
      state.responseName = state.responseNames[0];
    }
    // Reveal the results card *before* drawing into it: Plotly sizes each figure to its
    // container, and a display:none container measures 0×0 — the plots would come up at
    // Plotly's default size and only correct themselves on the next window resize.
    const results = $("step-results");
    const wasHidden = results.hidden;
    results.hidden = false;
    try {
      renderResultsResponseSelector();
      await renderResults();
    } catch (err) {
      results.hidden = wasHidden;
      throw err;
    }
    showBalanceCardIfEligible();
    results.scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    showError("error-plan", err);
  }
}

/* Show/hide the `#results-response` selector (results-controls toolbar) based on how
 * many measurements are attached — a single measurement needs no selector at all. */
function renderResultsResponseSelector() {
  const wrap = $("results-response-wrap");
  const many = state.responseNames.length > 1;
  wrap.hidden = !many;
  if (!many) return;
  const sel = $("results-response");
  sel.innerHTML = state.responseNames.map((n) => `<option value="${escapeAttr(n)}">${n}</option>`).join("");
  sel.value = state.responseName;
}

/* A Plackett-Burman design is saturated for main effects (n ~= k + 1 runs): the default
 * "quadratic" model (main effects + every pairwise interaction) has far more terms than
 * runs (e.g. 6 factors -> 12 runs but 22 terms) and is rank-deficient. Its recorded
 * generator (present whether the plan was just generated or loaded from JSON — see
 * isScreeningPlan) picks the main-effects-only spec instead; every other plan keeps the
 * quadratic model unchanged. */
function fitRequest() {
  const generator = state.design && state.design.meta && state.design.meta.generator;
  const model = generator && generator.name === "plackett_burman"
    ? { order: 1, interactions: false }
    : "quadratic";
  return { design: state.design, response: state.responseName, model };
}

/* Best predicted settings for a design with a categorical factor. The service optimizes the
 * continuous factors exactly within each combination of categorical levels (which the plain
 * coded-box optimizer can't search) and returns the winner. Its `settings` field carries all
 * factors (continuous values + the winning level labels); we expose it as `natural` so the
 * rest of the UI consumes it exactly like the `/optimize/optimum` response. */
async function bestSettingsCategorical(maximize) {
  const opt = await api("/optimize/categorical-optimum", { ...fitRequest(), maximize });
  return { natural: opt.settings, response: opt.response, at_bound: opt.at_bound, warnings: opt.warnings };
}

/* Render one factor's optimum setting: a categorical level verbatim, a continuous value formatted. */
function formatOptimumSettings(natural) {
  return state.design.factors
    .map((f) => `${f.name} = ${f.type === "categorical" ? natural[f.name] : fmt(natural[f.name], 4)}`)
    .join(", ");
}

async function renderResults() {
  const fit = await api("/analysis/fit", fitRequest());

  // The R² tile and effects table are shown in both views, so fill them once up front.
  $("stat-r2").textContent = fmt(fit.r_squared, 3);
  $("stat-adjr2").textContent = fit.adjusted_r2 === null ? "" : `adjusted: ${fmt(fit.adjusted_r2, 3)}`;

  if (isScreeningPlan()) {
    await renderScreeningView(fit);
  } else {
    await renderOptimizeView(fit);
  }
}

/* Toggle the results card between the response-surface view (best settings + contour +
 * adequacy) and the screening view (effect Pareto + half-normal). The R² tile and the effects
 * table live outside both blocks, so they stay visible either way. */
function setResultsView(mode) {
  const screening = mode === "screening";
  $("optimum-tile").hidden = screening;
  $("results-controls").hidden = screening;
  $("response-map-block").hidden = screening;
  $("adequacy").hidden = screening;
  $("screening-view").hidden = !screening;
}

/* Response-surface view: the best predicted settings, a fitted contour map, model-adequacy
 * diagnostics, and the effects table. Used for every plan except the 2-level screen. */
async function renderOptimizeView(fit) {
  setResultsView("optimize");
  const categorical = hasCategorical();
  const maximize = $("maximize").checked;

  // The optimum for an all-continuous fit comes from the service's surface optimizer; a
  // categorical fit has no coded box to search, so we grid-search it here through /predict.
  const opt = categorical
    ? await bestSettingsCategorical(maximize)
    : await api("/optimize/optimum", { ...fitRequest(), maximize });

  $("optimum-label").textContent = maximize ? "Best predicted settings (max)" : "Best predicted settings (min)";
  $("stat-optimum").textContent = formatOptimumSettings(opt.natural);
  $("stat-optimum-y").textContent =
    `predicted ${state.responseName}: ${fmt(opt.response, 4)}` +
    (opt.at_bound ? " (at the edge of the tested range)" : "");

  const warnings = [...(fit.warnings || []), ...(opt.warnings || [])];
  $("fit-warnings").hidden = warnings.length === 0;
  $("fit-warnings").textContent = warnings.join("; ");

  renderTermsTable(fit.terms);
  await renderAdequacy(fit);
  setupAxisPickers(opt);
  await renderContour(opt.natural);
  await renderInteractionBlock(fit, opt.natural);
}

/* Screening view: a 2-level "which factors matter?" plan has no curvature to map and no
 * meaningful interior optimum, so it swaps the best-settings tile + response map for the two
 * classic screening reads (effect Pareto + half-normal) over the same fitted effects, and
 * ranks the effects table by magnitude so the vital few sit at the top. */
async function renderScreeningView(fit) {
  setResultsView("screening");

  const warnings = fit.warnings || [];
  $("fit-warnings").hidden = warnings.length === 0;
  $("fit-warnings").textContent = warnings.join("; ");

  const effects = screeningEffects(fit.terms);
  renderParetoPlot(effects);
  renderHalfNormalPlot(effects);
  renderTermsTable(fit.terms, { sortByEffect: true });
  await renderInteractionBlock(fit, screeningHeldNatural());
}

/* Non-intercept model terms as effects for the screening plots: the signed ±1 coded swing
 * (falling back to twice the coefficient when the service leaves `effect` null, e.g. a
 * saturated fit), its magnitude, and whether the service flagged it significant (p < 0.05). */
function screeningEffects(terms) {
  return terms
    .filter((t) => t.term !== "Intercept")
    .map((t) => {
      const effect = t.effect ?? 2 * t.coefficient;
      return {
        term: t.term,
        effect,
        abs: Math.abs(effect),
        significant: t.p !== null && t.p !== undefined && t.p < 0.05,
      };
    });
}

/* Pareto chart of |effect|: horizontal bars, largest at the top, blue where the service found
 * significance (p < 0.05) and grey otherwise. Degrades to a caption when Plotly is offline. */
function renderParetoPlot(effects) {
  const el = $("pareto");
  if (typeof Plotly === "undefined") { el.innerHTML = OFFLINE_PLOT_MSG; return; }
  // Ascending, so Plotly's bottom-up horizontal bars put the largest effect at the top.
  const sorted = [...effects].sort((a, b) => a.abs - b.abs);
  drawPlot(el, [{
    type: "bar",
    orientation: "h",
    x: sorted.map((e) => e.abs),
    y: sorted.map((e) => e.term),
    marker: { color: sorted.map((e) => (e.significant ? "#256abf" : "#b7c0d0")) },
    hovertemplate: "%{y}<br>|effect| %{x:.4~g}<extra></extra>",
  }], {
    margin: { t: 10, r: 10, b: 45, l: 60 },
    xaxis: { title: { text: `|effect| on ${state.responseName}` } },
    yaxis: { automargin: true },
    font: { family: "system-ui, sans-serif", color: "#1f2430" },
    paper_bgcolor: "rgba(0,0,0,0)",
    showlegend: false,
  });
}

/* Half-normal plot: |effects| (ascending) against half-normal quantiles. Inactive (noise)
 * effects fall on a line through the origin; active factors pull up and to the right, so the
 * ones the service flagged significant are labelled. Reuses invNormalCDF from the adequacy panel. */
function renderHalfNormalPlot(effects) {
  const el = $("halfnormal");
  if (typeof Plotly === "undefined") { el.innerHTML = OFFLINE_PLOT_MSG; return; }
  const sorted = [...effects].sort((a, b) => a.abs - b.abs);
  const m = sorted.length;
  const quant = sorted.map((_, i) => invNormalCDF(0.5 + 0.5 * ((i + 0.5) / m)));
  drawPlot(el, [{
    type: "scatter",
    mode: "markers+text",
    x: quant,
    y: sorted.map((e) => e.abs),
    text: sorted.map((e) => (e.significant ? e.term : "")),
    textposition: "top left",
    textfont: { size: 11 },
    customdata: sorted.map((e) => e.term),
    marker: {
      size: 8,
      color: sorted.map((e) => (e.significant ? "#256abf" : "#b7c0d0")),
      line: { color: "#ffffff", width: 1 },
    },
    hovertemplate: "%{customdata}<br>|effect| %{y:.4~g}<extra></extra>",
  }], {
    margin: { t: 10, r: 10, b: 45, l: 55 },
    xaxis: { title: { text: "half-normal quantile" } },
    yaxis: { title: { text: "|effect|" } },
    font: { family: "system-ui, sans-serif", color: "#1f2430" },
    paper_bgcolor: "rgba(0,0,0,0)",
    showlegend: false,
  });
}

function renderTermsTable(terms, { sortByEffect = false } = {}) {
  let terms_ = terms.filter((t) => t.term !== "Intercept");
  if (sortByEffect) {
    const mag = (t) => Math.abs(t.effect ?? 2 * t.coefficient);
    terms_ = [...terms_].sort((a, b) => mag(b) - mag(a));
  }
  const rows = terms_.map((t) => {
    const sig = t.p !== null && t.p < 0.05;
    return `<tr${sig ? ' class="significant"' : ""}>` +
      `<td>${t.term}</td><td class="num">${fmt(t.coefficient, 4)}</td>` +
      `<td class="num">${t.p === null ? "–" : t.p < 0.001 ? "&lt; 0.001" : fmt(t.p, 2)}</td>` +
      `<td>${t.p === null ? "not estimable" : sig ? "strong evidence" : "weak / none"}</td></tr>`;
  });
  $("terms-table").innerHTML =
    "<thead><tr><th>Effect</th><th class=\"num\">Coefficient (coded)</th>" +
    "<th class=\"num\">p-value</th><th>Evidence it matters</th></tr></thead>" +
    `<tbody>${rows.join("")}</tbody>`;
}

/* Model-adequacy panel (below the best-settings tile): is the fit trustworthy enough to act
 * on? Three complementary, best-effort checks — a saturated or unreplicated design leaves some
 * of them undefined, which we surface as a plain caveat rather than failing the analysis:
 *   - predicted R² and the lack-of-fit F-test, from `/analysis/anova`;
 *   - the two standard residual diagnostics (residuals-vs-fitted, normal Q–Q), drawn from the
 *     fit's own `fitted`/`residuals` (already in hand — no extra call for the plots). */
async function renderAdequacy(fit) {
  clearError("error-adequacy");
  renderResidualPlots(fit);

  // predicted-R² and the lack-of-fit test come from the ANOVA endpoint (same fit request).
  let anova = null;
  try {
    anova = await api("/analysis/anova", fitRequest());
  } catch (err) {
    // Supplementary panel — a failed ANOVA must not sink the whole results view.
    showError("error-adequacy", err);
  }
  const predR2 = anova ? anova.predicted_r2 : null;
  const lof = anova ? anova.lack_of_fit : null;
  const lofP = lof ? lof.p : null;

  $("stat-predr2").textContent =
    predR2 === null || predR2 === undefined ? "n/a" : fmt(predR2, 3);

  // Lack-of-fit: a *small* p means the model misses real structure (bad); a large p reassures.
  if (lofP === null || lofP === undefined) {
    $("stat-lof").textContent = "n/a";
    $("stat-lof-sub").textContent = "needs replicated runs";
  } else if (lofP < 0.05) {
    $("stat-lof").textContent = "possible misfit";
    $("stat-lof-sub").textContent = `p = ${fmt(lofP, 2)} (< 0.05)`;
  } else {
    $("stat-lof").textContent = "no concern";
    $("stat-lof-sub").textContent = `p = ${fmt(lofP, 2)}`;
  }

  renderVerdict(fit, predR2, lofP);
}

/* One plain-language sentence weighing the adequacy signals, colour-coded ok / warn / plain.
 * "warn" wins if any signal is bad; "plain" when nothing could be computed (no residual dof). */
function renderVerdict(fit, predR2, lofP) {
  const el = $("adequacy-verdict");
  const parts = [];
  let tone = "plain";

  const r2 = fit.r_squared;
  if (r2 !== null && r2 !== undefined) {
    parts.push(`The model explains ${Math.round(100 * r2)}% of the variation in your results (R²).`);
  }

  if (lofP !== null && lofP !== undefined) {
    if (lofP < 0.05) {
      parts.push("A lack-of-fit test flags structure the model is missing — treat the best " +
        "settings as a rough guide, and consider adding runs (e.g. a fuller design).");
      tone = "warn";
    } else {
      parts.push("A lack-of-fit test finds no evidence the model is missing structure.");
      if (tone !== "warn") tone = "ok";
    }
  }

  if (predR2 !== null && predR2 !== undefined) {
    if (predR2 < 0) {
      parts.push("Its predicted R² is negative — it forecasts new runs worse than the overall " +
        "average would, so don't rely on the predictions.");
      tone = "warn";
    } else {
      parts.push(`It should predict fresh runs with an R² of about ${fmt(predR2, 2)}.`);
      if (tone !== "warn") tone = "ok";
    }
  }

  if (parts.length === 0) {
    parts.push("This design has no spare runs left over to check the model against — add " +
      "replicated runs (repeat a setting) to test whether the model actually fits.");
  }

  el.className = `verdict verdict-${tone}`;
  el.textContent = parts.join(" ");
}

/* The two residual diagnostics, from the fit's own fitted values + residuals, drawn with Plotly
 * to match the contour's styling. Degrades to a caption when Plotly is unavailable offline. */
function renderResidualPlots(fit) {
  const rf = $("resid-fitted");
  const qq = $("resid-qq");
  if (typeof Plotly === "undefined") {
    rf.innerHTML = OFFLINE_PLOT_MSG;
    qq.innerHTML = OFFLINE_PLOT_MSG;
    return;
  }

  const fitted = fit.fitted || [];
  const residuals = fit.residuals || [];
  const layoutBase = {
    margin: { t: 10, r: 10, b: 45, l: 55 },
    showlegend: false,
    font: { family: "system-ui, sans-serif", color: "#1f2430" },
    paper_bgcolor: "rgba(0,0,0,0)",
  };
  const marker = { color: "#256abf", size: 7, line: { color: "#ffffff", width: 1 } };
  const refLine = { color: "#8a93a6", width: 1, dash: "dash" };

  // Residuals vs fitted: want a flat, patternless band about the dashed zero line.
  const xr = fitted.length ? [Math.min(...fitted), Math.max(...fitted)] : [0, 1];
  drawPlot(rf, [
    { type: "scatter", mode: "lines", x: xr, y: [0, 0], line: refLine, hoverinfo: "skip" },
    { type: "scatter", mode: "markers", x: fitted, y: residuals, marker,
      hovertemplate: "fitted %{x:.4~g}<br>residual %{y:.4~g}<extra></extra>" },
  ], { ...layoutBase, xaxis: { title: { text: "fitted value" } },
    yaxis: { title: { text: "residual" }, zeroline: false } });

  // Normal Q–Q: ordered residuals vs theoretical standard-normal quantiles. The dashed
  // reference is the line expected if the residuals were exactly normal (intercept = mean,
  // slope = sd) — points hugging it means the normality assumption holds.
  const sorted = [...residuals].sort((a, b) => a - b);
  const n = sorted.length;
  const theo = sorted.map((_, i) => invNormalCDF((i + 0.5) / n));
  const mean = n ? sorted.reduce((s, v) => s + v, 0) / n : 0;
  const sd = n ? Math.sqrt(sorted.reduce((s, v) => s + (v - mean) ** 2, 0) / n) : 0;
  const tr = n ? [theo[0], theo[n - 1]] : [-1, 1];
  drawPlot(qq, [
    { type: "scatter", mode: "lines", x: tr, y: tr.map((z) => mean + sd * z),
      line: refLine, hoverinfo: "skip" },
    { type: "scatter", mode: "markers", x: theo, y: sorted, marker,
      hovertemplate: "theoretical %{x:.3~g}<br>residual %{y:.4~g}<extra></extra>" },
  ], { ...layoutBase, xaxis: { title: { text: "theoretical quantile" } },
    yaxis: { title: { text: "ordered residual" } } });
}

/* Acklam's rational approximation to the inverse standard-normal CDF (Φ⁻¹), accurate to
 * ~1e-9 — ample for Q–Q plotting positions, and avoids pulling in a stats library client-side. */
function invNormalCDF(p) {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
    1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
    6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
    -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
    3.754408661907416e+00];
  const plow = 0.02425;
  const phigh = 1 - plow;
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p <= phigh) {
    const q = p - 0.5;
    const r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  const q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

function setupAxisPickers(opt) {
  // The response map's axes are always two *continuous* factors (a categorical factor has
  // no coded sweep); pickers only appear when there are more than two to choose between.
  const names = continuousFactorNames();
  const many = names.length > 2;
  $("axis-pickers").hidden = !many;
  if (many) {
    for (const [id, dflt] of [["axis-x", names[0]], ["axis-y", names[1]]]) {
      const sel = $(id);
      const current = sel.value;
      sel.innerHTML = names.map((n) => `<option value="${escapeAttr(n)}">${n}</option>`).join("");
      sel.value = names.includes(current) ? current : dflt;
    }
    if ($("axis-x").value === $("axis-y").value) $("axis-y").value = names.find((n) => n !== $("axis-x").value);
  }

  // One level picker per categorical factor — the map holds that factor at the chosen level.
  // Preserve a level the user already picked; otherwise default to the optimum's level so the
  // map first shows the best-settings slice (with the star on it).
  const cats = state.design.factors.filter((f) => f.type === "categorical");
  const box = $("cat-levels");
  box.hidden = cats.length === 0 || names.length < 2;
  if (!box.hidden) {
    const prev = {};
    for (const s of box.querySelectorAll(".cat-level")) prev[s.dataset.factor] = s.value;
    box.innerHTML = cats.map((f) => {
      const chosen = prev[f.name] ?? (opt ? opt.natural[f.name] : f.levels[0]);
      const opts = f.levels
        .map((l) => `<option value="${escapeAttr(l)}"${l === chosen ? " selected" : ""}>${l}</option>`)
        .join("");
      return `<label class="inline-label">${f.name} = ` +
        `<select class="cat-level" data-factor="${escapeAttr(f.name)}">${opts}</select></label>`;
    }).join("");
    for (const sel of box.querySelectorAll("select")) sel.addEventListener("change", rerender);
  }

  $("held-hint").textContent = hasCategorical()
    ? "(other factors held at the values shown; the ★ marks the best settings)"
    : "(other factors held at their optimum)";
}

async function renderContour(optimumNatural) {
  const el = $("contour");
  if (typeof Plotly === "undefined") {
    el.innerHTML = "<p class=\"plot-fallback\">Plot unavailable offline (Plotly.js loads from a CDN) — " +
      "the numbers above are unaffected.</p>";
    return;
  }
  const names = continuousFactorNames();
  if (names.length < 2) {
    el.innerHTML = "<p class=\"plot-fallback\">A response map needs at least two continuous " +
      "factors to draw. The table below still shows which factors matter.</p>";
    return;
  }
  const x = names.length > 2 ? $("axis-x").value : names[0];
  let y = names.length > 2 ? $("axis-y").value : names[1];
  if (x === y) y = names.find((n) => n !== x);

  // Hold every non-axis factor: continuous at its optimum and each categorical at its chosen
  // level. The star (best settings) belongs on this slice only when the held categorical
  // levels match the optimum's — otherwise the slice shown isn't the one the optimum lives on.
  const catLevel = {};
  for (const s of document.querySelectorAll(".cat-level")) catLevel[s.dataset.factor] = s.value;
  const fixed = {};
  let onOptimumSlice = true;
  for (const f of state.design.factors) {
    if (f.name === x || f.name === y) continue;
    if (f.type === "categorical") {
      const held = catLevel[f.name] ?? f.levels[0];
      fixed[f.name] = held;
      if (held !== optimumNatural[f.name]) onOptimumSlice = false;
    } else {
      fixed[f.name] = optimumNatural[f.name];
    }
  }

  const surface = await api("/plot-data/surface", {
    ...fitRequest(), x, y, resolution: 40,
    ...(Object.keys(fixed).length ? { fixed } : {}),
  });

  const label = (n) => {
    const f = state.design.factors.find((ff) => ff.name === n);
    return f.units ? `${n} (${f.units})` : n;
  };

  const traces = [
    {
      type: "contour",
      x: surface.x[0],
      y: surface.y.map((row) => row[0]),
      z: surface.z,
      colorscale: CONTOUR_COLORSCALE,
      line: { width: 0.5, color: "rgba(255,255,255,0.6)" },
      colorbar: { title: { text: state.responseName, side: "right" }, thickness: 12, outlinewidth: 0 },
      hovertemplate: `${x}: %{x:.4~g}<br>${y}: %{y:.4~g}<br>${state.responseName}: %{z:.4~g}<extra></extra>`,
    },
    {
      type: "scatter", mode: "markers", name: "your runs",
      x: state.design.runs.map((r) => r[x]),
      y: state.design.runs.map((r) => r[y]),
      marker: { color: "#ffffff", size: 8, symbol: "circle", line: { color: "#1f2430", width: 1.5 } },
      hovertemplate: `run at ${x} = %{x:.4~g}, ${y} = %{y:.4~g}<extra></extra>`,
    },
  ];
  // Show the best-settings star only on the slice the optimum actually lives on (always true
  // for an all-continuous fit; for a categorical one, only when the held levels match).
  if (onOptimumSlice) {
    traces.push({
      type: "scatter", mode: "markers", name: "predicted best",
      x: [optimumNatural[x]], y: [optimumNatural[y]],
      marker: { color: "#ffffff", size: 14, symbol: "star", line: { color: "#1f2430", width: 1.5 } },
      hovertemplate: "predicted best<extra></extra>",
    });
  }

  drawPlot(el, traces, {
    margin: { t: 10, r: 10, b: 55, l: 65 },
    xaxis: { title: { text: label(x) } },
    yaxis: { title: { text: label(y) } },
    showlegend: false,
    font: { family: "system-ui, sans-serif", color: "#1f2430" },
    paper_bgcolor: "rgba(0,0,0,0)",
  });
}

/* ---------- Step 3: interaction plot ---------- */

/* Stash for the picker's redraw-only-this-plot behaviour: the qualifying pairs (so the
 * picker's <select> can be rebuilt without re-fitting) and the natural-unit values every
 * non-axis factor is held at. Set once by renderInteractionBlock; read again by
 * renderInteractionPlot when the picker changes without re-running renderResults(). */
let interactionContext = null;

/* Significant (p < 0.05) continuous x continuous interaction terms, sorted by |effect|
 * descending. A term only qualifies when both ":"-split halves are exact continuous factor
 * names — a term involving a categorical factor's deviation-coded contrast column (e.g.
 * "catalyst[B]:temperature") never matches a bare factor name, so it's excluded (the
 * library's interaction_lines rejects a categorical axis, and the router turns that into a
 * 422). */
function continuousInteractionPairs(terms) {
  const continuous = new Set(continuousFactorNames());
  return terms
    .filter((t) => t.p !== null && t.p !== undefined && t.p < 0.05 && t.term.includes(":"))
    .map((t) => ({ term: t.term, parts: t.term.split(":"), effect: t.effect ?? 2 * t.coefficient }))
    .filter((t) => t.parts.length === 2 && t.parts.every((p) => continuous.has(p)))
    .sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect));
}

/* Show (and draw) the interaction-plot block when at least one significant continuous x
 * continuous interaction exists in `fit.terms`; hide the whole block otherwise, so the page
 * stays uncluttered when there's nothing to show. `heldNatural` supplies the natural-unit
 * value every non-axis factor is held at while sweeping the plot: the optimum for a
 * response-surface fit (renderOptimizeView), or each factor's midpoint / first level for a
 * screening fit (renderScreeningView), which has no optimum to hold at. */
async function renderInteractionBlock(fit, heldNatural) {
  const pairs = continuousInteractionPairs(fit.terms);
  const block = $("interaction-block");
  if (!pairs.length) { block.hidden = true; interactionContext = null; return; }
  block.hidden = false;

  const many = pairs.length > 1;
  $("interaction-pick-wrap").hidden = !many;
  if (many) {
    const sel = $("interaction-pick");
    sel.innerHTML = pairs.map((p) => `<option value="${escapeAttr(p.term)}">${p.term}</option>`).join("");
    sel.value = pairs[0].term;
  }

  $("interaction-held-hint").textContent = state.design.factors.length > 2
    ? `(other factors held at ${isScreeningPlan() ? "their midpoint / first option" : "the best predicted settings"})`
    : "";

  interactionContext = { pairs, heldNatural };
  await renderInteractionPlot(pairs[0]);
}

/* Fetch and draw the interaction plot for one qualifying term — its two ":"-split parts are
 * the swept factor (`x`) and the factor whose low/high settings each become a line
 * (`trace`). Only this plot is refetched/redrawn when the picker changes, not the whole
 * results view (renderResults() is not re-run). A failed call renders its message inline
 * rather than sinking the rest of the results view (mirrors renderAdequacy's soft-fail). */
async function renderInteractionPlot(pair) {
  const el = $("interaction-plot");
  if (typeof Plotly === "undefined") { el.innerHTML = OFFLINE_PLOT_MSG; return; }

  const [x, trace] = pair.parts;
  const heldNatural = interactionContext.heldNatural;
  const fixed = {};
  for (const f of state.design.factors) {
    if (f.name === x || f.name === trace) continue;
    fixed[f.name] = heldNatural[f.name];
  }

  let data;
  try {
    data = await api("/plot-data/interactions", { ...fitRequest(), x, trace, fixed, resolution: 25 });
  } catch (err) {
    el.innerHTML = `<p class="plot-fallback">${escapeAttr(err instanceof Error ? err.message : String(err))}</p>`;
    return;
  }

  const label = (n) => {
    const f = state.design.factors.find((ff) => ff.name === n);
    return f.units ? `${n} (${f.units})` : n;
  };
  const traceFactor = state.design.factors.find((f) => f.name === trace);
  const traceUnits = traceFactor.units ? ` ${traceFactor.units}` : "";
  // Colorblind-safe blue/orange pair, matching the palette used elsewhere on the page.
  const colors = ["#256abf", "#e07b39"];

  const traces = data.lines.map((line, i) => ({
    type: "scatter",
    mode: "lines+markers",
    name: `${trace} = ${fmt(line.trace_value, 4)}${traceUnits}`,
    x: data.x,
    y: line.z,
    line: { color: colors[i % colors.length], width: 2 },
    marker: { color: colors[i % colors.length], size: 6, line: { color: "#ffffff", width: 1 } },
    hovertemplate: `${x}: %{x:.4~g}<br>predicted ${state.responseName}: %{y:.4~g}` +
      `<extra>${trace} = ${fmt(line.trace_value, 4)}${traceUnits}</extra>`,
  }));

  drawPlot(el, traces, {
    margin: { t: 10, r: 10, b: 55, l: 65 },
    xaxis: { title: { text: label(x) } },
    yaxis: { title: { text: `predicted ${state.responseName}` } },
    legend: { orientation: "h", y: -0.25 },
    font: { family: "system-ui, sans-serif", color: "#1f2430" },
    paper_bgcolor: "rgba(0,0,0,0)",
  });
}

/* Held values for the interaction plot's non-axis factors when there is no fitted optimum to
 * hold at (the screening view): each continuous factor at its midpoint, each categorical
 * factor at its first level. A Plackett-Burman screen fits main effects only (no ":" terms),
 * so continuousInteractionPairs() naturally finds nothing and the block stays hidden — this
 * only matters for a (rare) screening plan whose model does carry an interaction term. */
function screeningHeldNatural() {
  const held = {};
  for (const f of state.design.factors) {
    held[f.name] = f.type === "categorical" ? f.levels[0] : (f.low + f.high) / 2;
  }
  return held;
}

/* ---------- Step 4: balance (desirability) ---------- */

/* The most recent /optimize/desirability result and the goals that produced it — cached
 * so changing the desirability map's axis pickers can redraw the contour without paying
 * for another search (the goals/optimum don't change, only which slice is plotted). */
let lastBalanceResult = null;
let lastBalanceGoals = null;

/* Show (or hide) the balance card after a successful analyse(): it only makes sense with
 * 2+ measurements, and only for an all-continuous design — the desirability endpoint
 * searches the coded box, which has no place for a categorical factor (mirrors the
 * `#design-cat-note` gating pattern used for CCD/Box-Behnken). */
function showBalanceCardIfEligible() {
  const card = $("step-balance");
  if (state.responseNames.length < 2) { card.hidden = true; return; }
  card.hidden = false;
  const categorical = hasCategorical();
  $("balance-cat-note").hidden = !categorical;
  $("balance-editor").hidden = categorical;
  $("balance-results").hidden = true;
  clearError("error-balance");
  if (!categorical) renderGoalsTable();
}

/* Default goal per response: maximize, ramping across the observed min/max of that
 * column's attached values (the user can change goal/bounds/weight before running). */
function defaultGoals() {
  return state.responseNames.map((name) => {
    const values = state.design.runs
      .map((r) => r[name])
      .filter((v) => v !== null && v !== undefined);
    const low = Math.min(...values);
    const high = Math.max(...values);
    return { response: name, goal: "max", low, high, target: (low + high) / 2, weight: 1 };
  });
}

function renderGoalsTable() {
  const goals = defaultGoals();
  const rows = goals.map((g) => `
    <tr data-response="${escapeAttr(g.response)}">
      <td>${g.response}</td>
      <td><select class="goal-type">
        <option value="max" selected>Maximize</option>
        <option value="min">Minimize</option>
        <option value="target">Hit target</option>
      </select></td>
      <td><input type="number" class="goal-low" step="any" value="${g.low}"></td>
      <td><input type="number" class="goal-high" step="any" value="${g.high}"></td>
      <td><input type="number" class="goal-target" step="any" value="${g.target}" hidden></td>
      <td><input type="number" class="goal-weight" step="any" value="1" min="0.01"></td>
    </tr>`).join("");
  $("goals-table").innerHTML =
    "<thead><tr><th>Response</th><th>Goal</th><th class=\"num\">Low</th>" +
    "<th class=\"num\">High</th><th class=\"num\">Target</th><th class=\"num\">Weight</th></tr></thead>" +
    `<tbody>${rows}</tbody>`;
  for (const tr of $("goals-table").querySelectorAll("tbody tr")) {
    const sel = tr.querySelector(".goal-type");
    sel.addEventListener("change", () => {
      tr.querySelector(".goal-target").hidden = sel.value !== "target";
    });
  }
}

/* Read the goals table into the wire shape POST /optimize/desirability expects, with the
 * same client-side sanity checks the library itself enforces (low < high; a target goal's
 * target strictly between low and high) so a typo fails fast with a friendly message
 * instead of a round-tripped 422. Every goal is fit as "quadratic", matching fitRequest(). */
function readGoals() {
  const goals = [];
  for (const tr of $("goals-table").querySelectorAll("tbody tr")) {
    const response = tr.dataset.response;
    const goal = tr.querySelector(".goal-type").value;
    const low = parseFloat(tr.querySelector(".goal-low").value);
    const high = parseFloat(tr.querySelector(".goal-high").value);
    const weight = parseFloat(tr.querySelector(".goal-weight").value);
    if (Number.isNaN(low) || Number.isNaN(high)) {
      throw new Error(`"${response}": enter both a low and a high value.`);
    }
    if (low >= high) throw new Error(`"${response}": low must be less than high.`);
    if (Number.isNaN(weight) || weight <= 0) {
      throw new Error(`"${response}": weight must be a positive number.`);
    }
    let target = null;
    if (goal === "target") {
      target = parseFloat(tr.querySelector(".goal-target").value);
      if (Number.isNaN(target)) throw new Error(`"${response}": enter a target value.`);
      if (!(low < target && target < high)) {
        throw new Error(`"${response}": target must be strictly between low and high.`);
      }
    }
    goals.push({ response, model: "quadratic", goal, low, high, target, weight });
  }
  if (!goals.length) throw new Error("Add at least one measurement to balance.");
  return goals;
}

async function runBalance() {
  clearError("error-balance");
  const button = $("run-balance");
  try {
    const goals = readGoals();
    setButtonBusy(button, true, "Searching…", [{ after: 4000, label: "Still working…" }]);
    const result = await api("/optimize/desirability", { design: state.design, goals });
    lastBalanceResult = result;
    lastBalanceGoals = goals;

    $("stat-balance-settings").textContent = formatOptimumSettings(result.natural);
    $("stat-balance-overall").textContent = fmt(result.overall, 3);

    const rows = goals.map((g) => {
      const summary = g.goal === "max" ? `maximize (ramps ${fmt(g.low, 3)} → ${fmt(g.high, 3)})`
        : g.goal === "min" ? `minimize (ramps ${fmt(g.high, 3)} → ${fmt(g.low, 3)})`
        : `hit target ${fmt(g.target, 3)} (range ${fmt(g.low, 3)}–${fmt(g.high, 3)})`;
      return `<tr><td>${g.response}</td>` +
        `<td class="num">${fmt(result.responses[g.response], 4)}</td>` +
        `<td class="num">${fmt(result.individual[g.response], 3)}</td>` +
        `<td>${summary}</td></tr>`;
    });
    $("balance-table").innerHTML =
      "<thead><tr><th>Response</th><th class=\"num\">Predicted value</th>" +
      "<th class=\"num\">Desirability</th><th>Goal</th></tr></thead>" +
      `<tbody>${rows.join("")}</tbody>`;

    const warnings = result.warnings || [];
    $("balance-warnings").hidden = warnings.length === 0;
    $("balance-warnings").textContent = warnings.join("; ");

    $("balance-results").hidden = false;
    setupBalanceAxisPickers();
    await renderBalanceContour();
  } catch (err) {
    showError("error-balance", err);
  } finally {
    setButtonBusy(button, false);
  }
}

/* Its own (simpler) axis-picker pair — the balance view's optimum is independent of step
 * 3's, so this doesn't reuse setupAxisPickers()'s state. Only shown when there are more
 * than two continuous factors to choose between (same gate as the step-3 picker). */
function setupBalanceAxisPickers() {
  const names = continuousFactorNames();
  const box = $("balance-axis-pickers");
  const many = names.length > 2;
  box.hidden = !many;
  if (!many) return;
  box.innerHTML = `<label class="inline-label">Map axes:
    <select id="balance-axis-x"></select> ×
    <select id="balance-axis-y"></select></label>`;
  for (const [id, dflt] of [["balance-axis-x", names[0]], ["balance-axis-y", names[1]]]) {
    const sel = $(id);
    sel.innerHTML = names.map((n) => `<option value="${escapeAttr(n)}">${n}</option>`).join("");
    sel.value = dflt;
  }
  const redraw = () => renderBalanceContour().catch((err) => showError("error-balance", err));
  $("balance-axis-x").addEventListener("change", redraw);
  $("balance-axis-y").addEventListener("change", redraw);
}

/* Client-side mirror of doe.analysis.optimize.ResponseGoal.desirability (see
 * src/doe/analysis/optimize.py:709) — maps a predicted response value to [0, 1] per the
 * goal's ramp. Kept exactly piecewise-identical (including the edge behaviour at low/high
 * and the "outside [low, high] -> 0" branch for a target goal) so the client-drawn
 * desirability map matches what the service itself would compute. */
function desirabilityValue(goal, value) {
  const { goal: kind, low: lo, high: hi, target, weight: wt } = goal;
  if (kind === "max") {
    if (value <= lo) return 0;
    if (value >= hi) return 1;
    return ((value - lo) / (hi - lo)) ** wt;
  }
  if (kind === "min") {
    if (value <= lo) return 1;
    if (value >= hi) return 0;
    return ((hi - value) / (hi - lo)) ** wt;
  }
  // target
  if (value < lo || value > hi) return 0;
  if (value <= target) return ((value - lo) / (target - lo)) ** wt;
  return ((hi - value) / (hi - target)) ** wt;
}

/* Desirability map: one /plot-data/surface call per response (identical x/y/resolution/
 * fixed across all of them, so their grids line up cell-for-cell), each transformed
 * through desirabilityValue() and combined as the unweighted geometric mean (matching
 * doe.analysis.optimize.desirability's overall D), then drawn with the same contour
 * styling as the step-3 response map. */
async function renderBalanceContour() {
  const el = $("balance-contour");
  if (typeof Plotly === "undefined") { el.innerHTML = OFFLINE_PLOT_MSG; return; }
  const names = continuousFactorNames();
  if (names.length < 2) {
    el.innerHTML = "<p class=\"plot-fallback\">A desirability map needs at least two " +
      "continuous factors to draw.</p>";
    return;
  }
  const result = lastBalanceResult;
  const goals = lastBalanceGoals;
  const x = names.length > 2 ? $("balance-axis-x").value : names[0];
  let y = names.length > 2 ? $("balance-axis-y").value : names[1];
  if (x === y) y = names.find((n) => n !== x);

  // Hold every other factor at the optimum's setting (continuous-only — this view is
  // gated off for a categorical design entirely).
  const fixed = {};
  for (const f of state.design.factors) {
    if (f.name === x || f.name === y) continue;
    fixed[f.name] = result.natural[f.name];
  }

  const surfaces = await Promise.all(goals.map((g) => api("/plot-data/surface", {
    design: state.design, response: g.response, model: g.model, x, y, resolution: 40,
    ...(Object.keys(fixed).length ? { fixed } : {}),
  })));

  const gx = surfaces[0].x[0];
  const gy = surfaces[0].y.map((row) => row[0]);
  const combined = surfaces[0].z.map((row, i) => row.map((_, j) => {
    let prod = 1;
    goals.forEach((g, k) => { prod *= desirabilityValue(g, surfaces[k].z[i][j]); });
    return prod ** (1 / goals.length);
  }));

  const label = (n) => {
    const f = state.design.factors.find((ff) => ff.name === n);
    return f.units ? `${n} (${f.units})` : n;
  };

  const traces = [
    {
      type: "contour",
      x: gx, y: gy, z: combined,
      colorscale: CONTOUR_COLORSCALE,
      zmin: 0, zmax: 1,
      line: { width: 0.5, color: "rgba(255,255,255,0.6)" },
      colorbar: { title: { text: "overall desirability", side: "right" }, thickness: 12, outlinewidth: 0 },
      hovertemplate: `${x}: %{x:.4~g}<br>${y}: %{y:.4~g}<br>desirability: %{z:.3~g}<extra></extra>`,
    },
    {
      type: "scatter", mode: "markers", name: "predicted best",
      x: [result.natural[x]], y: [result.natural[y]],
      marker: { color: "#ffffff", size: 14, symbol: "star", line: { color: "#1f2430", width: 1.5 } },
      hovertemplate: "predicted best<extra></extra>",
    },
  ];

  drawPlot(el, traces, {
    margin: { t: 10, r: 10, b: 55, l: 65 },
    xaxis: { title: { text: label(x) } },
    yaxis: { title: { text: label(y) } },
    showlegend: false,
    font: { family: "system-ui, sans-serif", color: "#1f2430" },
    paper_bgcolor: "rgba(0,0,0,0)",
  });
}

/* ---------- wiring ---------- */

EXAMPLE_FACTORS.forEach(addFactorRow);
$("add-factor").addEventListener("click", () => { addFactorRow(); onFactorsChanged(); });
// Recompute the plan hints whenever a factor's name/levels/type changes (event delegation).
$("factor-rows").addEventListener("input", onFactorsChanged);
$("design-type").addEventListener("change", onFactorsChanged);
// Once the user edits the run count, stop auto-overwriting it.
$("dopt-runs").addEventListener("input", (e) => { e.target.dataset.auto = ""; });
$("generate").addEventListener("click", generatePlan);
$("seed-toggle").addEventListener("click", () => { $("seed-wrap").hidden = !$("seed-wrap").hidden; });
onFactorsChanged();
$("load-plan").addEventListener("click", () => $("load-plan-file").click());
$("load-plan-file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) loadDesign(file);
  e.target.value = ""; // reset so re-selecting the same file fires 'change' again
});
$("download-csv").addEventListener("click", downloadCsv);
$("import-results").addEventListener("click", () => $("import-results-file").click());
$("import-results-file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) importResults(file);
  e.target.value = ""; // reset so re-selecting the same file fires 'change' again
});
$("run-table").addEventListener("paste", pasteResults);
$("add-response").addEventListener("click", addResponse);
$("save-plan").addEventListener("click", saveDesign);
$("demo-fill").addEventListener("click", demoFill);
$("analyze").addEventListener("click", analyze);
const rerender = () => renderResults().catch((err) => showError("error-results", err));
$("maximize").addEventListener("change", rerender);
$("axis-x").addEventListener("change", rerender);
$("axis-y").addEventListener("change", rerender);
$("results-response").addEventListener("change", (e) => {
  state.responseName = e.target.value;
  rerender();
});
$("run-balance").addEventListener("click", runBalance);
$("interaction-pick").addEventListener("change", (e) => {
  const pair = interactionContext.pairs.find((p) => p.term === e.target.value);
  renderInteractionPlot(pair).catch((err) => showError("error-results", err));
});
