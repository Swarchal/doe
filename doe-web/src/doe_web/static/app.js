/* DoE planner — a thin, dependency-free UI over the doe-service API mounted at /api.
 * State is a single design document (the Design.to_dict wire format); every step
 * round-trips it through the service unchanged. */

"use strict";

const state = {
  design: null,       // current design document
  responseName: "response",
};

/* Non-factor run columns that are bookkeeping, not measured responses: `std_order` is the
 * standard-order index that `randomize` preserves so results can be re-joined. Excluded when
 * detecting response columns in a loaded design. */
const RESERVED_RUN_COLUMNS = new Set(["std_order"]);

/* Perceptually uniform colormap for the fitted-surface contour. Marker overlays are
 * white-filled with a dark outline so they stay visible on both the dark (low) and
 * bright (high) ends of the scale. */
const CONTOUR_COLORSCALE = "Viridis";

const $ = (id) => document.getElementById(id);

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

/* Keep the plan controls in step with the current factor rows: the categorical note, and
 * the D-optimal run-count field / its suggested default. Called whenever a row or the plan
 * type changes; failures (a half-typed row) are swallowed — the note just stays as-is. */
function onFactorsChanged() {
  let factors = [];
  try { factors = readFactors(); } catch { /* incomplete rows — leave hints unchanged */ }
  const anyCategorical = factors.some((f) => f.type === "categorical");
  const type = $("design-type").value;

  $("design-cat-note").hidden = !(anyCategorical && (type === "ccd" || type === "bb"));

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
    const endpoint = {
      ccd: "/designs/central-composite",
      bb: "/designs/box-behnken",
      ff2: "/designs/full-factorial",
      dopt: "/designs/optimal",
    }[type];
    const body = { factors };
    if (type === "ff2") body.levels = 2;
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
    // Generation can take a moment (the D-optimal coordinate-exchange search especially),
    // so show a spinner once the request is actually in flight — after all validation above.
    // If it runs long, escalate the label so it clearly hasn't hung.
    setButtonBusy($("generate"), true, "Creating plan…", [
      { after: 4000, label: "Still working…" },
      { after: 12000, label: "Hang tight — optimising…" },
    ]);
    const { design } = await api(endpoint, body);
    // Randomize the run order so the sheet is bench-ready.
    state.design = (await api("/designs/randomize", { design })).design;
    renderPlan();
    $("step-plan").hidden = false;
    $("step-results").hidden = true;
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

function renderPlan() {
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
    ...(showPointType ? ["Point type"] : []), "Your measurement"];
  const rows = d.runs.map((run, i) => {
    const cells = names.map((n) => cellFor(n, run[n])).join("");
    const pt = showPointType ? `<td class="point-type">${d.point_types[i]}</td>` : "";
    return `<tr><td class="num">${i + 1}</td>${cells}${pt}` +
      `<td><input type="number" class="resp" step="any" data-run="${i}"></td></tr>`;
  });
  $("run-table").innerHTML =
    `<thead><tr>${head.map((h) => `<th${h === "Run" ? ' class="num"' : ""}>${h}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.join("")}</tbody>`;
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
  header.push(state.responseName || "response");

  const lines = d.runs.map((run, i) => {
    const cells = [i + 1];
    for (const f of d.factors) {
      cells.push(run[f.name]);
      if (f.units) cells.push(f.units);
    }
    if (showPointType) cells.push(d.point_types[i]);
    cells.push(""); // empty response cell to fill in at the bench
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

/* Save the current design as a JSON file — the same Design.to_dict() document the service
 * speaks, so it round-trips through Load (and any other doe tooling). Any measurements typed
 * but not yet analysed are folded in first, so a work-in-progress sheet is saved intact. */
function saveDesign() {
  if (!state.design) return;
  let design = state.design;
  const inputs = document.querySelectorAll("input.resp");
  if (inputs.length) {
    const name = ($("response-name").value.trim() || "response");
    const runs = state.design.runs.map((r) => ({ ...r }));
    let any = false;
    inputs.forEach((input) => {
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

    // Rebuild step 1 so the factor table reflects the loaded design.
    $("factor-rows").innerHTML = "";
    for (const f of design.factors) addFactorRow(f);
    onFactorsChanged();

    // A measured response is a run column that is neither a factor nor bookkeeping
    // (`std_order` et al.); pre-fill the first one so the results can be analysed.
    const factorSet = new Set(design.factors.map((f) => f.name));
    const responseCols = design.runs.length
      ? Object.keys(design.runs[0]).filter((k) => !factorSet.has(k) && !RESERVED_RUN_COLUMNS.has(k))
      : [];

    renderPlan();
    $("step-plan").hidden = false;
    $("step-results").hidden = true;

    if (responseCols.length) {
      state.responseName = responseCols[0];
      $("response-name").value = state.responseName;
      for (const input of document.querySelectorAll("input.resp")) {
        const v = design.runs[Number(input.dataset.run)][state.responseName];
        if (v !== null && v !== undefined) input.value = v;
      }
    }
    $("step-plan").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    showError("error-factors", err);
  }
}

/* Deterministic synthetic response so the whole flow can be demoed without a lab:
 * a smooth quadratic over the first two factors (in coded units) plus a small
 * run-dependent wiggle standing in for noise. */
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
    const wiggle = 0.3 * Math.sin(37 * (a + 2 * b) + Number(input.dataset.run));
    input.value = (70 + 4 * a + 3 * b - 3 * a * a - 2 * b * b + 1.5 * a * b + catBump + wiggle).toFixed(2);
  });
}

/* ---------- Step 3: analysis ---------- */

function readResponses() {
  const values = [];
  for (const input of document.querySelectorAll("input.resp")) {
    const v = parseFloat(input.value);
    if (Number.isNaN(v)) {
      throw new Error(`Run ${Number(input.dataset.run) + 1} has no measurement yet — ` +
        "fill in every row (or use the demo button).");
    }
    values[Number(input.dataset.run)] = v;
  }
  return values;
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
    state.responseName = $("response-name").value.trim() || "response";
    const responses = readResponses();
    // Drop any existing column of this name first: `with_response` refuses to overwrite, so
    // a loaded-with-results plan (or a re-analysis with edited numbers) would otherwise clash.
    const base = designWithoutColumn(state.design, state.responseName);
    const attached = await api("/designs/responses", {
      design: base,
      responses: { [state.responseName]: responses },
    });
    state.design = attached.design;
    await renderResults();
    $("step-results").hidden = false;
    $("step-results").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    showError("error-plan", err);
  }
}

function fitRequest() {
  return { design: state.design, response: state.responseName, model: "quadratic" };
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
  const categorical = hasCategorical();
  const maximize = $("maximize").checked;

  // The optimum for an all-continuous fit comes from the service's surface optimizer; a
  // categorical fit has no coded box to search, so we grid-search it here through /predict.
  const fit = await api("/analysis/fit", fitRequest());
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

  $("stat-r2").textContent = fmt(fit.r_squared, 3);
  $("stat-adjr2").textContent = fit.adjusted_r2 === null ? "" : `adjusted: ${fmt(fit.adjusted_r2, 3)}`;

  renderTermsTable(fit.terms);
  setupAxisPickers(opt);
  await renderContour(opt.natural);
}

function renderTermsTable(terms) {
  const rows = terms
    .filter((t) => t.term !== "Intercept")
    .map((t) => {
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

  Plotly.react(el, traces, {
    margin: { t: 10, r: 10, b: 55, l: 65 },
    xaxis: { title: { text: label(x) } },
    yaxis: { title: { text: label(y) } },
    showlegend: false,
    font: { family: "system-ui, sans-serif", color: "#1f2430" },
    paper_bgcolor: "rgba(0,0,0,0)",
  }, { responsive: true, displaylogo: false });
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
onFactorsChanged();
$("load-plan").addEventListener("click", () => $("load-plan-file").click());
$("load-plan-file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) loadDesign(file);
  e.target.value = ""; // reset so re-selecting the same file fires 'change' again
});
$("download-csv").addEventListener("click", downloadCsv);
$("save-plan").addEventListener("click", saveDesign);
$("demo-fill").addEventListener("click", demoFill);
$("analyze").addEventListener("click", analyze);
const rerender = () => renderResults().catch((err) => showError("error-results", err));
$("maximize").addEventListener("change", rerender);
$("axis-x").addEventListener("change", rerender);
$("axis-y").addEventListener("change", rerender);
