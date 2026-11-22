// Fearless Draft Lab — draft simulator + opponent scouting (static app, no build step).
// Data: ../data/aggregates.json (aggregate statistics only; no raw GRID data),
//       model.json (logistic-regression draft win-rate model, trained offline),
//       ../data/patch_event.json (aggregate patch/event summaries).

"use strict";

// Standard LoL draft order: 6 bans, 6 picks, 4 bans, 4 picks.
const DRAFT_ORDER = [
  { side: "blue", type: "ban" }, { side: "red", type: "ban" },
  { side: "blue", type: "ban" }, { side: "red", type: "ban" },
  { side: "blue", type: "ban" }, { side: "red", type: "ban" },
  { side: "blue", type: "pick" }, { side: "red", type: "pick" },
  { side: "red", type: "pick" }, { side: "blue", type: "pick" },
  { side: "blue", type: "pick" }, { side: "red", type: "pick" },
  { side: "red", type: "ban" }, { side: "blue", type: "ban" },
  { side: "red", type: "ban" }, { side: "blue", type: "ban" },
  { side: "red", type: "pick" }, { side: "blue", type: "pick" },
  { side: "blue", type: "pick" }, { side: "red", type: "pick" },
];

let AGG = null;            // aggregates.json contents
let MODEL = null;          // model.json contents (may stay null)
let PATCH_EVENT = null;    // patch_event.json contents (may stay null)
let CHAMP_NAME = {};       // champion id -> display name
let TEAM_NAME = {};        // team id -> display name
let PLAYER_NAME = {};      // player id -> display name
let PLAYER_TEAM = {};      // player id -> team id

const state = {
  started: false,
  format: 5,               // best-of
  fearless: "hard",        // hard | soft | off
  blueName: "Cloud9 Kia",
  redName: "Opponent",
  currentGame: 1,          // 1-based
  viewingGame: 1,          // game shown on the board (may be a finished game)
  games: {},               // gameNumber -> { actions: [{champId, side, type}], winner: 'blue'|'red'|null }
  score: { blue: 0, red: 0 },
  seriesOver: false,
};

// ---------- Data loading ----------

async function loadJson(candidates) {
  for (const url of candidates) {
    try {
      const res = await fetch(url);
      if (res.ok) return await res.json();
    } catch (e) { /* try next */ }
  }
  return null;
}

async function loadAll() {
  AGG = await loadJson(["../data/aggregates.json", "data/aggregates.json"]);
  MODEL = await loadJson(["model.json", "app/model.json"]);
  PATCH_EVENT = await loadJson(["../data/patch_event.json", "data/patch_event.json"]);
}

function initData() {
  for (const c of AGG.champions) CHAMP_NAME[c.id] = c.name;
  for (const t of AGG.teams) TEAM_NAME[t.id] = t.name;
  for (const p of AGG.players) {
    PLAYER_NAME[p.id] = p.name;
    PLAYER_TEAM[p.id] = p.team_id;
  }

  const m = AGG.meta;
  document.getElementById("dataBadge").textContent =
    `Data: ${m.series_processed} series · ${m.games_processed} games with draft data · ` +
    `${AGG.champions.length} champions · ${AGG.teams.length} teams (${m.source_seasons})`;

  // Team dropdown (sorted by games played desc)
  const sel = document.getElementById("teamSelect");
  const teams = [...AGG.teams].sort((a, b) => (b.games || 0) - (a.games || 0));
  for (const t of teams) {
    const opt = document.createElement("option");
    opt.value = t.id;
    opt.textContent = `${t.name} (${t.games || 0} games)`;
    sel.appendChild(opt);
  }

  // Patch / event dropdowns
  if (PATCH_EVENT) {
    const patchSel = document.getElementById("patchSelect");
    const patches = Object.keys(PATCH_EVENT.patches || {})
      .sort((a, b) => (PATCH_EVENT.patches[b].games - PATCH_EVENT.patches[a].games));
    for (const p of patches) {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = `Patch ${p} (${PATCH_EVENT.patches[p].games} games)`;
      patchSel.appendChild(opt);
    }
    const eventSel = document.getElementById("eventSelect");
    const events = Object.keys(PATCH_EVENT.events || {})
      .sort((a, b) => (PATCH_EVENT.events[b].games - PATCH_EVENT.events[a].games));
    for (const e of events) {
      const opt = document.createElement("option");
      opt.value = e;
      opt.textContent = `${e} (${PATCH_EVENT.events[e].games} games)`;
      eventSel.appendChild(opt);
    }
  }

  renderInsights();
  renderComfortInsights();
  renderShallowInsights();
  renderFearlessStats();
  renderChampWr();
  renderChampGrid();
}

// ---------- Fearless logic ----------

// Champions picked in completed games before the current game.
function lockedSets() {
  const hard = new Set();
  const soft = { blue: new Set(), red: new Set() };
  for (let g = 1; g < state.currentGame; g++) {
    const game = state.games[g];
    if (!game) continue;
    for (const a of game.actions) {
      if (a.type !== "pick") continue;
      hard.add(a.champId);
      soft[a.side].add(a.champId);
    }
  }
  return { hard, soft };
}

// Locked champions with provenance: [{champId, game, side}]
function lockedWithProvenance() {
  const out = [];
  for (let g = 1; g < state.currentGame; g++) {
    const game = state.games[g];
    if (!game) continue;
    for (const a of game.actions) {
      if (a.type === "pick") out.push({ champId: a.champId, game: g, side: a.side });
    }
  }
  return out;
}

function isLocked(champId, side) {
  if (state.fearless === "off") return false;
  const { hard, soft } = lockedSets();
  if (state.fearless === "hard") return hard.has(champId);
  return soft[side].has(champId);
}

function currentGameActions() {
  const g = state.games[state.viewingGame];
  return g ? g.actions : [];
}

function usedThisGame() {
  const used = new Set();
  for (const a of currentGameActions()) used.add(a.champId);
  return used;
}

function isAvailable(champId) {
  if (!state.started || state.viewingGame !== state.currentGame) return false;
  if (usedThisGame().has(champId)) return false;
  const step = DRAFT_ORDER[currentGameActions().length];
  if (!step) return false;
  return !isLocked(champId, step.side);
}

// ---------- Win-rate model ----------

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

// P(Blue wins) from the picks made so far in a game (bans do not enter the model).
function predictBlueWinProb(actions, gameNumber) {
  if (!MODEL) return null;
  let z = MODEL.intercept + (MODEL.game_weights[String(gameNumber)] || 0);
  for (const a of actions) {
    if (a.type !== "pick") continue;
    const w = MODEL.champion_weights[a.champId] || 0;
    z += a.side === "blue" ? w : -w;
  }
  return sigmoid(z);
}

// Rank available champions for the side to act.
// Pick step: rank by gain in the acting side's win probability.
// Ban step: rank by deny value — how much the opponent's win probability
// would rise if they picked this champion next.
function rankSuggestions() {
  if (!MODEL || !state.started || state.viewingGame !== state.currentGame || state.seriesOver) return null;
  const actions = currentGameActions();
  const step = DRAFT_ORDER[actions.length];
  if (!step) return null;
  const base = predictBlueWinProb(actions, state.currentGame);
  const scored = [];
  for (const c of AGG.champions) {
    if (!isAvailable(c.id)) continue;
    const w = MODEL.champion_weights[c.id] || 0;
    let gain;
    if (step.type === "pick") {
      // Acting side picks c: P(blue) moves by +w (blue) or -w (red).
      const after = sigmoid((Math.log(base / (1 - base))) + (step.side === "blue" ? w : -w));
      gain = step.side === "blue" ? after - base : base - after;
    } else {
      // Opponent picks c next: how much does the acting side lose?
      const oppSide = step.side === "blue" ? "red" : "blue";
      const after = sigmoid((Math.log(base / (1 - base))) + (oppSide === "blue" ? w : -w));
      gain = step.side === "blue" ? base - after : after - base;
    }
    scored.push({ id: c.id, gain });
  }
  scored.sort((a, b) => b.gain - a.gain);
  return { step, base, top: scored.slice(0, 5) };
}

// ---------- Rendering ----------

function champLabel(id) {
  return CHAMP_NAME[id] || id;
}

function pct(x, digits = 0) {
  return (x * 100).toFixed(digits) + "%";
}

function renderAll() {
  renderSetup();
  renderPool();
  renderBoard();
  renderChampGrid();
  renderTabs();
  renderLockedList();
  renderHistory();
  renderWinrate();
  renderSuggestions();
}

function renderSetup() {
  document.getElementById("blueTitle").textContent = `Blue — ${state.blueName}`;
  document.getElementById("redTitle").textContent = `Red — ${state.redName}`;
  document.getElementById("seriesScore").textContent =
    state.started
      ? `Series score: ${state.blueName} ${state.score.blue} — ${state.score.red} ${state.redName}` +
        (state.seriesOver ? " — series decided." : "")
      : "";
}

function renderPool() {
  const total = AGG ? AGG.champions.length : 0;
  const { hard, soft } = lockedSets();
  let lockedCount;
  if (state.fearless === "hard") lockedCount = hard.size;
  else if (state.fearless === "soft") lockedCount = new Set([...soft.blue, ...soft.red]).size;
  else lockedCount = 0;

  const actions = currentGameActions();
  const banned = actions.filter((a) => a.type === "ban").length;
  const picked = actions.filter((a) => a.type === "pick").length;
  const available = Math.max(0, total - lockedCount - banned - picked);

  document.getElementById("gameLabel").textContent = `Game ${state.viewingGame}`;

  const bar = document.getElementById("poolBar");
  const pctBar = (n) => (total ? ((n / total) * 100).toFixed(1) : 0);
  bar.innerHTML =
    `<div class="seg-available" style="width:${pctBar(available)}%" title="Available"></div>` +
    `<div class="seg-locked" style="width:${pctBar(lockedCount)}%" title="Fearless-locked"></div>` +
    `<div class="seg-banned" style="width:${pctBar(banned)}%" title="Banned this game"></div>` +
    `<div class="seg-picked" style="width:${pctBar(picked)}%" title="Picked this game"></div>`;

  document.getElementById("poolStats").textContent =
    `${total} champions · ${available} available · ${lockedCount} Fearless-locked · ` +
    `${banned} banned · ${picked} picked (Game ${state.viewingGame})`;

  // Observed Fearless pool math from the dataset.
  const mathEl = document.getElementById("fearlessMath");
  if (AGG && AGG.fearless && AGG.fearless.locked_per_game) {
    const lp = AGG.fearless.locked_per_game;
    mathEl.textContent =
      `Observed in ${AGG.fearless.fearless_like_series} Fearless series in the dataset — ` +
      `champions locked per game: Game 2: ${lp["2"].avg_locked} · Game 3: ${lp["3"].avg_locked} · ` +
      `Game 4: ${lp["4"].avg_locked} · Game 5: ${lp["5"].avg_locked}. ` +
      `A Bo5 Game 5 starts with 40 champions gone before the first ban.`;
  }
}

function renderLockedList() {
  const el = document.getElementById("lockedList");
  if (!state.started || state.fearless === "off" || state.currentGame <= 1) {
    el.innerHTML = "";
    return;
  }
  const locked = lockedWithProvenance();
  const byGame = {};
  for (const l of locked) {
    (byGame[l.game] = byGame[l.game] || []).push(l);
  }
  let html = `<h3>Fearless-locked champions (${locked.length})</h3><div class="locked-groups">`;
  for (const g of Object.keys(byGame).sort((a, b) => a - b)) {
    const names = byGame[g].map((l) =>
      `${champLabel(l.champId)} <span class="locked-side ${l.side}">(${l.side === "blue" ? state.blueName : state.redName})</span>`
    ).join(", ");
    html += `<div class="locked-group"><strong>Game ${g}:</strong> ${names}</div>`;
  }
  html += "</div>";
  el.innerHTML = html;
}

function renderTabs() {
  const tabs = document.getElementById("gameTabs");
  tabs.innerHTML = "";
  const maxGame = state.started ? Math.min(state.format, state.currentGame) : 1;
  for (let g = 1; g <= maxGame; g++) {
    const btn = document.createElement("button");
    btn.className = "game-tab" + (g === state.viewingGame ? " active" : "") +
      (state.games[g] && state.games[g].winner ? " done" : "");
    const w = state.games[g] && state.games[g].winner;
    btn.textContent = w ? `Game ${g} ✓ ${w === "blue" ? state.blueName : state.redName}` : `Game ${g}`;
    btn.addEventListener("click", () => { state.viewingGame = g; renderAll(); });
    tabs.appendChild(btn);
  }
}

function slotEl(side, type, idx, champId) {
  const div = document.createElement("div");
  div.className = `slot ${type}` + (champId ? " filled" : "");
  const label = document.createElement("span");
  label.textContent = champId ? champLabel(champId) : `${type === "ban" ? "Ban" : "Pick"} ${idx + 1}`;
  div.appendChild(label);
  return div;
}

function renderBoard() {
  const actions = currentGameActions();
  const bySide = {
    blue: { ban: [], pick: [] },
    red: { ban: [], pick: [] },
  };
  for (const a of actions) bySide[a.side][a.type].push(a.champId);

  for (const side of ["blue", "red"]) {
    const bansEl = document.getElementById(side + "Bans");
    const picksEl = document.getElementById(side + "Picks");
    bansEl.innerHTML = "";
    picksEl.innerHTML = "";
    for (let i = 0; i < 5; i++) bansEl.appendChild(slotEl(side, "ban", i, bySide[side].ban[i]));
    for (let i = 0; i < 5; i++) picksEl.appendChild(slotEl(side, "pick", i, bySide[side].pick[i]));
  }

  const indicator = document.getElementById("turnIndicator");
  const finishBtn = document.getElementById("finishGameBtn");
  const undoBtn = document.getElementById("undoBtn");

  if (!state.started) {
    indicator.textContent = "Start a series to begin the draft.";
    indicator.className = "turn-indicator";
    finishBtn.disabled = true;
    undoBtn.disabled = true;
    return;
  }

  const viewingCurrent = state.viewingGame === state.currentGame;
  const step = DRAFT_ORDER[actions.length];
  if (state.seriesOver) {
    indicator.textContent = `Series over — ${state.score.blue > state.score.red ? state.blueName : state.redName} wins ${Math.max(state.score.blue, state.score.red)}–${Math.min(state.score.blue, state.score.red)}.`;
    indicator.className = "turn-indicator";
  } else if (!step) {
    indicator.textContent = `Draft complete for Game ${state.viewingGame}. Record the winner to advance.`;
    indicator.className = "turn-indicator";
  } else if (!viewingCurrent) {
    indicator.textContent = `Viewing finished Game ${state.viewingGame}. Switch to Game ${state.currentGame} to continue drafting.`;
    indicator.className = "turn-indicator";
  } else {
    const sideName = step.side === "blue" ? state.blueName : state.redName;
    indicator.textContent = `${sideName} (${step.side}) — ${step.type.toUpperCase()} ${bySide[step.side][step.type].length + 1} (step ${actions.length + 1}/20)`;
    indicator.className = `turn-indicator ${step.side}-turn`;
  }

  finishBtn.disabled = !(viewingCurrent && !step && !state.seriesOver);
  undoBtn.disabled = !(viewingCurrent && actions.length > 0 && !state.seriesOver);
}

function renderWinrate() {
  const panel = document.getElementById("winratePanel");
  if (!state.started) {
    panel.innerHTML = "";
    return;
  }
  if (!MODEL) {
    panel.innerHTML = `<p class="hint">Win-rate model not loaded — run pipeline/train_model.py to generate app/model.json.</p>`;
    return;
  }
  const actions = currentGameActions();
  const p = predictBlueWinProb(actions, state.viewingGame);
  const bluePct = (p * 100).toFixed(1);
  const redPct = ((1 - p) * 100).toFixed(1);
  const m = MODEL.metrics;
  panel.innerHTML =
    `<h3>Live draft win-rate — Game ${state.viewingGame}</h3>` +
    `<div class="wr-bar"><div class="wr-blue" style="width:${bluePct}%"></div><div class="wr-red" style="width:${redPct}%"></div></div>` +
    `<div class="wr-labels"><span class="wr-blue-text">${state.blueName}: ${bluePct}%</span>` +
    `<span class="wr-red-text">${state.redName}: ${redPct}%</span></div>` +
    `<p class="hint">Logistic regression on draft composition, trained on ${MODEL.n_games} games. ` +
    `Held-out accuracy ${pct(m.test_accuracy, 1)} vs ${pct(m.majority_baseline, 1)} baseline — ` +
    `a directional signal, not a forecast. Bans do not enter the model.</p>`;
}

function renderSuggestions() {
  const el = document.getElementById("suggestions");
  const ranked = rankSuggestions();
  if (!ranked) {
    el.innerHTML = "";
    return;
  }
  const { step, top } = ranked;
  const sideName = step.side === "blue" ? state.blueName : state.redName;
  const title = step.type === "pick"
    ? `Suggested picks for ${sideName} — biggest win-rate gain`
    : `Suggested bans for ${sideName} — deny the opponent's best pick`;
  let html = `<h3>${title}</h3><ol class="suggestion-list">`;
  for (const s of top) {
    const w = AGG.champion_winrate[s.id];
    const wrText = w && w.picks >= 5
      ? ` · season WR ${pct(w.winrate)} (n=${w.picks})` + (w.picks < 20 ? " <span class='small-sample'>small sample</span>" : "")
      : "";
    const delta = (s.gain * 100).toFixed(1);
    html += `<li><button class="suggest-btn" data-champ="${s.id}">${champLabel(s.id)}</button> ` +
      `<span class="rate">${step.type === "pick" ? "+" : "deny +" }${delta} pts${wrText}</span></li>`;
  }
  html += "</ol>";
  el.innerHTML = html;
  el.querySelectorAll(".suggest-btn").forEach((btn) =>
    btn.addEventListener("click", () => pickChampion(btn.getAttribute("data-champ"))));
}

function renderHistory() {
  const el = document.getElementById("historyList");
  const finished = Object.keys(state.games)
    .map(Number)
    .sort((a, b) => a - b)
    .filter((g) => state.games[g].winner);
  if (finished.length === 0) {
    el.innerHTML = `<p class="hint">No finished games yet. Complete a draft and record a winner to build the history.</p>`;
    return;
  }
  let html = "";
  for (const g of finished) {
    const game = state.games[g];
    const bySide = { blue: { ban: [], pick: [] }, red: { ban: [], pick: [] } };
    for (const a of game.actions) bySide[a.side][a.type].push(champLabel(a.champId));
    const winnerName = game.winner === "blue" ? state.blueName : state.redName;
    html += `<div class="history-game"><strong>Game ${g}</strong> — winner: ${winnerName}<br>` +
      `<span class="hist-blue">${state.blueName} picks: ${bySide.blue.pick.join(", ") || "—"} · bans: ${bySide.blue.ban.join(", ") || "—"}</span><br>` +
      `<span class="hist-red">${state.redName} picks: ${bySide.red.pick.join(", ") || "—"} · bans: ${bySide.red.ban.join(", ") || "—"}</span></div>`;
  }
  el.innerHTML = html;
}

function renderChampGrid() {
  const grid = document.getElementById("champGrid");
  const query = (document.getElementById("champSearch").value || "").toLowerCase();
  grid.innerHTML = "";
  if (!AGG) {
    grid.innerHTML = "<p>Champion data not loaded — run the pipeline to generate data/aggregates.json, then serve this project from its root folder.</p>";
    return;
  }
  const wr = AGG.champion_winrate || {};
  const champs = [...AGG.champions]
    .filter((c) => c.name.toLowerCase().includes(query))
    .sort((a, b) => a.name.localeCompare(b.name));
  for (const c of champs) {
    const btn = document.createElement("button");
    btn.className = "champ-btn";
    btn.disabled = !isAvailable(c.id);
    const w = wr[c.id];
    const wrText = w && w.picks >= 5 ? `WR ${pct(w.winrate)} (n=${w.picks})` : "";
    const modelW = MODEL ? MODEL.champion_weights[c.id] : null;
    const modelText = modelW !== null && modelW !== undefined && Math.abs(modelW) >= 0.3
      ? `model ${modelW > 0 ? "+" : ""}${modelW.toFixed(2)}` : "";
    btn.innerHTML = `${c.name}${wrText ? `<span class="wr">${wrText}</span>` : ""}${modelText ? `<span class="wr model-w">${modelText}</span>` : ""}`;
    btn.title = isLocked(c.id, "blue") || isLocked(c.id, "red")
      ? "Fearless-locked from an earlier game"
      : usedThisGame().has(c.id) ? "Already picked or banned this game" : c.name;
    btn.addEventListener("click", () => pickChampion(c.id));
    grid.appendChild(btn);
  }
}

// ---------- Draft actions ----------

function pickChampion(champId) {
  if (!state.started || state.viewingGame !== state.currentGame || state.seriesOver) return;
  const actions = currentGameActions();
  const step = DRAFT_ORDER[actions.length];
  if (!step || !isAvailable(champId)) return;
  actions.push({ champId, side: step.side, type: step.type });
  renderAll();
}

function undoLast() {
  const actions = currentGameActions();
  if (actions.length === 0) return;
  actions.pop();
  renderAll();
}

function finishGame() {
  const actions = currentGameActions();
  if (DRAFT_ORDER[actions.length]) return; // draft not complete
  const winner = document.getElementById("winnerSelect").value;
  state.games[state.currentGame].winner = winner;
  state.score[winner] += 1;
  const need = Math.floor(state.format / 2) + 1;
  if (state.score[winner] >= need || state.currentGame >= state.format) {
    state.seriesOver = true;
  } else {
    state.currentGame += 1;
    state.viewingGame = state.currentGame;
    state.games[state.currentGame] = { actions: [], winner: null };
  }
  renderAll();
}

function startSeries() {
  state.format = parseInt(document.getElementById("formatSelect").value, 10);
  state.fearless = document.getElementById("fearlessSelect").value;
  state.blueName = document.getElementById("blueName").value.trim() || "Blue";
  state.redName = document.getElementById("redName").value.trim() || "Red";
  state.started = true;
  state.currentGame = 1;
  state.viewingGame = 1;
  state.games = { 1: { actions: [], winner: null } };
  state.score = { blue: 0, red: 0 };
  state.seriesOver = false;
  document.getElementById("winnerSelect").innerHTML =
    `<option value="blue">${state.blueName}</option><option value="red">${state.redName}</option>`;
  renderAll();
}

function resetSeries() {
  state.started = false;
  state.currentGame = 1;
  state.viewingGame = 1;
  state.games = {};
  state.score = { blue: 0, red: 0 };
  state.seriesOver = false;
  renderAll();
}

// ---------- Scouting ----------

function renderTendencies() {
  const teamId = document.getElementById("teamSelect").value;
  const gameNum = document.getElementById("scoutGameSelect").value;
  const picksEl = document.getElementById("topPicks");
  const bansEl = document.getElementById("topBans");
  const sampleEl = document.getElementById("scoutSample");
  picksEl.innerHTML = "";
  bansEl.innerHTML = "";
  renderPoolDepth(teamId);
  if (!teamId || !AGG) {
    sampleEl.textContent = "";
    return;
  }
  const entry = (AGG.team_pick_ban_by_game[teamId] || {})[gameNum];
  if (!entry) {
    sampleEl.textContent = "No data for this team in this game number.";
    return;
  }
  const top = (obj, n) => Object.entries(obj || {})
    .map(([id, v]) => ({ id, ...v }))
    .sort((a, b) => b.count - a.count)
    .slice(0, n);
  for (const p of top(entry.picks, 5)) {
    const li = document.createElement("li");
    li.innerHTML = `${champLabel(p.id)} <span class="rate">${pct(p.rate)} — ${p.count}/${entry.games} games</span>`;
    picksEl.appendChild(li);
  }
  for (const b of top(entry.bans, 5)) {
    const li = document.createElement("li");
    li.innerHTML = `${champLabel(b.id)} <span class="rate">${pct(b.rate)} — ${b.count}/${entry.games} games</span>`;
    bansEl.appendChild(li);
  }
  sampleEl.textContent = `Based on ${entry.games} games (Game ${gameNum}) for ${TEAM_NAME[teamId]}.`;
}

function renderPoolDepth(teamId) {
  const el = document.getElementById("poolDepth");
  if (!teamId || !AGG) {
    el.innerHTML = `<p class="hint">Select a team to see its players' champion pools.</p>`;
    return;
  }
  const players = AGG.players
    .filter((p) => p.team_id === teamId)
    .map((p) => ({ ...p, pool: AGG.player_champion_pool[p.id] }))
    .filter((p) => p.pool && p.pool.games >= 3)
    .sort((a, b) => b.pool.games - a.pool.games)
    .slice(0, 5);
  if (players.length === 0) {
    el.innerHTML = `<p class="hint">No player pool data for ${TEAM_NAME[teamId]}.</p>`;
    return;
  }
  let html = `<table class="pool-table"><thead><tr><th>Player</th><th>Games</th><th>Champions</th><th>Most played</th></tr></thead><tbody>`;
  for (const p of players) {
    const topChamps = Object.entries(p.pool.champions)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([id, n]) => `${champLabel(id)} (${n})`)
      .join(", ");
    const shallow = p.pool.unique <= 5
      ? ` <span class="small-sample">shallow pool — target it</span>` : "";
    html += `<tr><td>${p.name}</td><td>${p.pool.games}</td><td>${p.pool.unique}${shallow}</td><td>${topChamps}</td></tr>`;
  }
  html += "</tbody></table>";
  el.innerHTML = html;
}

// ---------- Patch & event context ----------

function renderPatchEvent() {
  const el = document.getElementById("patchEventInfo");
  if (!PATCH_EVENT) {
    el.innerHTML = `<p class="hint">Patch/event data not loaded — run pipeline/train_model.py to generate data/patch_event.json.</p>`;
    return;
  }
  const patch = document.getElementById("patchSelect").value;
  const event = document.getElementById("eventSelect").value;
  let html = "";
  const renderGroup = (label, entry) => {
    if (!entry) return "";
    const top = entry.top_picks.slice(0, 5).map((tp) =>
      `<li>${champLabel(tp.champion_id)} <span class="rate">${tp.picks} picks · WR ${pct(tp.winrate)}</span></li>`
    ).join("");
    return `<div><h3>${label}</h3><p class="hint">${entry.games} games · Blue-side win rate ${pct(entry.blue_winrate)}</p>` +
      `<ol class="tendency-list">${top}</ol></div>`;
  };
  if (patch) html += renderGroup(`Patch ${patch}`, PATCH_EVENT.patches[patch]);
  if (event) html += renderGroup(event, PATCH_EVENT.events[event]);
  el.innerHTML = html || `<p class="hint">Pick a patch or an event to see its most-picked champions and Blue-side win rate.</p>`;
}

// ---------- Champion win rates ----------

function renderChampWr() {
  const topEl = document.getElementById("wrTop");
  const botEl = document.getElementById("wrBottom");
  if (!AGG) return;
  const rated = Object.entries(AGG.champion_winrate)
    .filter(([, v]) => v.picks >= 30)
    .map(([id, v]) => ({ id, ...v }));
  const top = [...rated].sort((a, b) => b.winrate - a.winrate).slice(0, 5);
  const bot = [...rated].sort((a, b) => a.winrate - b.winrate).slice(0, 5);
  topEl.innerHTML = top.map((c) =>
    `<li>${champLabel(c.id)} <span class="rate">WR ${pct(c.winrate)} — ${c.wins}/${c.picks} games</span></li>`).join("");
  botEl.innerHTML = bot.map((c) =>
    `<li>${champLabel(c.id)} <span class="rate">WR ${pct(c.winrate)} — ${c.wins}/${c.picks} games</span></li>`).join("");
}

// ---------- Insights ----------

function renderInsights() {
  const list = document.getElementById("insightsList");
  list.innerHTML = "";
  if (!AGG || !AGG.fearless || !AGG.fearless.key_ban_shifts) {
    list.innerHTML = "<p>No insights available yet.</p>";
    return;
  }
  const shifts = [...AGG.fearless.key_ban_shifts]
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
    .slice(0, 8);
  if (shifts.length === 0) {
    list.innerHTML = "<p>No key-ban insights met the sample-size threshold yet.</p>";
    return;
  }
  for (const s of shifts) {
    const div = document.createElement("div");
    div.className = "insight";
    const team = TEAM_NAME[s.team_id] || s.team_id;
    const champ = champLabel(s.champion_id);
    const dir = s.delta < 0 ? "drops" : "rises";
    const small = s.n_banned < 5 || s.n_not_banned < 5;
    div.innerHTML =
      `<strong>${team} picks ${champ} ${pct(s.game1_pick_rate)} of the time in Game 1.</strong> ` +
      `When ${champ} is banned in Game 2, their Game 2 win rate ${dir} by ${Math.abs(s.delta * 100).toFixed(0)} percentage points ` +
      `(${pct(s.game2_wr_when_banned)} vs ${pct(s.game2_wr_when_not_banned)}). ` +
      (s.delta < 0 ? `Recommend banning ${champ} early.` : `Banning ${champ} does not seem to hurt them.`) +
      `<div class="sample">Sample: ${s.n_banned} series where banned, ${s.n_not_banned} where not banned. ` +
      (small ? `<span class="small-sample">Small sample — treat as a lead.</span>` : "") + `</div>`;
    list.appendChild(div);
  }
}

function renderComfortInsights() {
  const list = document.getElementById("comfortList");
  list.innerHTML = "";
  if (!AGG) return;
  const found = [];
  for (const [teamId, byGame] of Object.entries(AGG.team_pick_ban_by_game)) {
    const g1 = byGame["1"];
    if (!g1 || g1.games < 10) continue;
    for (const [champId, p] of Object.entries(g1.picks || {})) {
      const wr = AGG.champion_winrate[champId];
      if (p.rate >= 0.30 && wr && wr.picks >= 10 && wr.winrate >= 0.55) {
        found.push({ teamId, champId, rate: p.rate, games: g1.games, wr: wr.winrate, picks: wr.picks });
      }
    }
  }
  found.sort((a, b) => b.rate - a.rate);
  const top = found.slice(0, 8);
  if (top.length === 0) {
    list.innerHTML = "<p>No comfort-pick insight met the thresholds yet.</p>";
    return;
  }
  for (const c of top) {
    const div = document.createElement("div");
    div.className = "insight";
    div.innerHTML =
      `<strong>${TEAM_NAME[c.teamId]} relies on ${champLabel(c.champId)} in Game 1 (${pct(c.rate)} of ${c.games} games).</strong> ` +
      `${champLabel(c.champId)} wins ${pct(c.wr)} across the dataset (n=${c.picks}). ` +
      `Banning it forces them off a proven comfort pick.` +
      `<div class="sample">Sample: ${c.games} Game 1s for the team; ${c.picks} picks of the champion overall.</div>`;
    list.appendChild(div);
  }
}

function renderShallowInsights() {
  const list = document.getElementById("shallowList");
  list.innerHTML = "";
  if (!AGG) return;
  const ranked = Object.entries(AGG.player_champion_pool)
    .filter(([, pool]) => pool.games >= 10)
    .map(([pid, pool]) => ({ pid, ...pool }))
    .sort((a, b) => (a.unique - b.unique) || (b.games - a.games))
    .slice(0, 5);
  if (ranked.length === 0) {
    list.innerHTML = "<p>No player pool insight met the thresholds yet.</p>";
    return;
  }
  for (const p of ranked) {
    const topChamps = Object.entries(p.champions)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([id, n]) => `${champLabel(id)} (${n})`)
      .join(", ");
    const team = TEAM_NAME[PLAYER_TEAM[p.pid]] || "";
    const div = document.createElement("div");
    div.className = "insight";
    div.innerHTML =
      `<strong>${PLAYER_NAME[p.pid] || p.pid}${team ? ` (${team})` : ""} played only ${p.unique} champions in ${p.games} games.</strong> ` +
      `Most played: ${topChamps}. Banning two of them leaves this player off their comfort pool.` +
      `<div class="sample">Sample: ${p.games} games, ${p.unique} unique champions.</div>`;
    list.appendChild(div);
  }
}

function renderFearlessStats() {
  const el = document.getElementById("fearlessStats");
  if (!AGG || !AGG.fearless) {
    el.innerHTML = "<p>No Fearless statistics available yet.</p>";
    return;
  }
  const lp = AGG.fearless.locked_per_game;
  const rows = [["2", lp["2"]], ["3", lp["3"]], ["4", lp["4"]], ["5", lp["5"]]]
    .filter(([, v]) => v)
    .map(([g, v]) => `<tr><td>Game ${g}</td><td>${v.avg_locked}</td><td>${v.series} series</td></tr>`)
    .join("");
  el.innerHTML =
    `<p>All ${AGG.fearless.fearless_like_series} multi-game series in the dataset are Fearless-format ` +
    `(no champion picked twice in a series). Champions locked before each game starts:</p>` +
    `<table class="pool-table"><thead><tr><th>Game</th><th>Champions locked</th><th>Series observed</th></tr></thead>` +
    `<tbody>${rows}</tbody></table>` +
    `<p class="hint">Planning implication: by Game 4, both teams have lost 30 champions from their pools — ` +
    `depth wins series, not Game 1 comfort picks.</p>`;
}

// ---------- Wiring ----------

document.getElementById("startSeriesBtn").addEventListener("click", startSeries);
document.getElementById("resetSeriesBtn").addEventListener("click", resetSeries);
document.getElementById("undoBtn").addEventListener("click", undoLast);
document.getElementById("finishGameBtn").addEventListener("click", finishGame);
document.getElementById("champSearch").addEventListener("input", renderChampGrid);
document.getElementById("teamSelect").addEventListener("change", renderTendencies);
document.getElementById("scoutGameSelect").addEventListener("change", renderTendencies);
document.getElementById("patchSelect").addEventListener("change", renderPatchEvent);
document.getElementById("eventSelect").addEventListener("change", renderPatchEvent);

(async () => {
  await loadAll();
  if (AGG) {
    initData();
  } else {
    document.getElementById("dataBadge").textContent =
      "Data not found — run pipeline/extract_lol.py to generate data/aggregates.json, then serve this project from its root folder.";
  }
  renderAll();
})();
