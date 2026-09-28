// Exact and two-phase search over Kociemba coordinates of the cubie model in rubiks-cube.js.
// Exact search is iterative-deepening A* with admissible pattern tables; it proves distances
// within its node budget. Two-phase search solves any position in about twenty turns but does
// not prove optimality. Tables build once, in slices, so a page can keep drawing meanwhile.
const CubeSolver = (() => {
  const MOVE = Cube.MOVE_CUBIES;
  const FACE_OF = Uint8Array.from({length: 18}, (_, m) => m / 3 | 0);
  // Phase two keeps the cube in <U, D, R2, L2, F2, B2>.
  const PHASE2 = Uint8Array.of(0, 1, 2, 4, 7, 9, 10, 11, 13, 16);
  const IN_PHASE2 = new Uint8Array(18);
  for (const m of PHASE2) IN_PHASE2[m] = 1;
  const ALL = Uint8Array.from({length: 18}, (_, m) => m);
  const N_TWIST = 2187, N_FLIP = 2048, N_SLICE = 495, N_EDGE4 = 11880, N_PERM8 = 40320, N_PERM4 = 24;

  const binom = (n, k) => { if (k > n) return 0; let r = 1; for (let i = 0; i < k; i++) r = r * (n - i) / (i + 1); return Math.round(r); };
  const CHOOSE = Array.from({length: 12}, (_, n) => Array.from({length: 5}, (_, k) => binom(n, k)));

  // Lehmer rank of values[offset .. offset+n); only their relative order matters.
  function rankPerm(values, offset, n) {
    let rank = 0;
    for (let i = 0; i < n; i++) {
      let smaller = 0;
      for (let j = i + 1; j < n; j++) if (values[offset + j] < values[offset + i]) smaller++;
      rank = rank * (n - i) + smaller;
    }
    return rank;
  }
  const digits = new Uint8Array(12), pool = new Uint8Array(12);
  function unrankInto(rank, n, out) {
    for (let i = n - 1; i >= 0; i--) { digits[i] = rank % (n - i); rank = Math.floor(rank / (n - i)); }
    for (let i = 0; i < n; i++) pool[i] = i;
    for (let i = 0; i < n; i++) {
      out[i] = pool[digits[i]];
      for (let j = digits[i]; j < n - 1 - i; j++) pool[j] = pool[j + 1];
    }
    return out;
  }
  const twistOf = co => { let t = 0; for (let i = 0; i < 7; i++) t = 3 * t + co[i]; return t; };
  const flipOf = eo => { let f = 0; for (let i = 0; i < 11; i++) f = 2 * f + eo[i]; return f; };
  function twistInto(t, co) {
    let sum = 0;
    for (let i = 6; i >= 0; i--) { co[i] = t % 3; sum += co[i]; t = t / 3 | 0; }
    co[7] = (3 - sum % 3) % 3;
    return co;
  }
  function flipInto(f, eo) {
    let sum = 0;
    for (let i = 10; i >= 0; i--) { eo[i] = f & 1; sum += eo[i]; f >>= 1; }
    eo[11] = sum & 1;
    return eo;
  }
  // Positions and order of the four edges base..base+3: combination of positions times 4!.
  const order = new Uint8Array(4);
  function edge4Of(ep, base) {
    let comb = 0, k = 0;
    for (let pos = 0; pos < 12; pos++) {
      const t = ep[pos] - base;
      if (t >= 0 && t < 4) { comb += CHOOSE[pos][k + 1]; order[k++] = t; }
    }
    return comb * N_PERM4 + rankPerm(order, 0, 4);
  }
  function edge4Into(index, ep) {
    unrankInto(index % N_PERM4, 4, order);
    ep.fill(255);
    let comb = Math.floor(index / N_PERM4), k = 4;
    for (let pos = 11; pos >= 0 && k > 0; pos--) if (CHOOSE[pos][k] <= comb) { comb -= CHOOSE[pos][k]; k--; ep[pos] = order[k]; }
    return ep;
  }
  function multiply(a, b) {
    const out = {cp: new Uint8Array(8), co: new Uint8Array(8), ep: new Uint8Array(12), eo: new Uint8Array(12)};
    for (let i = 0; i < 8; i++) { out.cp[i] = a.cp[b.cp[i]]; out.co[i] = (a.co[b.cp[i]] + b.co[i]) % 3; }
    for (let i = 0; i < 12; i++) { out.ep[i] = a.ep[b.ep[i]]; out.eo[i] = (a.eo[b.ep[i]] + b.eo[i]) & 1; }
    return out;
  }
  const IDENTITY = {cp: Uint8Array.from({length: 8}, (_, i) => i), co: new Uint8Array(8), ep: Uint8Array.from({length: 12}, (_, i) => i), eo: new Uint8Array(12)};
  const SLICE_HOME = edge4Of(IDENTITY.ep, 8), U_HOME = edge4Of(IDENTITY.ep, 0), D_HOME = edge4Of(IDENTITY.ep, 4);
  const SLICE_HOME_POS = SLICE_HOME / N_PERM4 | 0;

  let TWIST_MOVE, FLIP_MOVE, CORNER_MOVE, EDGE4_MOVE, SLICE_MOVE, UD_MOVE, SLICEPERM_MOVE;
  let TWIST_SLICE, FLIP_SLICE, CORNER_DIST, SLICE_DIST, U_DIST, D_DIST, P2_CORNER, P2_EDGE;
  const built = new Set();

  // Decodes each coordinate once and applies only the generator turns; a half turn is then the
  // quarter turn twice and a counterclockwise turn is it three times.
  const QUARTERS = Uint8Array.of(0, 3, 6, 9, 12, 15), PHASE2_GENERATORS = Uint8Array.of(0, 9, 4, 7, 13, 16);
  function* moveTable(size, generators, load, step) {
    const out = new Uint16Array(size * 18);
    for (let i = 0; i < size; i++) {
      load(i);
      for (let k = 0; k < generators.length; k++) out[i * 18 + generators[k]] = step(MOVE[generators[k]]);
      if ((i & 511) === 511) yield;
    }
    for (let i = 0; i < size; i++) {
      for (let k = 0; k < generators.length; k++) {
        const g = generators[k];
        if (g % 3) continue;
        const twice = out[out[i * 18 + g] * 18 + g];
        out[i * 18 + g + 1] = twice;
        out[i * 18 + g + 2] = out[twice * 18 + g];
      }
      if ((i & 4095) === 4095) yield;
    }
    return out;
  }
  // Breadth-first distances over a pair of coordinates. Late layers scan the unvisited entries
  // backward, which is valid because every move set here is closed under inverses.
  function* pairTable(nA, moveA, nB, moveB, goal, moves) {
    const size = nA * nB, table = new Uint8Array(size).fill(255);
    table[goal] = 0;
    let depth = 0, done = 1;
    while (done < size) {
      let found = 0;
      const backward = done > size / 2;
      for (let a = 0, i = 0; a < nA; a++) {
        const rowA = a * 18;
        for (let b = 0; b < nB; b++, i++) {
          const value = table[i];
          if (backward ? value !== 255 : value !== depth) continue;
          for (let k = 0; k < moves.length; k++) {
            const m = moves[k], j = moveA[rowA + m] * nB + (nB > 1 ? moveB[b * 18 + m] : 0);
            if (backward) { if (table[j] === depth) { table[i] = depth + 1; found++; break; } }
            else if (table[j] === 255) { table[j] = depth + 1; found++; }
          }
        }
        if ((a & 63) === 63) yield;
      }
      if (!found) break;
      done += found;
      depth++;
    }
    return table;
  }

  // Exact search needs the full-group tables; two-phase adds its phase-two tables.
  const s8 = new Uint8Array(8), n8 = new Uint8Array(8), s12 = new Uint8Array(12), n12 = new Uint8Array(12), s4 = new Uint8Array(4), n4 = new Uint8Array(4);
  function* buildExact() {
    if (built.has('exact')) return;
    TWIST_MOVE = yield* moveTable(N_TWIST, QUARTERS, t => twistInto(t, s8), M => {
      for (let i = 0; i < 8; i++) n8[i] = (s8[M.cp[i]] + M.co[i]) % 3;
      return twistOf(n8);
    });
    FLIP_MOVE = yield* moveTable(N_FLIP, QUARTERS, f => flipInto(f, s12), M => {
      for (let i = 0; i < 12; i++) n12[i] = (s12[M.ep[i]] + M.eo[i]) & 1;
      return flipOf(n12);
    });
    CORNER_MOVE = yield* moveTable(N_PERM8, QUARTERS, c => unrankInto(c, 8, s8), M => {
      for (let i = 0; i < 8; i++) n8[i] = s8[M.cp[i]];
      return rankPerm(n8, 0, 8);
    });
    // The transition of a tracked-edge coordinate depends only on positions, so one table
    // serves the slice, up and down edge quartets.
    EDGE4_MOVE = yield* moveTable(N_EDGE4, QUARTERS, e => edge4Into(e, s12), M => {
      for (let i = 0; i < 12; i++) n12[i] = s12[M.ep[i]];
      return edge4Of(n12, 0);
    });
    SLICE_MOVE = new Uint16Array(N_SLICE * 18);
    for (let s = 0; s < N_SLICE; s++) for (let m = 0; m < 18; m++) SLICE_MOVE[s * 18 + m] = EDGE4_MOVE[s * N_PERM4 * 18 + m] / N_PERM4 | 0;
    TWIST_SLICE = yield* pairTable(N_TWIST, TWIST_MOVE, N_SLICE, SLICE_MOVE, SLICE_HOME_POS, ALL);
    FLIP_SLICE = yield* pairTable(N_FLIP, FLIP_MOVE, N_SLICE, SLICE_MOVE, SLICE_HOME_POS, ALL);
    CORNER_DIST = yield* pairTable(N_PERM8, CORNER_MOVE, 1, null, 0, ALL);
    SLICE_DIST = yield* pairTable(N_EDGE4, EDGE4_MOVE, 1, null, SLICE_HOME, ALL);
    U_DIST = yield* pairTable(N_EDGE4, EDGE4_MOVE, 1, null, U_HOME, ALL);
    D_DIST = yield* pairTable(N_EDGE4, EDGE4_MOVE, 1, null, D_HOME, ALL);
    built.add('exact');
  }
  function* buildTwoPhase() {
    if (built.has('two-phase')) return;
    UD_MOVE = yield* moveTable(N_PERM8, PHASE2_GENERATORS, u => unrankInto(u, 8, s8), M => {
      for (let i = 0; i < 8; i++) n8[i] = s8[M.ep[i]];
      return rankPerm(n8, 0, 8);
    });
    SLICEPERM_MOVE = yield* moveTable(N_PERM4, PHASE2_GENERATORS, s => unrankInto(s, 4, s4), M => {
      for (let i = 0; i < 4; i++) n4[i] = s4[M.ep[8 + i] - 8];
      return rankPerm(n4, 0, 4);
    });
    P2_CORNER = yield* pairTable(N_PERM8, CORNER_MOVE, N_PERM4, SLICEPERM_MOVE, 0, PHASE2);
    P2_EDGE = yield* pairTable(N_PERM8, UD_MOVE, N_PERM4, SLICEPERM_MOVE, 0, PHASE2);
    built.add('two-phase');
  }
  // One builder, exact tables first. Synchronous and sliced callers drive the same sequence, so
  // a table is never built twice.
  let steps = null;
  function* buildAll() { yield* buildExact(); yield* buildTwoPhase(); }
  function advance(stage, until) {
    steps ??= buildAll();
    while (!built.has(stage) && performance.now() < until) if (steps.next().done) break;
  }
  function prepare(stage = 'exact') { advance(stage, Infinity); }
  // Builds in slices of about sliceMs, yielding to the host between slices.
  async function prepareAsync(stage = 'exact', {sliceMs = 8, pause = () => new Promise(resolve => setTimeout(resolve, 0))} = {}) {
    for (;;) {
      advance(stage, performance.now() + sliceMs);
      if (built.has(stage)) return;
      await pause();
    }
  }
  const ready = (stage = 'exact') => built.has(stage);

  function coordinates(cube) {
    return {twist: twistOf(cube.co), flip: flipOf(cube.eo), corners: rankPerm(cube.cp, 0, 8),
      slice: edge4Of(cube.ep, 8), up: edge4Of(cube.ep, 0), down: edge4Of(cube.ep, 4)};
  }
  // Admissible: each table is an exact distance for a projection of the cube.
  function bound(tw, fl, co, sl, up, dn) {
    const pos = sl / N_PERM4 | 0;
    let h = TWIST_SLICE[tw * N_SLICE + pos], v = FLIP_SLICE[fl * N_SLICE + pos];
    if (v > h) h = v;
    v = CORNER_DIST[co]; if (v > h) h = v;
    v = SLICE_DIST[sl]; if (v > h) h = v;
    v = U_DIST[up]; if (v > h) h = v;
    v = D_DIST[dn]; if (v > h) h = v;
    return h;
  }

  // Iterative-deepening search over canonical sequences: never the same face twice in a row,
  // and commuting opposite faces only in U-before-D order.
  const path = new Uint8Array(32);
  let nodes = 0, limit = 0, stopped = false;
  function search(tw, fl, co, sl, up, dn, depth, togo, last) {
    if (togo === 0) return tw === 0 && fl === 0 && co === 0 && sl === SLICE_HOME && up === U_HOME && dn === D_HOME;
    for (let m = 0; m < 18; m++) {
      const face = FACE_OF[m];
      if (face === last || face === last - 3) continue;
      const tw2 = TWIST_MOVE[tw * 18 + m], sl2 = EDGE4_MOVE[sl * 18 + m], pos = sl2 / N_PERM4 | 0;
      if (TWIST_SLICE[tw2 * N_SLICE + pos] >= togo) continue;
      const fl2 = FLIP_MOVE[fl * 18 + m];
      if (FLIP_SLICE[fl2 * N_SLICE + pos] >= togo) continue;
      const co2 = CORNER_MOVE[co * 18 + m];
      if (CORNER_DIST[co2] >= togo || SLICE_DIST[sl2] >= togo) continue;
      const up2 = EDGE4_MOVE[up * 18 + m];
      if (U_DIST[up2] >= togo) continue;
      const dn2 = EDGE4_MOVE[dn * 18 + m];
      if (D_DIST[dn2] >= togo) continue;
      if (++nodes > limit) { stopped = true; return false; }
      path[depth] = m;
      if (search(tw2, fl2, co2, sl2, up2, dn2, depth + 1, togo - 1, face)) return true;
      if (stopped) return false;
    }
    return false;
  }
  // Exact distance and one optimal line, or bounds when the budget or depth cap stops it.
  // `atLeast` skips thresholds the caller has already proven impossible.
  function optimal(cube, {maxDepth = 20, budget = 4e6, atLeast = 0, exactly = null} = {}) {
    if (!ready('exact')) prepare('exact');
    const c = coordinates(cube), h = bound(c.twist, c.flip, c.corners, c.slice, c.up, c.down);
    nodes = 0; limit = budget; stopped = false;
    const first = exactly ?? Math.max(h, atLeast), last = exactly ?? maxDepth;
    for (let depth = first; depth <= last; depth++) {
      if (search(c.twist, c.flip, c.corners, c.slice, c.up, c.down, 0, depth, -1)) {
        return {distance: depth, moves: Array.from(path.subarray(0, depth), m => Cube.MOVES[m].id), lower: depth, nodes};
      }
      if (stopped) return {distance: null, moves: null, lower: depth, nodes, exhausted: true};
    }
    return {distance: null, moves: null, lower: last + 1, nodes, exhausted: false};
  }
  // Distance after each of the eighteen turns from a position at a known distance d. Every
  // neighbor is d - 1, d or d + 1 away, so at most two fixed-depth searches decide each one.
  function analyze(cube, distance, {budget = 2e6} = {}) {
    const result = [];
    for (let m = 0; m < 18; m++) {
      const next = multiply(cube, MOVE[m]);
      let after = null;
      if (distance === 0) after = 1;
      else {
        const closer = optimal(next, {exactly: distance - 1, budget});
        if (closer.distance != null) after = distance - 1;
        else if (!closer.exhausted) {
          const level = optimal(next, {exactly: distance, budget});
          after = level.distance != null ? distance : level.exhausted ? null : distance + 1;
        }
      }
      result.push({id: Cube.MOVES[m].id, distance: after, delta: after == null ? null : after - distance});
    }
    return result;
  }

  // Kociemba two-phase search: reach <U, D, R2, L2, F2, B2>, then solve inside it. It keeps
  // lengthening phase one until it reaches the target length or spends its node budget, and
  // returns the shortest total found. A node budget, not a clock, keeps results reproducible.
  function twoPhase(cube, {target = 20, budget = 1e6, maxLength = 30} = {}) {
    if (!ready('two-phase')) prepare('two-phase');
    const started = performance.now(), p1 = new Uint8Array(32), p2 = new Uint8Array(32);
    let best = null, bestLength = maxLength + 1, spent = 0, done = false;
    const tw0 = twistOf(cube.co), fl0 = flipOf(cube.eo), sl0 = edge4Of(cube.ep, 8) / N_PERM4 | 0;
    const exhausted = () => (best && spent > budget) ? (done = true) : done;
    function phase2(co, ud, sp, depth, togo, last) {
      if (togo === 0) return co === 0 && ud === 0 && sp === 0;
      for (let k = 0; k < PHASE2.length; k++) {
        const m = PHASE2[k], face = FACE_OF[m];
        if (face === last || face === last - 3) continue;
        const co2 = CORNER_MOVE[co * 18 + m], sp2 = SLICEPERM_MOVE[sp * 18 + m];
        if (P2_CORNER[co2 * N_PERM4 + sp2] >= togo) continue;
        const ud2 = UD_MOVE[ud * 18 + m];
        if (P2_EDGE[ud2 * N_PERM4 + sp2] >= togo) continue;
        spent++;
        p2[depth] = m;
        if (phase2(co2, ud2, sp2, depth + 1, togo - 1, face)) return true;
      }
      return false;
    }
    function finish(n1) {
      // A phase one ending in a phase-two turn repeats a shorter phase one.
      if (n1 > 0 && IN_PHASE2[p1[n1 - 1]]) return false;
      let c = cube;
      for (let i = 0; i < n1; i++) c = multiply(c, MOVE[p1[i]]);
      const co = rankPerm(c.cp, 0, 8), ud = rankPerm(c.ep, 0, 8), sp = rankPerm(c.ep, 8, 4);
      const h = Math.max(P2_CORNER[co * N_PERM4 + sp], P2_EDGE[ud * N_PERM4 + sp]);
      for (let n2 = h; n1 + n2 < bestLength; n2++) {
        if (phase2(co, ud, sp, 0, n2, n1 ? FACE_OF[p1[n1 - 1]] : -1)) {
          bestLength = n1 + n2;
          best = [...p1.subarray(0, n1), ...p2.subarray(0, n2)];
          if (bestLength <= target) done = true;
          return done;
        }
      }
      return false;
    }
    function phase1(tw, fl, sl, depth, togo, last) {
      if (togo === 0) return tw === 0 && fl === 0 && sl === SLICE_HOME_POS && finish(depth);
      for (let m = 0; m < 18; m++) {
        const face = FACE_OF[m];
        if (face === last || face === last - 3) continue;
        const tw2 = TWIST_MOVE[tw * 18 + m], sl2 = SLICE_MOVE[sl * 18 + m];
        if (TWIST_SLICE[tw2 * N_SLICE + sl2] >= togo) continue;
        const fl2 = FLIP_MOVE[fl * 18 + m];
        if (FLIP_SLICE[fl2 * N_SLICE + sl2] >= togo) continue;
        spent++;
        p1[depth] = m;
        if (phase1(tw2, fl2, sl2, depth + 1, togo - 1, face) || exhausted()) return true;
      }
      return false;
    }
    const h1 = Math.max(TWIST_SLICE[tw0 * N_SLICE + sl0], FLIP_SLICE[fl0 * N_SLICE + sl0]);
    for (let n1 = h1; n1 < bestLength && !done; n1++) phase1(tw0, fl0, sl0, 0, n1, -1);
    return {moves: best.map(m => Cube.MOVES[m].id), length: best.length, nodes: spent, ms: performance.now() - started};
  }

  // A uniformly random reachable cube: any corner and edge arrangement of equal parity, any
  // twist and any flip.
  function randomCube(random) {
    const shuffle = n => {
      const a = Array.from({length: n}, (_, i) => i);
      for (let i = n - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
      return a;
    };
    const parity = p => { let s = 0; for (let i = 0; i < p.length; i++) for (let j = i + 1; j < p.length; j++) if (p[j] < p[i]) s ^= 1; return s; };
    const cp = shuffle(8), ep = shuffle(12);
    if (parity(cp) !== parity(ep)) [ep[10], ep[11]] = [ep[11], ep[10]];
    return {cp: Uint8Array.from(cp), co: twistInto(Math.floor(random() * N_TWIST), new Uint8Array(8)), ep: Uint8Array.from(ep), eo: flipInto(Math.floor(random() * N_FLIP), new Uint8Array(12))};
  }
  const lowerBound = cube => { if (!ready('exact')) prepare('exact'); const c = coordinates(cube); return bound(c.twist, c.flip, c.corners, c.slice, c.up, c.down); };

  return {prepare, prepareAsync, ready, optimal, analyze, twoPhase, randomCube, lowerBound, multiply, IDENTITY};
})();
