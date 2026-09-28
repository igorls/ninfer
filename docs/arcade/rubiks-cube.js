// Cube geometry and standard face-turn notation. The facelet permutations, the solver's cubie
// tables and the renderer's layer rotations all derive from one rule: a clockwise face turn is a
// right-handed rotation of -90 degrees about that face's outward normal.
const Cube = (() => {
  const FACES = ['U', 'R', 'F', 'D', 'L', 'B'];
  const FACE_NAME = {U: 'up', R: 'right', F: 'front', D: 'down', L: 'left', B: 'back'};
  // Sticker letters by home face, in the Western scheme: white up, green front, red right.
  const LETTERS = ['W', 'R', 'G', 'Y', 'O', 'B'];
  const COLOR_NAME = ['white', 'red', 'green', 'yellow', 'orange', 'blue'];
  // Outward normal, plus the right and up directions of a face read while looking at it. The up
  // face is read with its back edge on top and the down face with its front edge on top.
  const FRAME = {
    U: {n: [0, 1, 0], right: [1, 0, 0], up: [0, 0, -1]},
    R: {n: [1, 0, 0], right: [0, 0, -1], up: [0, 1, 0]},
    F: {n: [0, 0, 1], right: [1, 0, 0], up: [0, 1, 0]},
    D: {n: [0, -1, 0], right: [1, 0, 0], up: [0, 0, 1]},
    L: {n: [-1, 0, 0], right: [0, 0, 1], up: [0, 1, 0]},
    B: {n: [0, 0, -1], right: [-1, 0, 0], up: [0, 1, 0]}
  };
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  // Facelet i = 9 * face + 3 * row + col, faces in the order U R F D L B. This is the standard
  // facelet order of Kociemba's two-phase solver, so a state reads as the usual 54-letter string.
  const FACELETS = [];
  for (let face = 0; face < 6; face++) {
    const {n, right, up} = FRAME[FACES[face]];
    for (let row = 0; row < 3; row++) for (let col = 0; col < 3; col++) {
      FACELETS.push({index: FACELETS.length, face, row, col, n: n.slice(),
        pos: [0, 1, 2].map(k => n[k] + right[k] * (col - 1) + up[k] * (1 - row))});
    }
  }
  const faceletAt = (pos, n) => FACELETS.findIndex(f => f.n.every((v, k) => v === n[k]) && f.pos.every((v, k) => v === pos[k]));

  // Right-handed quarter turns about a unit axis: one step maps v to a(a.v) + a x v.
  function quarter(v, a, times) {
    let out = v.slice();
    for (let i = 0; i < ((times % 4) + 4) % 4; i++) {
      const along = dot(a, out);
      out = [a[0] * along + a[1] * out[2] - a[2] * out[1], a[1] * along + a[2] * out[0] - a[0] * out[2], a[2] * along + a[0] * out[1] - a[1] * out[0]];
    }
    return out;
  }

  // Move index m = 3 * face + k: k = 0 clockwise quarter, 1 half, 2 counterclockwise quarter.
  // `turn` is the signed quarter count the renderer animates: 1, 2 or -1 clockwise quarters.
  const SUFFIX = ['', '2', "'"];
  const MOVES = [];
  for (let face = 0; face < 6; face++) for (let k = 0; k < 3; k++) {
    const quarters = k + 1, turn = k === 2 ? -1 : quarters, n = FRAME[FACES[face]].n;
    const perm = new Uint8Array(54);
    for (const source of FACELETS) {
      if (dot(source.pos, n) !== 1) { perm[source.index] = source.index; continue; }
      perm[faceletAt(quarter(source.pos, n, -quarters), quarter(source.n, n, -quarters))] = source.index;
    }
    MOVES.push({index: MOVES.length, id: FACES[face] + SUFFIX[k], face: FACES[face], faceIndex: face, quarters, turn,
      axis: n.slice(), angle: -turn * Math.PI / 2, perm});
  }
  const MOVE_BY_ID = new Map(MOVES.map(move => [move.id, move]));
  const move = id => {
    const found = MOVE_BY_ID.get(id);
    if (!found) throw new Error('Unknown face turn ' + id + '.');
    return found;
  };
  const SOLVED = Uint8Array.from(FACELETS, f => f.face);

  // Every sticker moves with its layer: the sticker that lands on facelet i came from perm[i].
  function apply(colors, turn) {
    const next = new Uint8Array(54);
    for (let i = 0; i < 54; i++) next[i] = colors[turn.perm[i]];
    return next;
  }
  const applyAll = (colors, ids) => ids.reduce((state, id) => apply(state, move(id)), colors);
  function solved(colors) {
    for (let i = 0; i < 54; i++) if (colors[i] !== SOLVED[i]) return false;
    return true;
  }
  // Centers never move, so only the 48 outer stickers can be misplaced.
  function misplaced(colors) {
    let count = 0;
    for (let i = 0; i < 54; i++) if (colors[i] !== SOLVED[i]) count++;
    return count;
  }
  const faceText = (colors, face) => Array.from(colors.subarray(9 * FACES.indexOf(face), 9 * FACES.indexOf(face) + 9), c => LETTERS[c]).join('');
  const faceletString = colors => Array.from(colors, c => FACES[c]).join('');
  const inverse = id => id.endsWith('2') ? id : id.endsWith("'") ? id[0] : id + "'";
  const invert = ids => ids.slice().reverse().map(inverse);

  // Cubie view. Corners URF UFL ULB UBR DFR DLF DBL DRB and edges UR UF UL UB DR DF DL DB FR FL BL BR,
  // each listed by facelet with its U or D sticker first; corner facelets run clockwise.
  const CORNER_FACELETS = [[8, 9, 20], [6, 18, 38], [0, 36, 47], [2, 45, 11], [29, 26, 15], [27, 44, 24], [33, 53, 42], [35, 17, 51]];
  const EDGE_FACELETS = [[5, 10], [7, 19], [3, 37], [1, 46], [32, 16], [28, 25], [30, 43], [34, 52], [23, 12], [21, 41], [50, 39], [48, 14]];
  const CORNER_COLORS = CORNER_FACELETS.map(list => list.map(i => SOLVED[i]));
  const EDGE_COLORS = EDGE_FACELETS.map(list => list.map(i => SOLVED[i]));

  function cubieOf(colors) {
    const cube = {cp: new Uint8Array(8), co: new Uint8Array(8), ep: new Uint8Array(12), eo: new Uint8Array(12)};
    const cornerSeen = new Set(), edgeSeen = new Set();
    for (let i = 0; i < 8; i++) {
      const f = CORNER_FACELETS[i];
      let ori = 0;
      while (ori < 3 && colors[f[ori]] !== 0 && colors[f[ori]] !== 3) ori++;
      const a = colors[f[(ori + 1) % 3]], b = colors[f[(ori + 2) % 3]];
      const piece = CORNER_COLORS.findIndex(c => c[1] === a && c[2] === b);
      if (ori === 3 || piece < 0 || cornerSeen.has(piece)) throw new Error('The stickers do not form a cube.');
      cornerSeen.add(piece);
      cube.cp[i] = piece; cube.co[i] = ori;
    }
    for (let i = 0; i < 12; i++) {
      const [x, y] = EDGE_FACELETS[i].map(k => colors[k]);
      let piece = EDGE_COLORS.findIndex(c => c[0] === x && c[1] === y), ori = 0;
      if (piece < 0) { piece = EDGE_COLORS.findIndex(c => c[0] === y && c[1] === x); ori = 1; }
      if (piece < 0 || edgeSeen.has(piece)) throw new Error('The stickers do not form a cube.');
      edgeSeen.add(piece);
      cube.ep[i] = piece; cube.eo[i] = ori;
    }
    return cube;
  }
  function colorsOf(cube) {
    const colors = SOLVED.slice();
    for (let i = 0; i < 8; i++) for (let n = 0; n < 3; n++) colors[CORNER_FACELETS[i][(n + cube.co[i]) % 3]] = CORNER_COLORS[cube.cp[i]][n];
    for (let i = 0; i < 12; i++) for (let n = 0; n < 2; n++) colors[EDGE_FACELETS[i][(n + cube.eo[i]) % 2]] = EDGE_COLORS[cube.ep[i]][n];
    return colors;
  }
  // The eighteen turns in cubie form, read back from their facelet permutations.
  const MOVE_CUBIES = MOVES.map(turn => cubieOf(apply(SOLVED, turn)));

  return {FACES, FACE_NAME, LETTERS, COLOR_NAME, FRAME, FACELETS, MOVES, MOVE_BY_ID, move, SOLVED, apply, applyAll,
    solved, misplaced, faceText, faceletString, inverse, invert, cubieOf, colorsOf, MOVE_CUBIES};
})();
