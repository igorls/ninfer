# Chess rules dependency

`chess.js` vendors the CommonJS distribution from **chess.js 1.4.0**, licensed under
BSD-2-Clause; see [chess.LICENSE](chess.LICENSE).

- Upstream: <https://github.com/jhlywa/chess.js>
- API documentation: <https://jhlywa.github.io/chess.js/>
- Package: <https://registry.npmjs.org/chess.js/-/chess.js-1.4.0.tgz>
- Verified package SHA-512 (base64):
  `BBJgrrtKQOzFLonR0l+k64A98NLemPwNsCskwb+29bRwobUa4iTm51E1kwGPbWXAcfdDa18nad6vpPPKPWarqw==`

The distribution is wrapped in a local `ChessRules` closure for classic browser scripts, and its
unavailable source-map comment is removed. The rules implementation is otherwise unchanged. No
runtime CDN or package installation is required. The library supplies move legality and game
end conditions; it supplies no computer opponent or search engine.

## Three.js

`three.min.js` vendors the UMD build of **Three.js r159**, licensed under MIT. Copyright 2010–2023
Three.js Authors. The startup deprecation warning was removed so the cube page does not print it;
the library body is otherwise unchanged. Cube and Arena Duel use it for their WebGL views.
Game rules, decisions and evaluation do not depend on it.
