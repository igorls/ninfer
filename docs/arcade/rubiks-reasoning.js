// Reasoning-only comparison: the same observation and candidate policy as System One.
// No simulator, scramble, solver grades, or hidden solution enters this request.
const RubiksReasoning = (() => {
  const DEFAULT_MAX_TOKENS = 4096;
  function requestFor(game, model, {maxTokens = DEFAULT_MAX_TOKENS, avoidRepeats = true} = {}) {
    if (!Number.isInteger(maxTokens) || maxTokens < 128 || maxTokens > 32768) throw new Error('Output budget must be a whole number from 128 to 32768.');
    const {body: decision, options, excluded} = Rubiks.requestFor(game, model, {avoidRepeats});
    const question = decision.questions.move;
    const prompt = decision.state + '\n\n' + question.instructions + '\n' +
      Object.entries(question.criteria).map(([code, description]) => code + ': ' + description).join('\n') +
      '\nReturn JSON with the selected option token in the move field.';
    return {options, excluded, body: {model,
      messages: [{role: 'system', content: 'You solve Rubik cubes. Work out a short plan before selecting the next turn. The number of misplaced stickers can increase on a correct route to solved.'}, {role: 'user', content: prompt}],
      enable_thinking: true, reasoning_effort: 'low', max_tokens: maxTokens, temperature: 0, seed: game.seed,
      response_format: {type: 'json_schema', json_schema: {name: 'cube_turn', strict: true,
        schema: {type: 'object', properties: {move: {type: 'string', enum: options.map(option => option.code)}}, required: ['move'], additionalProperties: false}}}}};
  }
  function usageOf(body) {
    const usage = body?.usage;
    if (!Number.isInteger(usage?.prompt_tokens) || usage.prompt_tokens < 0 || !Number.isInteger(usage?.completion_tokens) || usage.completion_tokens < 0)
      throw new Error('Unexpected reasoning token usage.');
    return {input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens,
      reasoning_tokens: usage.completion_tokens_details?.reasoning_tokens ?? null};
  }
  function decode(body, options) {
    const choice = body?.choices?.[0];
    if (choice?.finish_reason !== 'stop') throw new Error(choice?.finish_reason === 'length'
      ? 'Reasoning reached the output-token limit before completing an answer. Increase the output budget or retry.'
      : 'Reasoning did not finish with a complete answer.');
    let answer;
    try { answer = JSON.parse(choice.message.content); } catch { throw new Error('Reasoning returned an invalid JSON answer.'); }
    const option = options.find(option => option.code === answer?.move);
    if (!option || Object.keys(answer).length !== 1) throw new Error('Reasoning returned an unknown action or answer shape.');
    return {option, usage: usageOf(body), probabilities: null, choiceWarning: null};
  }
  async function choose(game, connection, {signal, fetchImpl = fetch, now = () => performance.now()} = {}) {
    const {options, excluded, body: request} = requestFor(game, connection.model, connection);
    const headers = {'Content-Type': 'application/json'};
    if (connection.apiKey) headers.Authorization = 'Bearer ' + connection.apiKey;
    const started = now();
    let response, body, raw;
    const fail = (message, kind) => Object.assign(new Error(message), {kind, status: response?.status ?? null,
      receipt: {request, response: body ?? raw ?? null, ms: now() - started}});
    try {
      response = await fetchImpl(connection.endpoint.replace(/\/+$/, '') + '/v1/chat/completions', {method: 'POST', headers, body: JSON.stringify(request), signal});
      raw = await response.text();
    } catch (error) {
      if (signal?.aborted || error.name === 'AbortError') throw error;
      throw fail('Could not complete the reasoning request.', 'network');
    }
    try { body = JSON.parse(raw); } catch { throw fail('The reasoning response is not valid JSON.', 'response'); }
    if (!response.ok) throw fail('HTTP ' + response.status + ': ' + (body?.error?.message || raw.slice(0, 200)), 'http');
    try { return {...decode(body, options), ms: now() - started, request, response: body, options, excluded}; }
    catch (error) { throw fail(error.message, 'response'); }
  }
  return {DEFAULT_MAX_TOKENS, requestFor, decode, choose};
})();
