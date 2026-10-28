# JEV-MCP

Local stdio MCP server for the OpenAI Decisions API. One tool, `decide`, accepts shared input and 1–400 caller-defined predicate, choice, or score questions. Returns the native response, including probabilities, confidence, refusals and usage. No actions execute from answers.

Requires Node 22+. Install with `npm ci`, build with `npm run build`, then run `npm start`. Provide OPENAI_API_KEY in the server process environment. For a local .env file, use `node --env-file=.env dist/index.js`. Never pass credentials to the tool.

MCP client configuration: launch `node` with the absolute path to `dist/index.js` as its argument and supply OPENAI_API_KEY through the client's environment configuration. stdout carries MCP only.

The model defaults to gpt-6-luna and accepts only that supported model. Requests time out after 30 seconds, have no retries, and receive the MCP cancellation signal. Serialized request bodies are limited to 1 MiB. Provider error bodies are never returned.

Supports text, user image/JSON content, system/developer text, assistant messages, and prompt-cache settings. Function and custom tool calls and matching outputs are supported. JSON data is validated locally against 32 container levels, 10,000 values and 256 KiB compact UTF-8. Safety identifiers are limited to 128 characters. Cache write limits do not limit supplied breakpoint counts. The reference publishes no cache-key length limit, so none is invented locally. Score questions accept 2–10 levels, following the documented guide.

Sources:
- https://developers.openai.com/api/docs/guides/decisions
- https://developers.openai.com/api/reference/typescript/resources/decisions/methods/create

Run `npm run typecheck`, `npm run build`, and `npm test`. Tests use mocked/local HTTP, in-memory MCP, and a real built stdio subprocess. They do not establish live account access or model availability.
