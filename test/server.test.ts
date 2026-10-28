import { test } from "node:test";
import assert from "node:assert/strict";
import OpenAI from "openai";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createServer, requestSchema } from "../src/server.js";

test("rejects invalid batches and choices", () => {
  for (const questions of [
    [],
    Array(401).fill({ type: "predicate", instructions: "x" }),
    [
      {
        type: "choice",
        instructions: "x",
        choices: [{ value: true }, { value: true }],
      },
    ],
  ])
    assert.equal(
      requestSchema.safeParse({ input: "x", questions }).success,
      false,
    );
  assert.equal(
    requestSchema.safeParse({
      input: "x",
      questions: [{ type: "predicate", instructions: "x" }],
      api_key: "secret",
    }).success,
    false,
  );
});
test("MCP initialization and native batch result through mocked HTTP", async () => {
  const answers = [
    { type: "predicate", name: null, probability: 0.7 },
    { type: "refusal", name: "second" },
  ];
  const native = { model: "gpt-6-luna", answers, usage: { total_tokens: 42 } };
  const provider = new OpenAI({
    apiKey: "test-only",
    fetch: async (url, init) => {
      assert.match(String(url), /\/v1\/decisions$/);
      const body = JSON.parse(String(init?.body)) as { questions: unknown[] };
      assert.equal(body.questions.length, 2);
      return new Response(JSON.stringify(native), {
        headers: { "content-type": "application/json" },
      });
    },
  });
  const server = createServer(provider),
    client = new Client({ name: "test", version: "1" });
  const [a, b] = InMemoryTransport.createLinkedPair();
  await server.connect(a);
  await client.connect(b);
  try {
    assert.equal((await client.listTools()).tools.length, 1);
    const result = await client.callTool({
      name: "decide",
      arguments: {
        input: "caller input",
        questions: [
          { type: "predicate", instructions: "caller question" },
          { type: "predicate", name: "second", instructions: "another" },
        ],
      },
    });
    assert.deepEqual(result.content, [
      { type: "text", text: JSON.stringify(native) },
    ]);
  } finally {
    await client.close();
    await server.close();
  }
});
test("provider errors do not expose credentials or response bodies", async () => {
  const server = createServer(
    new OpenAI({
      apiKey: "private-token",
      fetch: async () => new Response("private-token", { status: 401 }),
    }),
  );
  const client = new Client({ name: "test", version: "1" });
  const [a, b] = InMemoryTransport.createLinkedPair();
  await server.connect(a);
  await client.connect(b);
  try {
    const result = await client.callTool({
      name: "decide",
      arguments: {
        input: "x",
        questions: [{ type: "predicate", instructions: "x" }],
      },
    });
    assert.equal(result.isError, true);
    assert.doesNotMatch(JSON.stringify(result), /private-token/);
  } finally {
    await client.close();
    await server.close();
  }
});

test("tool evidence preserves all upstream fields and matches output IDs", () => {
  const input = [
    { role: "user", content: "question" },
    {
      type: "function_call",
      call_id: "a",
      arguments: "{}",
      name: "lookup",
      namespace: null,
      status: "completed",
      id: "f",
    },
    {
      type: "function_call_output",
      call_id: "a",
      output: [{ type: "input_text", text: "result" }],
    },
    { type: "custom_tool_call", call_id: "b", name: "custom", input: "raw" },
    { type: "custom_tool_call_output", call_id: "b", output: "result" },
  ];
  const body = {
    input,
    questions: [{ type: "predicate", instructions: "evaluate" }],
  };
  assert.deepEqual(requestSchema.parse(body).input, input);
  assert.equal(
    requestSchema.safeParse({ ...body, input: [input[2]] }).success,
    false,
  );
});
test("question parts exclude cache breakpoints while evidence accepts many", () => {
  const part = {
    type: "input_text",
    text: "x",
    prompt_cache_breakpoint: { mode: "explicit" },
  };
  const questions = [{ type: "predicate", instructions: "x" }];
  assert.equal(
    requestSchema.safeParse({
      input: [{ role: "user", content: Array(10).fill(part) }],
      questions,
    }).success,
    true,
  );
  assert.equal(
    requestSchema.safeParse({
      input: "x",
      questions: [{ type: "predicate", instructions: [part] }],
    }).success,
    false,
  );
});
test("JSON rejects nonfinite values, excess depth, count, and bytes", () => {
  const parse = (data: unknown) =>
    requestSchema.safeParse({
      input: [{ role: "user", content: [{ type: "input_json", data }] }],
      questions: [{ type: "predicate", instructions: "x" }],
    }).success;
  assert.equal(parse({ x: [true, null, 3, "text"] }), true);
  assert.equal(parse({ x: Infinity }), false);
  assert.equal(parse({ x: "x".repeat(256 * 1024) }), false);
  assert.equal(parse({ x: Array(10000).fill(null) }), false);
  let data: unknown = {};
  for (let i = 0; i < 32; i++) data = { data };
  assert.equal(parse(data), false);
});

test("identifier and score boundaries", () => {
  const body = {
    input: "x",
    questions: [{ type: "predicate", instructions: "x" }],
  };
  assert.equal(
    requestSchema.safeParse({ ...body, safety_identifier: "x".repeat(128) })
      .success,
    true,
  );
  assert.equal(
    requestSchema.safeParse({ ...body, safety_identifier: "x".repeat(129) })
      .success,
    false,
  );
  for (const length of [1, 2, 10, 11, 255])
    assert.equal(
      requestSchema.safeParse({
        ...body,
        questions: [
          {
            type: "score",
            instructions: "x",
            levels: Array.from({ length }, (_, i) => ({ label: String(i) })),
          },
        ],
      }).success,
      length >= 2 && length <= 10,
    );
  assert.equal(
    requestSchema.safeParse({
      ...body,
      questions: [
        { type: "score", instructions: "x", levels: [{ label: "only" }] },
      ],
    }).success,
    false,
  );
});
test("mismatched answer names and types are rejected", async () => {
  for (const answer of [
    { type: "predicate", name: "wrong", probability: 1 },
    { type: "score", name: null, score: 1 },
  ]) {
    const server = createServer(
      new OpenAI({
        apiKey: "fake",
        fetch: async () =>
          new Response(JSON.stringify({ answers: [answer] }), {
            headers: { "content-type": "application/json" },
          }),
      }),
    );
    const client = new Client({ name: "test", version: "1" });
    const [a, b] = InMemoryTransport.createLinkedPair();
    await server.connect(a);
    await client.connect(b);
    try {
      assert.equal(
        (
          await client.callTool({
            name: "decide",
            arguments: {
              input: "x",
              questions: [{ type: "predicate", instructions: "x" }],
            },
          })
        ).isError,
        true,
      );
    } finally {
      await client.close();
      await server.close();
    }
  }
});
