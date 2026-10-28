import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer as httpServer } from "node:http";
import { once } from "node:events";
import { spawnSync } from "node:child_process";
import OpenAI from "openai";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createServer } from "../src/server.js";

const args = {
  input: "x",
  questions: [{ type: "predicate", instructions: "x" }],
};
test("real HTTP cancellation and timeout close requests without retry", async () => {
  for (const cancel of [true, false]) {
    let requests = 0;
    let received!: () => void;
    let closed!: () => void;
    const requestReceived = new Promise<void>((resolve) => { received = resolve; });
    const requestClosed = new Promise<void>((resolve) => { closed = resolve; });
    const http = httpServer((_req, res) => {
      requests++;
      res.on("close", closed);
      received();
    });
    http.listen(0, "127.0.0.1");
    await once(http, "listening");
    const address = http.address();
    assert(address && typeof address !== "string");
    const server = createServer(
      new OpenAI({
        apiKey: "fake",
        baseURL: `http://127.0.0.1:${address.port}/v1`,
      }),
    );
    const client = new Client({ name: "test", version: "1" });
    const [a, b] = InMemoryTransport.createLinkedPair();
    await server.connect(a);
    await client.connect(b);
    const controller = new AbortController();
    const started = Date.now();
    try {
      const pending = client.callTool(
        { name: "decide", arguments: args },
        undefined,
        { signal: controller.signal, timeout: 40000 },
      );
      if (cancel) {
        const rejected = assert.rejects(pending);
        await requestReceived;
        controller.abort();
        await rejected;
      } else {
        assert.equal((await pending).isError, true);
        assert(Date.now() - started >= 29000);
      }
      let deadline: ReturnType<typeof setTimeout> | undefined;
      try {
        await Promise.race([
          requestClosed,
          new Promise<never>((_resolve, reject) => {
            deadline = setTimeout(() => reject(new Error("HTTP request stayed open")), 2000);
          }),
        ]);
      } finally {
        clearTimeout(deadline);
      }
      assert.equal(requests, 1);
      assert(Date.now() - started < (cancel ? 5000 : 35000));
    } finally {
      await client.close();
      await server.close();
      http.closeAllConnections();
      await new Promise<void>((resolve) => http.close(() => resolve()));
    }
  }
});
test("oversized body never reaches HTTP; errors never retry", async () => {
  let count = 0;
  const server = createServer(
    new OpenAI({
      apiKey: "fake",
      fetch: async () => {
        count++;
        return new Response("secret-provider-body", { status: 500 });
      },
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
          arguments: { ...args, input: "x".repeat(1024 * 1024) },
        })
      ).isError,
      true,
    );
    assert.equal(count, 0);
    const result = await client.callTool({ name: "decide", arguments: args });
    assert.equal(result.isError, true);
    assert.doesNotMatch(JSON.stringify(result), /secret-provider-body|fake/);
    assert.equal(count, 1);
  } finally {
    await client.close();
    await server.close();
  }
});
test("built subprocess initializes, lists and calls with protocol-only stdout", async () => {
  const http = httpServer((_req, res) => {
    res.setHeader("content-type", "application/json");
    res.end(
      JSON.stringify({
        answers: [{ type: "refusal", name: null }],
        usage: { total_tokens: 1 },
      }),
    );
  });
  http.listen(0, "127.0.0.1");
  await once(http, "listening");
  const address = http.address();
  assert(address && typeof address !== "string");
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: ["dist/index.js"],
    env: {
      OPENAI_API_KEY: "fake",
      OPENAI_BASE_URL: `http://127.0.0.1:${address.port}/v1`,
    },
    stderr: "pipe",
  });
  const client = new Client({ name: "test", version: "1" });
  try {
    await client.connect(transport);
    assert.equal((await client.listTools()).tools[0].name, "decide");
    assert.equal(
      (await client.callTool({ name: "decide", arguments: args })).isError,
      undefined,
    );
  } finally {
    await client.close();
    http.closeAllConnections();
    await new Promise<void>((resolve) => http.close(() => resolve()));
  }
  const env = { ...process.env };
  delete env.OPENAI_API_KEY;
  const missing = spawnSync(process.execPath, ["dist/index.js"], {
    env,
    encoding: "utf8",
  });
  assert.equal(missing.status, 1);
  assert.equal(missing.stdout, "");
  assert.equal(missing.stderr, "OPENAI_API_KEY is required.\n");
});
