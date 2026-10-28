import OpenAI from "openai";
import type { DecisionCreateParams } from "openai/resources/decisions";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

function validJSON(data: unknown): boolean {
  const stack: Array<{ value: unknown; depth: number }> = [
    { value: data, depth: 0 },
  ];
  let count = 0;
  while (stack.length) {
    const { value, depth } = stack.pop()!;
    if (++count > 10000) return false;
    if (value !== null && typeof value === "object") {
      if (depth >= 32) return false;
      for (const child of Object.values(value))
        stack.push({ value: child, depth: depth + 1 });
    } else if (
      typeof value === "number"
        ? !Number.isFinite(value)
        : !["string", "boolean"].includes(typeof value) && value !== null
    )
      return false;
  }
  return Buffer.byteLength(JSON.stringify(data)) <= 256 * 1024;
}
const text = z
  .object({
    type: z.literal("input_text"),
    text: z.string(),
    prompt_cache_breakpoint: z
      .object({ mode: z.literal("explicit") })
      .strict()
      .nullable()
      .optional(),
  })
  .strict();
const image = z
  .object({
    type: z.literal("input_image"),
    image_url: z.string(),
    detail: z.enum(["low", "high", "auto", "original"]).nullable().optional(),
    prompt_cache_breakpoint: z
      .object({ mode: z.literal("explicit") })
      .strict()
      .nullable()
      .optional(),
  })
  .strict();
const json = z
  .object({
    type: z.literal("input_json"),
    data: z
      .union([z.record(z.unknown()), z.array(z.record(z.unknown()))])
      .refine(validJSON),
  })
  .strict();
const parts = z.union([text, image, json]);
const instructions = z.union([
  z.string(),
  z.array(
    z.union([
      text.omit({ prompt_cache_breakpoint: true }).strict(),
      image.omit({ prompt_cache_breakpoint: true }).strict(),
      json,
    ]),
  ),
]);
const question = z.discriminatedUnion("type", [
  z
    .object({
      type: z.literal("predicate"),
      instructions,
      name: z.string().optional(),
    })
    .strict(),
  z
    .object({
      type: z.literal("choice"),
      instructions,
      name: z.string().optional(),
      choices: z
        .array(
          z
            .object({
              value: z.union([z.string(), z.boolean()]),
              description: z.string().optional(),
            })
            .strict(),
        )
        .min(2)
        .max(255)
        .refine(
          (v) =>
            new Set(v.map((x) => JSON.stringify(x.value))).size === v.length,
        ),
    })
    .strict(),
  z
    .object({
      type: z.literal("score"),
      instructions,
      name: z.string().optional(),
      levels: z
        .array(
          z
            .object({ label: z.string(), description: z.string().optional() })
            .strict(),
        )
        .min(2)
        .max(10),
    })
    .strict(),
]);
const callFields = {
  call_id: z.string(),
  id: z.string().nullable().optional(),
  status: z
    .enum(["in_progress", "completed", "incomplete"])
    .nullable()
    .optional(),
};
const message = z.union([
  z
    .object({
      ...callFields,
      type: z.literal("function_call"),
      arguments: z.string(),
      name: z.string(),
      namespace: z.string().nullable().optional(),
    })
    .strict(),
  z
    .object({
      ...callFields,
      type: z.literal("custom_tool_call"),
      input: z.string(),
      name: z.string(),
      namespace: z.string().nullable().optional(),
    })
    .strict(),
  z
    .object({
      ...callFields,
      type: z.literal("function_call_output"),
      output: z.union([z.string(), z.array(z.union([text, image]))]),
    })
    .strict(),
  z
    .object({
      ...callFields,
      type: z.literal("custom_tool_call_output"),
      output: z.union([z.string(), z.array(z.union([text, image]))]),
    })
    .strict(),
  z
    .object({
      role: z.literal("user"),
      type: z.literal("message").optional(),
      content: z.union([z.string(), z.array(parts)]),
    })
    .strict(),
  z
    .object({
      role: z.enum(["system", "developer"]),
      type: z.literal("message").optional(),
      content: z.union([z.string(), z.array(text)]),
    })
    .strict(),
  z
    .object({
      role: z.literal("assistant"),
      type: z.literal("message").optional(),
      phase: z.enum(["commentary", "final_answer"]).nullable().optional(),
      content: z.union([
        z.string(),
        z.array(
          z.union([
            z
              .object({ type: z.literal("output_text"), text: z.string() })
              .strict(),
            z
              .object({ type: z.literal("refusal"), refusal: z.string() })
              .strict(),
          ]),
        ),
      ]),
    })
    .strict(),
]);
export const requestSchema = z
  .object({
    input: z.union([
      z.string(),
      z.array(message).superRefine((items, ctx) => {
        for (const item of items) {
          if (
            item.type === "function_call_output" ||
            item.type === "custom_tool_call_output"
          ) {
            const expected =
              item.type === "function_call_output"
                ? "function_call"
                : "custom_tool_call";
            if (
              !items.some(
                (call) =>
                  call.type === expected && call.call_id === item.call_id,
              )
            )
              ctx.addIssue({
                code: "custom",
                message: "Tool output requires a matching call ID.",
              });
          }
        }
      }),
    ]),
    model: z.literal("gpt-6-luna").default("gpt-6-luna"),
    questions: z.array(question).min(1).max(400),
    prompt_cache_key: z.string().nullable().optional(),
    prompt_cache_options: z
      .object({
        mode: z.enum(["implicit", "explicit"]).nullable().optional(),
        ttl: z.literal("30m").optional(),
      })
      .strict()
      .nullable()
      .optional(),
    safety_identifier: z.string().max(128).nullable().optional(),
  })
  .strict();

export function createServer(client: OpenAI) {
  const server = new McpServer({ name: "jev-mcp", version: "0.1.0" });
  server.registerTool(
    "decide",
    {
      description:
        "Evaluate caller-defined questions using the Decisions API. Returns native answers and usage in question order. Does not execute actions.",
      inputSchema: requestSchema,
    },
    async (args, extra) => {
      try {
        const body = requestSchema.parse(args);
        if (Buffer.byteLength(JSON.stringify(body)) > 1024 * 1024)
          throw new Error("bounded");
        const result = await client.decisions.create(
          body satisfies DecisionCreateParams,
          { signal: extra.signal, timeout: 30000, maxRetries: 0 },
        );
        if (
          result.answers.length !== body.questions.length ||
          result.answers.some(
            (a, i) =>
              a.name !== (body.questions[i].name ?? null) ||
              (a.type !== "refusal" && a.type !== body.questions[i].type),
          )
        )
          throw new Error("correspondence");
        return { content: [{ type: "text", text: JSON.stringify(result) }] };
      } catch (error: unknown) {
        const status =
          error instanceof OpenAI.APIError ? error.status : undefined;
        const message = extra.signal.aborted
          ? "Decision request cancelled."
          : status
            ? `Decision provider returned HTTP ${status}.`
            : "Decision request failed, timed out, or exceeded local limits.";
        return { isError: true, content: [{ type: "text", text: message }] };
      }
    },
  );
  return server;
}
