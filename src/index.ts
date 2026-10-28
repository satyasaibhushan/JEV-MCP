import OpenAI from "openai";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { createServer } from "./server.js";
if (!process.env.OPENAI_API_KEY) {
  process.stderr.write("OPENAI_API_KEY is required.\n");
  process.exit(1);
}
const server = createServer(new OpenAI({ apiKey: process.env.OPENAI_API_KEY }));
await server.connect(new StdioServerTransport());
