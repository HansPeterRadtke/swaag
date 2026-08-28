#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [sdkRoot, swaagExecutable, sessionsRoot, workspaceRoot] = process.argv.slice(2);
if (!sdkRoot || !swaagExecutable || !sessionsRoot || !workspaceRoot) {
  throw new Error(
    "usage: mcp-sdk-conformance.mjs SDK_ROOT SWAAG_EXECUTABLE SESSIONS_ROOT WORKSPACE_ROOT",
  );
}

const packageVersion = JSON.parse(
  fs.readFileSync(
    path.join(sdkRoot, "node_modules", "@modelcontextprotocol", "client", "package.json"),
    "utf8",
  ),
).version;
const clientModule = await import(
  pathToFileURL(
    path.join(sdkRoot, "node_modules", "@modelcontextprotocol", "client", "dist", "index.mjs"),
  ).href
);
const stdioModule = await import(
  pathToFileURL(
    path.join(sdkRoot, "node_modules", "@modelcontextprotocol", "client", "dist", "stdio.mjs"),
  ).href
);
const { Client } = clientModule;
const { StdioClientTransport } = stdioModule;

const transport = new StdioClientTransport({
  command: swaagExecutable,
  args: ["mcp-stdio"],
  cwd: workspaceRoot,
  env: {
    ...process.env,
    SWAAG__SESSIONS__ROOT: sessionsRoot,
    SWAAG__MODEL__BASE_URL: "http://127.0.0.1:1",
    SWAAG__MCP__ENABLED: "true",
  },
  stderr: "pipe",
});
let stderr = "";
transport.stderr?.on("data", (chunk) => {
  stderr += chunk.toString();
});

const client = new Client(
  { name: "swaag-conformance-probe", version: "1.0.0" },
  {
    capabilities: {},
    versionNegotiation: {
      mode: { pin: "2026-07-28" },
      probe: { timeoutMs: 10000 },
    },
    enforceStrictCapabilities: true,
  },
);

let report;
try {
  await client.connect(transport);
  const discover = client.getDiscoverResult();
  if (!discover) throw new Error("official client did not retain server/discover result");
  if (client.getProtocolEra() !== "modern") throw new Error("official client did not negotiate the modern era");
  if (client.getNegotiatedProtocolVersion() !== "2026-07-28") {
    throw new Error("official client negotiated an unexpected protocol version");
  }
  const listed = await client.listTools(undefined, { cacheMode: "refresh" });
  if (!Array.isArray(listed.tools) || listed.tools.length === 0) {
    throw new Error("official client decoded an empty tools/list result");
  }
  if (!listed.tools.some((tool) => tool.name === "list_files")) {
    throw new Error("official client did not decode the list_files capability");
  }
  const called = await client.callTool({
    name: "list_files",
    arguments: { path: "." },
  });
  if (called.isError) throw new Error("official client decoded list_files as a tool error");
  if (!Array.isArray(called.content) || !called.content.some((item) => item.type === "text")) {
    throw new Error("official client did not decode tool text content");
  }
  if (called.structuredContent === undefined) {
    throw new Error("official client did not decode structured tool content");
  }
  report = {
    sdk: `@modelcontextprotocol/client@${packageVersion}`,
    protocolEra: client.getProtocolEra(),
    negotiatedProtocolVersion: client.getNegotiatedProtocolVersion(),
    serverVersion: client.getServerVersion(),
    discover,
    toolCount: listed.tools.length,
    toolCall: {
      isError: called.isError ?? false,
      contentTypes: called.content.map((item) => item.type),
      hasStructuredContent: true,
    },
  };
} finally {
  await client.close();
}

if (stderr !== "") throw new Error(`Swaag MCP stderr was not empty: ${stderr}`);
console.log(JSON.stringify({ ...report, stderr }, null, 2));
