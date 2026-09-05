#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [sdkRoot, command, argsJson, envJson, expectedToolsJson, callJson = "null"] = process.argv.slice(2);
if (!sdkRoot || !command || !argsJson || !envJson || !expectedToolsJson) {
  throw new Error(
    "usage: external-mcp-sdk-conformance.mjs SDK_ROOT COMMAND ARGS_JSON ENV_JSON EXPECTED_TOOLS_JSON [CALL_JSON]",
  );
}

const args = JSON.parse(argsJson);
const extraEnv = JSON.parse(envJson);
const expectedTools = JSON.parse(expectedToolsJson);
const callSpec = JSON.parse(callJson);
if (!Array.isArray(args) || typeof extraEnv !== "object" || !Array.isArray(expectedTools)) {
  throw new Error("invalid probe arguments");
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
  ).href,
);
const stdioModule = await import(
  pathToFileURL(
    path.join(sdkRoot, "node_modules", "@modelcontextprotocol", "client", "dist", "stdio.mjs"),
  ).href,
);
const { Client } = clientModule;
const { StdioClientTransport } = stdioModule;

const transport = new StdioClientTransport({
  command,
  args,
  env: { ...process.env, ...extraEnv },
  stderr: "pipe",
});
let stderr = "";
transport.stderr?.on("data", (chunk) => {
  stderr += chunk.toString();
});

const client = new Client(
  { name: "swaag-external-mcp-conformance", version: "1.0.0" },
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
  if (client.getProtocolEra() !== "modern") throw new Error("official client did not negotiate modern MCP");
  if (client.getNegotiatedProtocolVersion() !== "2026-07-28") {
    throw new Error("official client negotiated an unexpected protocol version");
  }
  const listed = await client.listTools(undefined, { cacheMode: "refresh" });
  const names = listed.tools.map((tool) => tool.name);
  for (const expected of expectedTools) {
    if (!names.includes(expected)) throw new Error(`missing expected external tool ${expected}`);
  }
  let callResult = null;
  if (callSpec !== null) {
    if (typeof callSpec !== "object" || typeof callSpec.name !== "string") {
      throw new Error("CALL_JSON must be null or an object with name/arguments");
    }
    const called = await client.callTool({
      name: callSpec.name,
      arguments: callSpec.arguments ?? {},
    });
    if (called.isError) throw new Error(`external tool ${callSpec.name} returned isError=true`);
    callResult = {
      name: callSpec.name,
      contentTypes: Array.isArray(called.content) ? called.content.map((item) => item.type) : [],
      hasStructuredContent: called.structuredContent !== undefined,
      structuredContent: called.structuredContent,
    };
  }
  report = {
    sdk: `@modelcontextprotocol/client@${packageVersion}`,
    protocolEra: client.getProtocolEra(),
    negotiatedProtocolVersion: client.getNegotiatedProtocolVersion(),
    serverVersion: client.getServerVersion(),
    discover,
    toolNames: names,
    call: callResult,
  };
} finally {
  await client.close();
}

if (stderr.trim() !== "") {
  throw new Error(`external MCP server stderr was not empty: ${stderr.trim()}`);
}
console.log(JSON.stringify(report, null, 2));
