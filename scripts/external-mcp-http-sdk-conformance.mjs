#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [sdkRoot, endpoint] = process.argv.slice(2);
if (!sdkRoot || !endpoint) {
  throw new Error("usage: external-mcp-http-sdk-conformance.mjs SDK_ROOT ENDPOINT");
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
const { Client, StreamableHTTPClientTransport } = clientModule;
const transport = new StreamableHTTPClientTransport(new URL(endpoint));
const client = new Client(
  { name: "swaag-external-http-conformance", version: "1.0.0" },
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
  if (!names.includes("external_echo")) throw new Error("official client did not decode external_echo");
  const called = await client.callTool({
    name: "external_echo",
    arguments: { text: "official-sdk" },
  });
  if (called.isError || called.structuredContent?.echo !== "official-sdk") {
    throw new Error("official client did not decode external_echo result");
  }
  report = {
    sdk: `@modelcontextprotocol/client@${packageVersion}`,
    transport: "streamable-http",
    protocolEra: client.getProtocolEra(),
    negotiatedProtocolVersion: client.getNegotiatedProtocolVersion(),
    serverVersion: client.getServerVersion(),
    discover,
    toolNames: names,
    call: {
      name: "external_echo",
      echo: called.structuredContent.echo,
      hasStructuredContent: called.structuredContent !== undefined,
    },
  };
} finally {
  await client.close();
}

console.log(JSON.stringify(report, null, 2));
