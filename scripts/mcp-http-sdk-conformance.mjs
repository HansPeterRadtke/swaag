#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [sdkRoot, endpoint, workspaceRoot, subscriptionReadyPath] = process.argv.slice(2);
if (!sdkRoot || !endpoint || !workspaceRoot || !subscriptionReadyPath) {
  throw new Error(
    "usage: mcp-http-sdk-conformance.mjs SDK_ROOT ENDPOINT WORKSPACE_ROOT SUBSCRIPTION_READY_PATH",
  );
}

const packageVersion = JSON.parse(
  fs.readFileSync(
    path.join(
      sdkRoot,
      "node_modules",
      "@modelcontextprotocol",
      "client",
      "package.json",
    ),
    "utf8",
  ),
).version;
const clientModule = await import(
  pathToFileURL(
    path.join(
      sdkRoot,
      "node_modules",
      "@modelcontextprotocol",
      "client",
      "dist",
      "index.mjs",
    ),
  ).href
);
const { Client, StreamableHTTPClientTransport } = clientModule;
const transport = new StreamableHTTPClientTransport(new URL(endpoint));
const client = new Client(
  { name: "swaag-http-conformance-probe", version: "1.0.0" },
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
  if (!discover) {
    throw new Error("official client did not retain server/discover result");
  }
  if (client.getProtocolEra() !== "modern") {
    throw new Error("official client did not negotiate the modern era");
  }
  if (client.getNegotiatedProtocolVersion() !== "2026-07-28") {
    throw new Error("official client negotiated an unexpected protocol version");
  }
  const listed = await client.listTools(undefined, { cacheMode: "refresh" });
  if (!listed.tools.some((tool) => tool.name === "list_files")) {
    throw new Error("official client did not decode list_files");
  }
  const calculator = listed.tools.find((tool) => tool.name === "calculator");
  if (
    calculator?.inputSchema?.properties?.expression?.["x-mcp-header"] !==
    "Expression"
  ) {
    throw new Error("official client did not retain the mirrored-header schema");
  }
  const files = await client.callTool({
    name: "list_files",
    arguments: { path: workspaceRoot },
  });
  if (files.isError || files.structuredContent === undefined) {
    throw new Error("official client did not decode list_files output");
  }
  const calculated = await client.callTool({
    name: "calculator",
    arguments: { expression: "6 * 7" },
  });
  if (
    calculated.isError ||
    calculated.structuredContent?.result !== 42
  ) {
    throw new Error("official client did not execute the mirrored-header call");
  }
  let resolveToolListChanged;
  const toolListChanged = new Promise((resolve) => {
    resolveToolListChanged = resolve;
  });
  client.setNotificationHandler(
    "notifications/tools/list_changed",
    (notification) => resolveToolListChanged(notification),
  );
  const subscription = await client.listen(
    { toolsListChanged: true },
    { timeout: 10000 },
  );
  if (subscription.honoredFilter?.toolsListChanged !== true) {
    throw new Error("official client subscription did not honor toolsListChanged");
  }
  fs.writeFileSync(subscriptionReadyPath, "ready\n", "utf8");
  const listChanged = await Promise.race([
    toolListChanged,
    new Promise((_, reject) =>
      setTimeout(
        () => reject(new Error("official client did not receive tools/list_changed")),
        10000,
      ),
    ),
  ]);
  if (listChanged?.method !== "notifications/tools/list_changed") {
    throw new Error("official client decoded an unexpected subscription notification");
  }
  const subscriptionId =
    listChanged?.params?._meta?.["io.modelcontextprotocol/subscriptionId"];
  if (typeof subscriptionId !== "string" || !subscriptionId.startsWith("listen:")) {
    throw new Error("tools/list_changed omitted the standard subscription id metadata");
  }
  await subscription.close();
  const closeCause = await subscription.closed;
  if (closeCause !== "local") {
    throw new Error(`official client subscription closed unexpectedly: ${closeCause}`);
  }
  report = {
    sdk: `@modelcontextprotocol/client@${packageVersion}`,
    transport: "streamable-http",
    protocolEra: client.getProtocolEra(),
    negotiatedProtocolVersion: client.getNegotiatedProtocolVersion(),
    serverVersion: client.getServerVersion(),
    discover,
    toolCount: listed.tools.length,
    toolCalls: {
      listFiles: {
        isError: files.isError ?? false,
        hasStructuredContent: true,
      },
      calculator: {
        isError: calculated.isError ?? false,
        result: calculated.structuredContent.result,
        mirroredParameterHeader: "Expression",
      },
    },
    subscription: {
      honoredFilter: subscription.honoredFilter,
      notificationMethod: listChanged.method,
      subscriptionId,
      closeCause,
    },
  };
} finally {
  await client.close();
}

console.log(JSON.stringify(report, null, 2));
