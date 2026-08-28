#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [sdkRoot, baseUrl, threadId, firstRunId, secondRunId, expectedResult] =
  process.argv.slice(2);
if (!sdkRoot || !baseUrl || !threadId || !firstRunId || !secondRunId || !expectedResult) {
  throw new Error(
    "usage: ag-ui-client-tool-conformance.mjs SDK_ROOT BASE_URL THREAD_ID FIRST_RUN_ID SECOND_RUN_ID EXPECTED_RESULT",
  );
}

const clientRoot = path.join(sdkRoot, "node_modules", "@ag-ui", "client");
const coreRoot = path.join(sdkRoot, "node_modules", "@ag-ui", "core");
const packageVersion = JSON.parse(
  fs.readFileSync(path.join(clientRoot, "package.json"), "utf8"),
).version;
const { HttpAgent } = await import(
  pathToFileURL(path.join(clientRoot, "dist", "index.mjs")).href
);
const { AgentCapabilitiesSchema } = await import(
  pathToFileURL(path.join(coreRoot, "dist", "index.mjs")).href
);

const normalizedBaseUrl = baseUrl.replace(/\/$/, "");
const capabilityResponse = await fetch(`${normalizedBaseUrl}/ag-ui/capabilities`);
if (!capabilityResponse.ok) {
  throw new Error(`AG-UI capability discovery returned ${capabilityResponse.status}`);
}
const capabilities = AgentCapabilitiesSchema.parse(await capabilityResponse.json());
if (capabilities.tools?.clientProvided !== true) {
  throw new Error("AG-UI did not advertise client-provided tool execution");
}
if (capabilities.tools?.parallelCalls !== false) {
  throw new Error("AG-UI did not advertise its sequential client-tool limit");
}

const clientTool = {
  name: "select_record",
  description: "Select one record in the connected client.",
  parameters: {
    type: "object",
    properties: { record_id: { type: "string" } },
    required: ["record_id"],
    additionalProperties: false,
  },
  metadata: { owner: "official-sdk-probe" },
};
const toolResultContent = JSON.stringify({ selected: "record-7", visible: true });
const agent = new HttpAgent({
  url: `${normalizedBaseUrl}/ag-ui`,
  threadId,
  initialMessages: [
    {
      id: "official-client-tool-request",
      role: "user",
      content: "Ask the connected client to select record 7, then report the result.",
    },
  ],
});

const firstEventTypes = [];
const first = await agent.runAgent(
  { runId: firstRunId, tools: [clientTool] },
  {
    onEvent({ event }) {
      firstEventTypes.push(event.type);
    },
  },
);
const assistantCallMessage = first.newMessages.find(
  (message) => message.role === "assistant" && message.toolCalls?.length === 1,
);
if (!assistantCallMessage) {
  throw new Error("official client did not assemble the delegated tool call");
}
const toolCall = assistantCallMessage.toolCalls[0];
if (toolCall.function.name !== clientTool.name) {
  throw new Error("official client assembled the wrong delegated tool name");
}
if (JSON.parse(toolCall.function.arguments).record_id !== "record-7") {
  throw new Error("official client assembled the wrong delegated tool arguments");
}
for (const required of [
  "RUN_STARTED",
  "STATE_SNAPSHOT",
  "TOOL_CALL_START",
  "TOOL_CALL_ARGS",
  "TOOL_CALL_END",
  "RUN_FINISHED",
]) {
  if (!firstEventTypes.includes(required)) {
    throw new Error(`delegated tool run omitted ${required}`);
  }
}

agent.addMessage({
  id: "official-client-tool-result",
  role: "tool",
  toolCallId: toolCall.id,
  content: toolResultContent,
  metadata: { durationMs: 4 },
});
const secondEventTypes = [];
const second = await agent.runAgent(
  { runId: secondRunId, tools: [clientTool] },
  {
    onEvent({ event }) {
      secondEventTypes.push(event.type);
    },
  },
);
if (second.result !== expectedResult) {
  throw new Error("official client did not decode the post-tool final result");
}
if (
  !second.newMessages.some(
    (message) => message.role === "assistant" && message.content === expectedResult,
  )
) {
  throw new Error("official client did not assemble the post-tool assistant message");
}
for (const required of [
  "RUN_STARTED",
  "STATE_SNAPSHOT",
  "TEXT_MESSAGE_START",
  "TEXT_MESSAGE_CONTENT",
  "TEXT_MESSAGE_END",
  "RUN_FINISHED",
]) {
  if (!secondEventTypes.includes(required)) {
    throw new Error(`post-tool run omitted ${required}`);
  }
}

console.log(
  JSON.stringify(
    {
      sdk: `@ag-ui/client@${packageVersion}`,
      endpoint: agent.url,
      capabilityEndpoint: `${normalizedBaseUrl}/ag-ui/capabilities`,
      capabilities,
      threadId: agent.threadId,
      firstRunId,
      secondRunId,
      firstEventTypes,
      secondEventTypes,
      toolCall,
      toolResultContent,
      result: second.result,
      messageRoles: agent.messages.map((message) => message.role),
    },
    null,
    2,
  ),
);
