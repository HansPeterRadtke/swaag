#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [sdkRoot, baseUrl, threadId, runId, expectedResult] = process.argv.slice(2);
if (!sdkRoot || !baseUrl || !threadId || !runId || !expectedResult) {
  throw new Error(
    "usage: ag-ui-sdk-conformance.mjs SDK_ROOT BASE_URL THREAD_ID RUN_ID EXPECTED_RESULT",
  );
}

const packageRoot = path.join(sdkRoot, "node_modules", "@ag-ui", "client");
const packageVersion = JSON.parse(
  fs.readFileSync(path.join(packageRoot, "package.json"), "utf8"),
).version;
const { HttpAgent } = await import(
  pathToFileURL(path.join(packageRoot, "dist", "index.mjs")).href
);

const agent = new HttpAgent({
  url: `${baseUrl.replace(/\/$/, "")}/ag-ui`,
  threadId,
  initialMessages: [
    {
      id: "official-sdk-request",
      role: "user",
      content: "Return the existing durable run result.",
    },
  ],
});
const eventTypes = [];
const run = await agent.runAgent(
  { runId },
  {
    onRunStartedEvent({ event }) {
      eventTypes.push(event.type);
    },
    onTextMessageStartEvent({ event }) {
      eventTypes.push(event.type);
    },
    onTextMessageContentEvent({ event }) {
      eventTypes.push(event.type);
    },
    onTextMessageEndEvent({ event }) {
      eventTypes.push(event.type);
    },
    onRunFinishedEvent({ event }) {
      eventTypes.push(event.type);
    },
  },
);

const requiredEvents = [
  "RUN_STARTED",
  "TEXT_MESSAGE_START",
  "TEXT_MESSAGE_CONTENT",
  "TEXT_MESSAGE_END",
  "RUN_FINISHED",
];
if (requiredEvents.some((eventType) => !eventTypes.includes(eventType))) {
  throw new Error("official client did not decode the complete AG-UI event lifecycle");
}
if (run.result !== expectedResult) {
  throw new Error("official client did not decode the exact RUN_FINISHED result");
}
const assistant = run.newMessages.find(
  (message) => message.role === "assistant" && message.content === expectedResult,
);
if (!assistant) {
  throw new Error("official client did not assemble the exact assistant message");
}

console.log(
  JSON.stringify(
    {
      sdk: `@ag-ui/client@${packageVersion}`,
      endpoint: agent.url,
      threadId: agent.threadId,
      eventTypes,
      result: run.result,
      newMessageRoles: run.newMessages.map((message) => message.role),
      assistantMessageId: assistant.id,
    },
    null,
    2,
  ),
);
