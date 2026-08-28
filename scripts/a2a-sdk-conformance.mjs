#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [sdkRoot, baseUrl, expectedTaskId] = process.argv.slice(2);
if (!sdkRoot || !baseUrl) {
  throw new Error("usage: a2a-sdk-conformance.mjs SDK_ROOT BASE_URL [EXPECTED_TASK_ID]");
}

const packageRoot = path.join(sdkRoot, "node_modules", "@a2a-js", "sdk");
const packageVersion = JSON.parse(
  fs.readFileSync(path.join(packageRoot, "package.json"), "utf8"),
).version;
const { ClientFactory, JsonRpcTransportFactory } = await import(
  pathToFileURL(path.join(packageRoot, "dist", "client", "index.js")).href
);
const { TaskState } = await import(
  pathToFileURL(path.join(packageRoot, "dist", "index.js")).href
);

const factory = new ClientFactory({
  transports: [new JsonRpcTransportFactory()],
});
const client = await factory.createFromUrl(baseUrl);
const card = await client.getAgentCard();
if (client.protocolVersion !== "1.0" || card.name !== "Swaag") {
  throw new Error("official client decoded an unexpected Swaag Agent Card");
}
const jsonRpc = card.supportedInterfaces?.find(
  (item) => item.protocolBinding === "JSONRPC" && item.protocolVersion === "1.0",
);
if (!jsonRpc?.url) {
  throw new Error("official client did not decode the A2A 1.0 JSON-RPC interface");
}

const listed = await client.listTasks({
  tenant: "",
  contextId: "",
  status: TaskState.TASK_STATE_UNSPECIFIED,
  pageToken: "",
  statusTimestampAfter: undefined,
  includeArtifacts: true,
});
if (!Array.isArray(listed.tasks) || typeof listed.totalSize !== "number") {
  throw new Error("official client did not decode the ListTasks response");
}

let task;
let stream;
if (expectedTaskId) {
  task = await client.getTask({ tenant: "", id: expectedTaskId });
  if (
    task.id !== expectedTaskId ||
    typeof task.status?.state !== "number" ||
    task.status.state === TaskState.UNRECOGNIZED
  ) {
    throw new Error("official client did not decode the expected task");
  }
  if (!listed.tasks.some((item) => item.id === expectedTaskId)) {
    throw new Error("the expected task was absent from the official ListTasks response");
  }

  const iterator = client
    .resubscribeTask({ tenant: "", id: expectedTaskId })
    [Symbol.asyncIterator]();
  const initial = await withTimeout(iterator.next(), "initial task stream event");
  if (initial.done || initial.value?.payload?.$case !== "task") {
    throw new Error("official client did not decode the initial streamed task");
  }
  const canceled = await client.cancelTask({ tenant: "", id: expectedTaskId });
  if (
    canceled.id !== expectedTaskId ||
    canceled.status?.state !== TaskState.TASK_STATE_CANCELED
  ) {
    throw new Error("official client did not decode task cancellation");
  }
  const updateCases = [];
  let sawCanceledUpdate = false;
  let closed = false;
  for (let index = 0; index < 12; index += 1) {
    const next = await withTimeout(iterator.next(), "terminal task stream event");
    if (next.done) {
      closed = true;
      break;
    }
    const payload = next.value?.payload;
    updateCases.push(payload?.$case ?? "unknown");
    if (
      payload?.$case === "statusUpdate" &&
      payload.value?.status?.state === TaskState.TASK_STATE_CANCELED
    ) {
      sawCanceledUpdate = true;
    }
  }
  if (!sawCanceledUpdate || !closed) {
    throw new Error("official client stream did not deliver and close after cancellation");
  }
  stream = {
    initialCase: initial.value.payload.$case,
    canceledState: TaskState[canceled.status.state],
    updateCases,
    closed,
  };
}

console.log(
  JSON.stringify(
    {
      sdk: `@a2a-js/sdk@${packageVersion}`,
      protocolVersion: client.protocolVersion,
      card: {
        name: card.name,
        version: card.version,
        streaming: card.capabilities?.streaming ?? false,
        interface: jsonRpc,
        skillCount: card.skills?.length ?? 0,
      },
      list: {
        taskCount: listed.tasks.length,
        totalSize: listed.totalSize,
        pageSize: listed.pageSize,
        nextPageToken: listed.nextPageToken,
      },
      task: task
        ? {
            id: task.id,
            contextId: task.contextId,
            state: TaskState[task.status.state],
            historyCount: task.history?.length ?? 0,
            artifactCount: task.artifacts?.length ?? 0,
          }
        : null,
      stream: stream ?? null,
    },
    null,
    2,
  ),
);

async function withTimeout(promise, label) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`timed out waiting for ${label}`)), 10000);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}
