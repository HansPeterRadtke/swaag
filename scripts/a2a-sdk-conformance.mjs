#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [sdkRoot, baseUrl, ...options] = process.argv.slice(2);
const exerciseNewTasks = options.includes("--exercise-new-tasks");
const transportOptions = options.filter((item) => item.startsWith("--transport="));
if (transportOptions.length > 1) {
  throw new Error("at most one --transport option may be provided");
}
const transport = transportOptions[0]?.slice("--transport=".length) ?? "jsonrpc";
if (!new Set(["jsonrpc", "http-json"]).has(transport)) {
  throw new Error("--transport must be jsonrpc or http-json");
}
const unknownOptions = options.filter(
  (item) => item.startsWith("--") &&
    item !== "--exercise-new-tasks" &&
    !item.startsWith("--transport="),
);
if (unknownOptions.length) {
  throw new Error(`unknown options: ${unknownOptions.join(", ")}`);
}
const taskIds = options.filter((item) => !item.startsWith("--"));
if (taskIds.length > 1) {
  throw new Error("at most one EXPECTED_TASK_ID may be provided");
}
const [expectedTaskId] = taskIds;
if (!sdkRoot || !baseUrl) {
  throw new Error(
    "usage: a2a-sdk-conformance.mjs SDK_ROOT BASE_URL [EXPECTED_TASK_ID] " +
      "[--exercise-new-tasks] [--transport=jsonrpc|http-json]",
  );
}

const packageRoot = path.join(sdkRoot, "node_modules", "@a2a-js", "sdk");
const packageVersion = JSON.parse(
  fs.readFileSync(path.join(packageRoot, "package.json"), "utf8"),
).version;
const { ClientFactory, JsonRpcTransportFactory, RestTransportFactory } =
  await import(
    pathToFileURL(path.join(packageRoot, "dist", "client", "index.js")).href
  );
const { Role, TaskState } = await import(
  pathToFileURL(path.join(packageRoot, "dist", "index.js")).href
);

const binding = transport === "http-json" ? "HTTP+JSON" : "JSONRPC";
const factory = new ClientFactory({
  transports: [
    transport === "http-json"
      ? new RestTransportFactory()
      : new JsonRpcTransportFactory(),
  ],
});
const client = await factory.createFromUrl(baseUrl);
const card = await client.getAgentCard();
if (client.protocolVersion !== "1.0" || card.name !== "Swaag") {
  throw new Error("official client decoded an unexpected Swaag Agent Card");
}
const selectedInterface = card.supportedInterfaces?.find(
  (item) => item.protocolBinding === binding && item.protocolVersion === "1.0",
);
if (!selectedInterface?.url) {
  throw new Error(`official client did not decode the A2A 1.0 ${binding} interface`);
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

let newTasks;
if (exerciseNewTasks) {
  const unary = await client.sendMessage(
    newTaskRequest("official-unary-message", "Exercise official unary task creation."),
  );
  requireSubmittedTask(unary, "unary SendMessage");
  const unaryCanceled = await client.cancelTask({ tenant: "", id: unary.id });
  requireCanceledTask(unaryCanceled, unary.id, "unary task cancellation");

  const streamingIterator = client
    .sendMessageStream(
      newTaskRequest("official-stream-message", "Exercise official streaming task creation."),
    )
    [Symbol.asyncIterator]();
  const streamingInitial = await withTimeout(
    streamingIterator.next(),
    "new streaming task event",
  );
  if (streamingInitial.done || streamingInitial.value?.payload?.$case !== "task") {
    throw new Error("official client did not decode the newly streamed task");
  }
  const streamingTask = streamingInitial.value.payload.value;
  requireSubmittedTask(streamingTask, "streaming SendMessage");
  const streamingCanceled = await client.cancelTask({
    tenant: "",
    id: streamingTask.id,
  });
  requireCanceledTask(
    streamingCanceled,
    streamingTask.id,
    "streaming task cancellation",
  );
  const streamingTerminal = await consumeCanceledStream(streamingIterator);

  newTasks = {
    unary: {
      id: unary.id,
      contextId: unary.contextId,
      initialState: TaskState[unary.status.state],
      canceledState: TaskState[unaryCanceled.status.state],
    },
    streaming: {
      id: streamingTask.id,
      contextId: streamingTask.contextId,
      initialCase: streamingInitial.value.payload.$case,
      initialState: TaskState[streamingTask.status.state],
      canceledState: TaskState[streamingCanceled.status.state],
      ...streamingTerminal,
    },
  };
}

console.log(
  JSON.stringify(
    {
      sdk: `@a2a-js/sdk@${packageVersion}`,
      transport,
      protocolVersion: client.protocolVersion,
      card: {
        name: card.name,
        version: card.version,
        streaming: card.capabilities?.streaming ?? false,
        interface: selectedInterface,
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
      newTasks: newTasks ?? null,
    },
    null,
    2,
  ),
);

function newTaskRequest(messageId, textValue) {
  return {
    tenant: "",
    message: {
      messageId,
      contextId: "",
      taskId: "",
      role: Role.ROLE_USER,
      parts: [
        {
          content: { $case: "text", value: textValue },
          metadata: undefined,
          filename: "",
          mediaType: "text/plain",
        },
      ],
      metadata: undefined,
      extensions: [],
      referenceTaskIds: [],
    },
    configuration: {
      acceptedOutputModes: ["text/plain"],
      taskPushNotificationConfig: undefined,
      historyLength: 0,
      returnImmediately: true,
    },
    metadata: undefined,
  };
}

function requireSubmittedTask(taskValue, label) {
  if (
    !taskValue?.id ||
    !taskValue.contextId ||
    taskValue.status?.state !== TaskState.TASK_STATE_SUBMITTED
  ) {
    throw new Error(`official client did not decode ${label} as a submitted task`);
  }
}

function requireCanceledTask(taskValue, expectedId, label) {
  if (
    taskValue?.id !== expectedId ||
    taskValue.status?.state !== TaskState.TASK_STATE_CANCELED
  ) {
    throw new Error(`official client did not decode ${label}`);
  }
}

async function consumeCanceledStream(iterator) {
  const updateCases = [];
  let sawCanceledUpdate = false;
  let closed = false;
  for (let index = 0; index < 12; index += 1) {
    const next = await withTimeout(iterator.next(), "new task terminal stream event");
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
    throw new Error(
      "official client new-task stream did not deliver and close after cancellation",
    );
  }
  return { updateCases, sawCanceledUpdate, closed };
}

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
