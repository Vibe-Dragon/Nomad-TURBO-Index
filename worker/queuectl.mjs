// Pause, resume, and edit the stock file-embeddings queue.
// Turbo does not patch NOMAD's embed worker. It runs this script inside the
// NOMAD image so the BullMQ calls match that install, and it only touches
// this queue. Other NOMAD queues keep running.
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire("/app/package.json");
const { Queue } = require("bullmq");

const queue = new Queue(process.env.QUEUE_NAME || "file-embeddings", {
  connection: {
    host: process.env.REDIS_HOST || "redis",
    port: Number(process.env.REDIS_PORT || 6379),
  },
});

function readStdin() {
  return readFileSync(0, "utf8");
}

const jobsToData = (jobs) =>
  jobs.filter(Boolean).map((job) => ({
    id: String(job.id),
    name: job.name,
    data: job.data,
  }));

try {
  const cmd = process.argv[2];
  if (cmd === "counts") {
    const counts = await queue.getJobCounts("active", "waiting", "paused", "delayed", "failed");
    process.stdout.write(JSON.stringify(counts));
  } else if (cmd === "pause") {
    await queue.pause();
    process.stdout.write("paused\n");
  } else if (cmd === "resume") {
    await queue.resume();
    process.stdout.write("resumed\n");
  } else if (cmd === "jobs") {
    const jobs = await queue.getJobs(["waiting", "paused", "delayed", "active", "prioritized"]);
    process.stdout.write(JSON.stringify(jobsToData(jobs)));
  } else if (cmd === "update") {
    const id = process.argv[3];
    const patch = JSON.parse(readStdin() || "{}");
    const job = await queue.getJob(id);
    if (!job) {
      throw new Error(`job ${id} not found`);
    }
    await job.updateData({ ...job.data, ...patch });
    process.stdout.write("updated\n");
  } else if (cmd === "remove") {
    const id = process.argv[3];
    const job = await queue.getJob(id);
    if (job) {
      await job.remove();
    }
    process.stdout.write("removed\n");
  } else if (cmd === "clear") {
    const counts = await queue.getJobCounts("active");
    if ((counts.active || 0) > 0) {
      throw new Error("refusing to clear the embed queue while a job is active");
    }
    await queue.obliterate({ force: false });
    process.stdout.write("cleared\n");
  } else if (cmd === "add") {
    const data = JSON.parse(readStdin());
    const job = await queue.add("embed-file", data, {
      attempts: 30,
      backoff: { type: "fixed", delay: 60000 },
      removeOnComplete: { count: 50 },
      removeOnFail: { count: 20 },
    });
    process.stdout.write(String(job.id) + "\n");
  } else {
    throw new Error("use counts|pause|resume|jobs|update|remove|add|clear");
  }
  await queue.close();
} catch (error) {
  console.error(error && error.stack ? error.stack : error);
  try {
    await queue.close();
  } catch {
    // already failing
  }
  process.exit(1);
}
