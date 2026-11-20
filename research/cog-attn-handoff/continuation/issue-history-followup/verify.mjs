import { readFileSync } from "node:fs";
import { checkInvariants, verifyRows } from "./anchor.mjs";

const read = name => JSON.parse(readFileSync(new URL(name, import.meta.url), "utf8"));
const cases = read("./cases.json");
const expected = read("./results.json");
const report = verifyRows(cases);
report.invariantChecks = checkInvariants(cases, report);
if (JSON.stringify(report) !== JSON.stringify(expected)) {
  throw new Error("Replayed output differs from the archived report");
}
console.log(JSON.stringify(report, null, 2));
