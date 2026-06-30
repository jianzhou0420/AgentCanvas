/** Node-type color mapping for the log viewer.
 * Colors help visually distinguish node categories in the entry list.
 * Prefix matching supports dynamic nodeset types (e.g., env_habitat__step → env prefix → green).
 */

interface NodeColor {
  dot: string; // Tailwind bg class for the 6px dot
  bg: string; // Tailwind bg class for subtle row tint
}

const EXACT_COLORS: Record<string, NodeColor> = {
  // LLM / VLM (blue)
  llmCall: { dot: "bg-blue-500", bg: "bg-blue-900/10" },
  // Control (gray)
  iterIn: { dot: "bg-gray-500", bg: "bg-gray-800/10" },
  iterOut: { dot: "bg-gray-500", bg: "bg-gray-800/10" },
  graphIn: { dot: "bg-gray-500", bg: "bg-gray-800/10" },
  graphOut: { dot: "bg-gray-500", bg: "bg-gray-800/10" },
  // Output viewers (purple)
  imageViewer: { dot: "bg-purple-500", bg: "bg-purple-900/10" },
  textViewer: { dot: "bg-purple-500", bg: "bg-purple-900/10" },
  textScroll: { dot: "bg-purple-500", bg: "bg-purple-900/10" },
  actionLog: { dot: "bg-purple-500", bg: "bg-purple-900/10" },
  metrics: { dot: "bg-purple-500", bg: "bg-purple-900/10" },
};

/** Prefix-based colors for dynamic nodeset types (matched before __). */
const PREFIX_COLORS: Record<string, NodeColor> = {
  env: { dot: "bg-green-500", bg: "bg-green-900/10" }, // Environment nodes
  policy: { dot: "bg-cyan-500", bg: "bg-cyan-900/10" }, // Policy nodes
  basic_agent: { dot: "bg-blue-400", bg: "bg-blue-900/10" }, // Agent tools
  vln_skills: { dot: "bg-teal-500", bg: "bg-teal-900/10" }, // VLN skills
  sam: { dot: "bg-pink-500", bg: "bg-pink-900/10" }, // SAM nodes
};

const DEFAULT_COLOR: NodeColor = { dot: "bg-gray-600", bg: "" };

/** Get the color for a node type. Checks exact match, then prefix before __. */
export function getNodeColor(nodeType: string): NodeColor {
  // Exact match
  if (EXACT_COLORS[nodeType]) return EXACT_COLORS[nodeType];

  // Prefix match: split on __ (nodeset convention: {nodeset}__{node})
  const parts = nodeType.split("__");
  if (parts.length >= 2 && PREFIX_COLORS[parts[0]]) {
    return PREFIX_COLORS[parts[0]];
  }

  return DEFAULT_COLOR;
}
