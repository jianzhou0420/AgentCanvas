/** Note layout — sticky-note billboard rendering markdown content.
 *
 * Pure annotation node: no port handles, no execution side-effects.
 * Backend: ``NoteNode`` (builtin_nodes.py).
 * Edit the ``markdown`` field via the PropertiesPanel; this component
 * just renders it inline on the canvas body.
 */

import ReactMarkdown from "react-markdown";
import type { NodeSchema } from "./layoutUtils";

interface NoteLayoutProps {
  id: string;
  data: Record<string, unknown>;
  schema: NodeSchema | undefined;
}

export default function NoteLayout({ data }: NoteLayoutProps) {
  const markdown =
    (data.markdown as string) ||
    "_(empty note — set the markdown field in the right panel)_";

  return (
    <div
      className="rounded border border-amber-300 bg-amber-50 px-3 py-2 shadow-sm"
      style={{
        minWidth: 200,
        maxWidth: 400,
        minHeight: 60,
      }}
    >
      <div className="prose prose-sm max-w-none text-amber-950 [&_a]:text-amber-700 [&_a]:underline [&_code]:rounded [&_code]:bg-amber-100 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-[0.85em] [&_h1]:mb-1 [&_h1]:mt-0 [&_h1]:text-base [&_h1]:font-semibold [&_h2]:mb-1 [&_h2]:mt-1 [&_h2]:text-sm [&_h2]:font-semibold [&_h3]:mb-1 [&_h3]:mt-1 [&_h3]:text-xs [&_h3]:font-semibold [&_li]:my-0 [&_p]:my-1 [&_pre]:overflow-x-auto [&_pre]:rounded [&_pre]:bg-amber-100 [&_pre]:p-2 [&_ul]:my-1 [&_ul]:pl-5 [&_ol]:my-1 [&_ol]:pl-5">
        <ReactMarkdown
          components={{
            a: ({ ...props }) => (
              <a {...props} target="_blank" rel="noopener noreferrer" />
            ),
          }}
        >
          {markdown}
        </ReactMarkdown>
      </div>
    </div>
  );
}
