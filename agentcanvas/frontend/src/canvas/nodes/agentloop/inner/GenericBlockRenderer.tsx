/** GenericBlockRenderer — universal node renderer driven by backend schema.
 *
 * Thin dispatcher that delegates to layout-specific components based on
 * ui_config.layout. All rendering logic lives in layouts/ subdirectory.
 *
 * ADR-007: BaseCanvasNode as universal base class.
 * ADR-019: Layout Component System with DisplayField.
 */

import type { NodeProps } from "@xyflow/react";
import BlockLayout from "./layouts/BlockLayout";
import StripLayout from "./layouts/StripLayout";
import ViewerLayout from "./layouts/ViewerLayout";
import ImageGridViewerLayout from "./layouts/ImageGridViewerLayout";
import NoteLayout from "./layouts/NoteLayout";
import type { NodeSchema, UIConfigSchema } from "./layouts/layoutUtils";

// Re-export types for backward compatibility
export type { NodeSchema } from "./layouts/layoutUtils";

export default function GenericBlockRenderer({ id, data }: NodeProps) {
  const schema = (data as Record<string, unknown>)._schema as
    | NodeSchema
    | undefined;
  const uiConfig = schema?.ui_config as UIConfigSchema | undefined;
  const layout = uiConfig?.layout || "block";

  switch (layout) {
    case "strip":
      return (
        <StripLayout
          id={id}
          data={data as Record<string, unknown>}
          schema={schema}
          uiConfig={uiConfig}
        />
      );
    case "viewer":
      return (
        <ViewerLayout
          id={id}
          data={data as Record<string, unknown>}
          schema={schema}
        />
      );
    case "imageGrid":
      return (
        <ImageGridViewerLayout
          id={id}
          data={data as Record<string, unknown>}
          schema={schema}
        />
      );
    case "note":
      return (
        <NoteLayout
          id={id}
          data={data as Record<string, unknown>}
          schema={schema}
        />
      );
    default:
      return (
        <BlockLayout
          id={id}
          data={data as Record<string, unknown>}
          schema={schema}
          uiConfig={uiConfig}
        />
      );
  }
}
