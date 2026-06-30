/** Access-grant edge — dashed line representing a node's read/write grant on a state container.
 *
 * Access grants are NOT wires. They carry no data, do not trigger firing;
 * they grant a node read/write access to a container's states. Rendered as
 * a dashed violet/gray line with no arrowhead, separate from the data wire
 * system. See ADR-026.
 */

import { memo } from "react";
import { BaseEdge, getStraightPath } from "@xyflow/react";
import type { EdgeProps } from "@xyflow/react";

function AccessGrantEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  style = {},
}: EdgeProps) {
  const [edgePath] = getStraightPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
  });

  return (
    <BaseEdge
      id={id}
      path={edgePath}
      style={{
        strokeDasharray: "6 4",
        stroke: "#8b5cf6",
        strokeWidth: 1.5,
        opacity: 0.5,
        ...style,
      }}
    />
  );
}

export default memo(AccessGrantEdge);
