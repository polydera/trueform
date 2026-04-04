/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */

export {
  Primitive, PrimitiveType,
  Point, Vector, Segment, Triangle,
  Ray, Line, Plane, AABB, Polygon,
} from "./Primitive";
export {
  point, vector, segment, triangle,
  ray, line, plane, aabb, aabbFrom, polygon,
} from "./factories";
export { makeTranslation, makeRotation, makeRandomRotation } from "./transformations";
export type { Axis } from "./transformations";
