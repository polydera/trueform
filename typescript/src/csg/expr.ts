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

const OR = -1;
const AND = -2;
const SUB = -3;
const NOT = -4;

function asExpr(value: Expr | number): Expr {
  if (value instanceof Expr) return value;
  if (Number.isInteger(value)) return op(value);
  throw new TypeError(`expected op(i) expression or operand index, got ${value}`);
}

/**
 * A boolean expression tree over operand indices.
 *
 * Build leaves with `op(i)` and combine with the builder methods:
 * `.or()` (union), `.and()` (intersection), `.sub()` (difference),
 * `.not()` (complement). Plain integers are accepted as operands:
 * `op(0).sub(1)` is the difference of operand 0 and operand 1.
 *
 * @example
 * const carved = op(0).sub(op(1)).sub(op(2));
 * const shared = op(0).and(op(1));
 * const outside = op(0).or(op(1)).not();
 */
export class Expr {
  /** @internal */
  readonly _program: number[];

  /** @internal */
  constructor(program: number[]) {
    this._program = program;
  }

  /** The flat postfix program consumed by the native layer. */
  program(): number[] {
    return [...this._program];
  }

  /** Union with `other`. */
  or(other: Expr | number): Expr {
    return new Expr([...this._program, ...asExpr(other)._program, OR]);
  }

  /** Intersection with `other`. */
  and(other: Expr | number): Expr {
    return new Expr([...this._program, ...asExpr(other)._program, AND]);
  }

  /** Difference: this minus `other`. */
  sub(other: Expr | number): Expr {
    return new Expr([...this._program, ...asExpr(other)._program, SUB]);
  }

  /** Complement of this expression. */
  not(): Expr {
    return new Expr([...this._program, NOT]);
  }
}

/**
 * A leaf expression selecting operand `operandId` (its index in the
 * CsgGraph's form list).
 */
export function op(operandId: number): Expr {
  if (!Number.isInteger(operandId) || operandId < 0) {
    throw new RangeError(`operand index must be a non-negative integer, got ${operandId}`);
  }
  return new Expr([operandId]);
}

/** @internal */
export function programOf(expr: Expr | number): number[] {
  return asExpr(expr).program();
}
