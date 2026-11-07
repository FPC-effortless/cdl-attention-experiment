/-!
LRC-V4: a typed algebra for parallel composition of fully materialized
relation functions. These results assert semantics only, not latency.
Toolchain verification is pending.
-/
import Relational.Basic

namespace Relational

/-- Apply a compiled relation program to many starting entities in parallel. -/
def executeMany {Entity Relation : Type}
    (edge : Relation → Entity → Entity) (path : List Relation)
    (starts : List Entity) : List Entity :=
  starts.map (compile edge path)

/-- Compiled and serial execution agree pointwise on every batch element. -/
theorem executeMany_correct {Entity Relation : Type}
    (edge : Relation → Entity → Entity) (path : List Relation)
    (starts : List Entity) :
    executeMany edge path starts = starts.map (follow edge path) := by
  simp [executeMany, compile_correct]

/-- Dynamic updates require recomputing the program against the new edge map. -/
theorem compilation_respects_edges {Entity Relation : Type}
    (oldEdge newEdge : Relation → Entity → Entity)
    (path : List Relation) (x : Entity) :
    compile newEdge path x = follow newEdge path x := by
  exact compile_correct newEdge path x

end Relational
