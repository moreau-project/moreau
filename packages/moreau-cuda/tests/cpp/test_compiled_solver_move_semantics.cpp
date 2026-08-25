// test_compiled_solver_move_semantics.cpp
// Pins the move semantics of CompiledSolver. The original move-assignment
// operator copied only ~23 of ~70 owned workspace pointers, leaving aliased raw
// pointers on both source and destination — double-free / use-after-free in the
// destructor. The fix is to delete the operator outright; all callers use
// unique_ptr<CompiledSolver>
// (which only requires move-construction).

#include <gtest/gtest.h>
#include <type_traits>

#include "moreau/solver/solver.hpp"

static_assert(!std::is_move_assignable_v<moreau::CompiledSolver>,
              "CompiledSolver move-assign was deleted because the "
              "original implementation moved only 23 of ~70 fields, causing "
              "double-free. unique_ptr<CompiledSolver> is the supported pattern.");
static_assert(std::is_move_constructible_v<moreau::CompiledSolver>,
              "Move-construct must remain available so unique_ptr<...> works.");
static_assert(!std::is_copy_constructible_v<moreau::CompiledSolver>,
              "CompiledSolver owns GPU resources and must not be copy-constructible.");
static_assert(!std::is_copy_assignable_v<moreau::CompiledSolver>,
              "CompiledSolver owns GPU resources and must not be copy-assignable.");

// Runtime no-op so this translation unit produces a runnable gtest binary.
TEST(CompiledSolverMoveSemantics, MoveAssignDeletedAtCompileTime) {
    SUCCEED();
}
