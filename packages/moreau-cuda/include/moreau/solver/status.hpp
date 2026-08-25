/**
 * @file status.hpp
 * @brief Solver status enumeration
 */

#pragma once

namespace moreau {

/**
 * @brief Status of solver at termination
 */
enum class SolverStatus {
    /// Problem is not solved (solver hasn't run)
    Unsolved = 0,
    /// Solver terminated with a solution
    Solved,
    /// Problem is primal infeasible. Solution returned is a certificate of primal infeasibility
    PrimalInfeasible,
    /// Problem is dual infeasible. Solution returned is a certificate of dual infeasibility
    DualInfeasible,
    /// Solver terminated with a solution (reduced accuracy)
    AlmostSolved,
    /// Problem is primal infeasible. Solution returned is a certificate of primal infeasibility (reduced accuracy)
    AlmostPrimalInfeasible,
    /// Problem is dual infeasible. Solution returned is a certificate of dual infeasibility (reduced accuracy)
    AlmostDualInfeasible,
    /// Iteration limit reached before solution or infeasibility certificate found
    MaxIterations,
    /// Time limit reached before solution or infeasibility certificate found
    MaxTime,
    /// Solver terminated with a numerical error
    NumericalError,
    /// Solver terminated due to lack of progress
    InsufficientProgress,
    /// Solver terminated due to user callback
    CallbackTerminated
};

} // namespace moreau
