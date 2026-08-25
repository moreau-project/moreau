/**
 * @file test_c_api_input_validation.cpp
 * @brief Defense-in-depth tests for the Moreau CUDA C API cone-validation.
 *
 * Malformed `moreau_cones_t` inputs (negative SOC dim, alpha outside (0, 1),
 * NaN alpha, etc.) previously reached
 * `convert_cones` and downstream cone-projection code, where they would
 * trigger device-side aborts or undefined behaviour. The Python wrapper
 * validates these at a higher level, but raw C/C++/Julia callers bypass
 * that. These tests verify that `moreau_solver_create` rejects every
 * malformed input with `MOREAU_ERROR_INVALID_ARGUMENT` before allocating
 * any solver state.
 *
 * Bidirectional check: with the validation patch removed, the negative
 * tests below would either crash the process (allocation panic for
 * negative-cast-to-huge SOC dim) or pass an out-of-range alpha downstream
 * causing UB, instead of returning the expected error code.
 *
 * Note: these tests only exercise the create() path — they never call
 * setup()/solve(), so they don't require GPU compute.
 * `moreau_solver_create` does construct the underlying CompiledSolver, so
 * a CUDA-capable device is still required for the test to run.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <limits>

extern "C" {
#include "moreau.h"
}

class CApiInputValidationTest : public ::testing::Test {
protected:
    // Trivial 2-variable problem skeleton; cones is what we vary.
    static constexpr int64_t n = 2;
    int64_t P_ro[3] = {0, 1, 2};
    int64_t P_ci[2] = {0, 1};
    static constexpr int64_t nnz_P = 2;

    moreau_settings_t settings;

    void SetUp() override {
        moreau_settings_default(&settings);
        settings.verbose = 0;
    }

    /// Build A row offsets/col indices for an `m`-row problem with one
    /// nonzero per row at column 0.
    moreau_error_t try_create(const moreau_cones_t& cones, int64_t m,
                              moreau_solver_t** solver_out) {
        std::vector<int64_t> A_ro(m + 1);
        for (int64_t i = 0; i <= m; ++i) A_ro[i] = i;
        std::vector<int64_t> A_ci(m, 0);

        return moreau_solver_create(
            solver_out, n, m,
            P_ro, P_ci, nnz_P,
            A_ro.data(),
            (m > 0) ? A_ci.data() : nullptr,
            m,
            &cones, &settings);
    }
};

// -- SOC dim validation -------------------------------------------------------

TEST_F(CApiInputValidationTest, NegativeSocDimRejected) {
    int64_t bad_dims[1] = {-3};
    moreau_cones_t cones = {};
    cones.num_soc_cones = 1;
    cones.soc_dims = bad_dims;

    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 3, &solver);
    EXPECT_EQ(rc, MOREAU_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(solver, nullptr);
    if (solver) moreau_solver_destroy(solver);
}

TEST_F(CApiInputValidationTest, SocDimOneRejected) {
    int64_t bad_dims[1] = {1};
    moreau_cones_t cones = {};
    cones.num_soc_cones = 1;
    cones.soc_dims = bad_dims;

    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 1, &solver);
    EXPECT_EQ(rc, MOREAU_ERROR_INVALID_ARGUMENT);
    if (solver) moreau_solver_destroy(solver);
}

TEST_F(CApiInputValidationTest, SocDimsNullWithCountRejected) {
    moreau_cones_t cones = {};
    cones.num_soc_cones = 1;
    cones.soc_dims = nullptr;

    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 0, &solver);
    EXPECT_EQ(rc, MOREAU_ERROR_INVALID_ARGUMENT);
    if (solver) moreau_solver_destroy(solver);
}

TEST_F(CApiInputValidationTest, NegativeNumSocConesRejected) {
    moreau_cones_t cones = {};
    cones.num_soc_cones = -1;

    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 0, &solver);
    EXPECT_EQ(rc, MOREAU_ERROR_INVALID_ARGUMENT);
    if (solver) moreau_solver_destroy(solver);
}

// -- Power-cone alpha validation ---------------------------------------------

TEST_F(CApiInputValidationTest, AlphaZeroRejected) {
    double alphas[1] = {0.0};
    moreau_cones_t cones = {};
    cones.num_power_cones = 1;
    cones.power_alphas = alphas;

    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 3, &solver);
    EXPECT_EQ(rc, MOREAU_ERROR_INVALID_ARGUMENT);
    if (solver) moreau_solver_destroy(solver);
}

TEST_F(CApiInputValidationTest, AlphaOneRejected) {
    double alphas[1] = {1.0};
    moreau_cones_t cones = {};
    cones.num_power_cones = 1;
    cones.power_alphas = alphas;

    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 3, &solver);
    EXPECT_EQ(rc, MOREAU_ERROR_INVALID_ARGUMENT);
    if (solver) moreau_solver_destroy(solver);
}

TEST_F(CApiInputValidationTest, AlphaNegativeRejected) {
    double alphas[1] = {-0.5};
    moreau_cones_t cones = {};
    cones.num_power_cones = 1;
    cones.power_alphas = alphas;

    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 3, &solver);
    EXPECT_EQ(rc, MOREAU_ERROR_INVALID_ARGUMENT);
    if (solver) moreau_solver_destroy(solver);
}

TEST_F(CApiInputValidationTest, AlphaNanRejected) {
    double alphas[1] = {std::numeric_limits<double>::quiet_NaN()};
    moreau_cones_t cones = {};
    cones.num_power_cones = 1;
    cones.power_alphas = alphas;

    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 3, &solver);
    EXPECT_EQ(rc, MOREAU_ERROR_INVALID_ARGUMENT);
    if (solver) moreau_solver_destroy(solver);
}

TEST_F(CApiInputValidationTest, AlphaInfRejected) {
    double alphas[1] = {std::numeric_limits<double>::infinity()};
    moreau_cones_t cones = {};
    cones.num_power_cones = 1;
    cones.power_alphas = alphas;

    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 3, &solver);
    EXPECT_EQ(rc, MOREAU_ERROR_INVALID_ARGUMENT);
    if (solver) moreau_solver_destroy(solver);
}

TEST_F(CApiInputValidationTest, PowerAlphasNullWithCountRejected) {
    moreau_cones_t cones = {};
    cones.num_power_cones = 1;
    cones.power_alphas = nullptr;

    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 0, &solver);
    EXPECT_EQ(rc, MOREAU_ERROR_INVALID_ARGUMENT);
    if (solver) moreau_solver_destroy(solver);
}

// -- Other count validations --------------------------------------------------

TEST_F(CApiInputValidationTest, NegativeNumZeroConesRejected) {
    moreau_cones_t cones = {};
    cones.num_zero_cones = -1;

    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 0, &solver);
    EXPECT_EQ(rc, MOREAU_ERROR_INVALID_ARGUMENT);
    if (solver) moreau_solver_destroy(solver);
}

TEST_F(CApiInputValidationTest, NegativeNumNonnegConesRejected) {
    moreau_cones_t cones = {};
    cones.num_nonneg_cones = -1;

    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 0, &solver);
    EXPECT_EQ(rc, MOREAU_ERROR_INVALID_ARGUMENT);
    if (solver) moreau_solver_destroy(solver);
}

TEST_F(CApiInputValidationTest, NegativeNumExpConesRejected) {
    moreau_cones_t cones = {};
    cones.num_exp_cones = -1;

    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 0, &solver);
    EXPECT_EQ(rc, MOREAU_ERROR_INVALID_ARGUMENT);
    if (solver) moreau_solver_destroy(solver);
}

TEST_F(CApiInputValidationTest, NegativeNumPowerConesRejected) {
    moreau_cones_t cones = {};
    cones.num_power_cones = -1;

    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 0, &solver);
    EXPECT_EQ(rc, MOREAU_ERROR_INVALID_ARGUMENT);
    if (solver) moreau_solver_destroy(solver);
}

// -- Positive control: valid input still accepted ----------------------------

TEST_F(CApiInputValidationTest, ValidInputsAccepted) {
    int64_t dims[1] = {3};
    double alphas[1] = {0.5};
    moreau_cones_t cones = {};
    cones.num_nonneg_cones = 1;
    cones.num_soc_cones = 1;
    cones.soc_dims = dims;
    cones.num_power_cones = 1;
    cones.power_alphas = alphas;

    // m = 1 (nonneg) + 3 (SOC) + 3 (power) = 7
    moreau_solver_t* solver = nullptr;
    auto rc = try_create(cones, 7, &solver);
    EXPECT_EQ(rc, MOREAU_OK);
    EXPECT_NE(solver, nullptr);
    if (solver) moreau_solver_destroy(solver);
}
