import moreau.__main__ as moreau_main


def test_diagnostics_cpu_solver_ok():
    ok, msg = moreau_main.test_cpu_solver()
    assert ok, f"expected diagnostic CPU solver check to succeed, got: {msg}"
