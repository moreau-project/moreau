use std::io::Write;

/// Print the Moreau banner.
pub(crate) fn print_banner(
    out: &mut dyn Write,
    is_verbose: bool,
    _subtitle: &str,
) -> std::io::Result<()> {
    if !is_verbose {
        return std::io::Result::Ok(());
    }

    writeln!(
        out,
        "┌─────────────────────────────────────────────────────────────┐"
    )?;
    writeln!(
        out,
        "│             Moreau v{}  ─  Conic Solver                  │",
        crate::VERSION
    )?;
    #[cfg(debug_assertions)]
    writeln!(
        out,
        "│                    ⚠ debug build                            │",
    )?;
    #[cfg(not(debug_assertions))]
    writeln!(
        out,
        "│                                                             │"
    )?;
    writeln!(
        out,
        "└─────────────────────────────────────────────────────────────┘"
    )?;
    std::io::Result::Ok(())
}
