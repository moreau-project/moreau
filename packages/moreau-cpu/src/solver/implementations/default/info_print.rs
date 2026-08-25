#[cfg(feature = "sdp")]
use crate::solver::chordal::ChordalInfo;

use crate::io::{ConfigurablePrintTarget, PrintTarget};
use crate::{
    algebra::*,
    solver::core::cones::{SupportedConeAsTag, SupportedConeTag},
};
use std::io::Write;

use super::*;
use crate::solver::core::{
    cones::{CompositeCone, Cone},
    traits::InfoPrint,
};
use crate::solver::SolverStatus;
use std::time::Duration;

impl<T> ConfigurablePrintTarget for DefaultInfo<T> {
    fn print_to_stdout(&mut self) {
        self.stream.print_to_stdout()
    }
    fn print_to_file(&mut self, file: std::fs::File) {
        self.stream.print_to_file(file)
    }
    fn print_to_stream(&mut self, stream: Box<dyn Write + Send + Sync>) {
        self.stream.print_to_stream(stream)
    }
    fn print_to_sink(&mut self) {
        self.stream.print_to_sink()
    }
    fn print_to_buffer(&mut self) {
        self.stream.print_to_buffer()
    }
    fn get_print_buffer(&mut self) -> std::io::Result<String> {
        self.stream.get_print_buffer()
    }
}

macro_rules! expformat {
    ($fmt:expr,$val:expr) => {
        if $val.is_finite() {
            _exp_str_reformat(format!($fmt, $val))
        } else {
            format!($fmt, $val)
        }
    };
}

impl<T> InfoPrint<T> for DefaultInfo<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type C = CompositeCone<T>;
    type SE = DefaultSettings<T>;

    fn print_configuration(
        &mut self,
        settings: &DefaultSettings<T>,
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
    ) -> std::io::Result<()> {
        if !settings.verbose {
            return std::io::Result::Ok(());
        }

        let out = &mut self.stream;

        if let Some(ref presolver) = data.presolver {
            writeln!(
                out,
                "\npresolve: removed {} constraints",
                presolver.count_reduced()
            )?;
        }

        #[cfg(feature = "sdp")]
        if let Some(ref chordal_info) = data.chordal_info {
            print_chordal_decomposition(out, chordal_info, settings)?;
        }

        writeln!(out, "\nproblem:")?;
        writeln!(out, "  ├ variables     = {}", data.n)?;
        writeln!(out, "  ├ constraints   = {}", data.m)?;
        writeln!(out, "  ├ nnz(P)        = {}", data.P.nnz())?;
        writeln!(out, "  ├ nnz(A)        = {}", data.A.nnz())?;
        writeln!(out, "  └ cones (total) = {}", cones.len())?;

        _print_conedims_by_type(out, cones, SupportedConeTag::ZeroCone)?;
        _print_conedims_by_type(out, cones, SupportedConeTag::NonnegativeCone)?;
        _print_conedims_by_type(out, cones, SupportedConeTag::SecondOrderCone)?;
        _print_conedims_by_type(out, cones, SupportedConeTag::ExponentialCone)?;
        _print_conedims_by_type(out, cones, SupportedConeTag::PowerCone)?;
        _print_conedims_by_type(out, cones, SupportedConeTag::GenPowerCone)?;
        #[cfg(feature = "sdp")]
        _print_conedims_by_type(out, cones, SupportedConeTag::PSDTriangleCone)?;

        writeln!(out)?;

        self.print_settings(settings)?;

        std::io::Result::Ok(())
    }

    fn print_status_header(&mut self, settings: &DefaultSettings<T>) -> std::io::Result<()> {
        if !settings.verbose {
            return std::io::Result::Ok(());
        }

        let out = &mut self.stream;

        write!(out, "iter    ")?;
        write!(out, "pcost        ")?;
        write!(out, "dcost       ")?;
        write!(out, "gap       ")?;
        write!(out, "pres      ")?;
        write!(out, "dres      ")?;
        write!(out, "k/t       ")?;
        write!(out, " μ       ")?;
        write!(out, "step      ")?;
        writeln!(out)?;
        writeln!(
            out,
            "─────────────────────────────────────────────────────────────────────────────────────────────"
        )?;
        out.flush()?;
        std::io::Result::Ok(())
    }

    fn print_status(&mut self, settings: &DefaultSettings<T>) -> std::io::Result<()> {
        if !settings.verbose {
            return std::io::Result::Ok(());
        }

        let out = &mut self.stream;

        write!(out, "{:>3}  ", self.iterations)?;
        write!(out, "{}  ", expformat!("{:+8.4e}", self.cost_primal))?;
        write!(out, "{}  ", expformat!("{:+8.4e}", self.cost_dual))?;
        let gapprint = T::min(self.gap_abs, self.gap_rel);
        write!(out, "{}  ", expformat!("{:6.2e}", gapprint))?;
        write!(out, "{}  ", expformat!("{:6.2e}", self.res_primal))?;
        write!(out, "{}  ", expformat!("{:6.2e}", self.res_dual))?;
        write!(out, "{}  ", expformat!("{:6.2e}", self.ktratio))?;
        write!(out, "{}  ", expformat!("{:6.2e}", self.mu))?;

        if self.iterations > 0 {
            write!(out, "{}  ", expformat!("{:>.2e}", self.step_length))?;
        } else {
            write!(out, " ------   ")?;
        }

        writeln!(out)?;

        std::io::Result::Ok(())
    }

    fn print_footer(&mut self, settings: &DefaultSettings<T>) -> std::io::Result<()> {
        if !settings.verbose {
            return std::io::Result::Ok(());
        }

        let out = &mut self.stream;

        writeln!(
            out,
            "─────────────────────────────────────────────────────────────────────────────────────────────"
        )?;

        // Status with indicator
        let status_indicator = match self.status {
            SolverStatus::Solved => "✓",
            SolverStatus::AlmostSolved => "≈",
            SolverStatus::PrimalInfeasible
            | SolverStatus::DualInfeasible
            | SolverStatus::AlmostPrimalInfeasible
            | SolverStatus::AlmostDualInfeasible => "∅",
            SolverStatus::MaxIterations | SolverStatus::MaxTime => "⏱",
            SolverStatus::NumericalError | SolverStatus::InsufficientProgress => "✗",
            _ => "•",
        };

        writeln!(
            out,
            "{} Terminated with status = {}",
            status_indicator, self.status
        )?;

        writeln!(
            out,
            "solve time = {:?}",
            Duration::from_secs_f64(self.solve_time)
        )?;

        std::io::Result::Ok(())
    }

    fn print_target(&mut self) -> &mut dyn std::io::Write {
        &mut self.stream
    }
}

impl<T> DefaultInfo<T>
where
    T: FloatT,
{
    fn print_settings(&mut self, settings: &DefaultSettings<T>) -> std::io::Result<()> {
        let out = &mut self.stream;

        let set = settings;

        writeln!(out, "settings:")?;

        write!(out, "  ├ linear algebra: ")?;
        if self.linsolver.direct {
            write!(out, "direct / {}, ", self.linsolver.name)?;
        } else {
            write!(out, "indirect / {}, ", self.linsolver.name)?;
        }
        print_nthreads(out, self.linsolver.threads)?;
        writeln!(out)?;

        writeln!(
            out,
            "  ├ max iter = {}, time limit = {:.1}s, max step = {:.3}",
            set.max_iter,
            if set.time_limit.is_infinite() {
                f64::INFINITY
            } else {
                set.time_limit
            },
            set.ipm.max_step_fraction
        )?;
        writeln!(
            out,
            "  └ tol_feas = {:.1e}, tol_gap_abs = {:.1e}, tol_gap_rel = {:.1e}",
            set.ipm.tol_feas, set.ipm.tol_gap_abs, set.ipm.tol_gap_rel
        )?;

        writeln!(out)?;

        std::io::Result::Ok(())
    }
}

fn _bool_on_off(v: bool) -> &'static str {
    match v {
        true => "on",
        false => "off",
    }
}

fn print_nthreads(out: &mut PrintTarget, nthreads: usize) -> std::io::Result<()> {
    match nthreads {
        0 => Ok(()),
        1 => write!(out, "(1 thread)"),
        _ => write!(out, "({nthreads} threads)"),
    }
}

#[cfg(feature = "sdp")]
fn print_chordal_decomposition<T: FloatT>(
    out: &mut PrintTarget,
    chordal_info: &ChordalInfo<T>,
    settings: &DefaultSettings<T>,
) -> std::io::Result<()> {
    writeln!(out, "\nchordal decomposition:")?;
    writeln!(
        out,
        "  compact format = {}, dual completion = {}",
        _bool_on_off(settings.ipm.chordal_decomposition_compact),
        _bool_on_off(settings.ipm.chordal_decomposition_complete_dual)
    )?;

    writeln!(
        out,
        "  merge method = {}",
        settings.ipm.chordal_decomposition_merge_method
    )?;

    writeln!(
        out,
        "  PSD cones initial             = {}",
        chordal_info.init_psd_cone_count()
    )?;

    writeln!(
        out,
        "  PSD cones decomposable        = {}",
        chordal_info.decomposable_cone_count()
    )?;

    writeln!(
        out,
        "  PSD cones after decomposition = {}",
        chordal_info.premerge_psd_cone_count()
    )?;

    writeln!(
        out,
        "  PSD cones after merges        = {}",
        chordal_info.final_psd_cone_count()
    )?;

    std::io::Result::Ok(())
}

fn _get_precision_string<T: FloatT>() -> String {
    (::std::mem::size_of::<T>() * 8).to_string()
}

fn _print_conedims_by_type<T: FloatT>(
    out: &mut PrintTarget,
    cones: &CompositeCone<T>,
    conetag: SupportedConeTag,
) -> std::io::Result<()> {
    let maxlistlen = 5;

    let count = cones.get_type_count(conetag);

    if count == 0 {
        return std::io::Result::Ok(());
    }

    let name = conetag.as_str();
    let name = &name[0..name.len() - 4];
    let name = format!("{name:>11}");

    let mut nvars = Vec::with_capacity(count);
    for cone in cones.iter() {
        if cone.as_tag() == conetag {
            nvars.push(cone.numel());
        }
    }
    write!(out, "    : {name} = {count}, ")?;

    if count == 1 {
        write!(out, " numel = {}", nvars[0])?;
    } else if count <= maxlistlen {
        write!(out, " numel = (")?;
        for nvar in nvars.iter().take(nvars.len() - 1) {
            write!(out, "{nvar},")?;
        }
        write!(out, "{})", nvars[nvars.len() - 1])?;
    } else {
        write!(out, " numel = (")?;
        for nvar in nvars.iter().take(maxlistlen - 1) {
            write!(out, "{nvar},")?;
        }
        write!(out, "...,{})", nvars[nvars.len() - 1])?;
    }

    writeln!(out)?;

    std::io::Result::Ok(())
}

// convert a string in LowerExp display format into one that
// 1) always has a sign after the exponent, and
// 2) has at least two digits in the exponent.
// This matches the Julia output formatting.

fn _exp_str_reformat(mut thestr: String) -> String {
    let eidx = thestr.find('e').unwrap();
    let has_sign = thestr.chars().nth(eidx + 1).unwrap() == '-';

    let has_short_exp = {
        if !has_sign {
            thestr.len() == eidx + 2
        } else {
            thestr.len() == eidx + 3
        }
    };

    let chars;
    if !has_sign {
        if has_short_exp {
            chars = "+0";
        } else {
            chars = "+";
        }
    } else if has_short_exp {
        chars = "0";
    } else {
        chars = "";
    }

    let shift = if has_sign { 2 } else { 1 };
    thestr.insert_str(eidx + shift, chars);
    thestr
}
