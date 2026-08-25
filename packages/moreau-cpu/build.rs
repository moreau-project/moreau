fn main() {
    // Tell cargo to check the sdp_pyblas cfg flag (suppress warnings)
    // Using single-colon for Rust 1.70+ compatibility
    println!("cargo:rustc-check-cfg=cfg(sdp_pyblas)");

    // When building with the "python" feature, enable sdp_pyblas
    // which uses scipy's BLAS/LAPACK at runtime instead of linking
    // against blas-src/lapack-src
    if std::env::var("CARGO_FEATURE_PYTHON").is_ok() {
        println!("cargo:rustc-cfg=sdp_pyblas");
    }

    build_active_set();
}

// Build C++ active-set solver via cmake (only with active-set feature).
#[cfg(feature = "active-set")]
fn build_active_set() {
    let dst = cmake::Config::new("cpp")
        .define("MOREAU_BUILD_TESTS", "OFF")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=moreau_active_set");

    // Link C++ standard library (platform-specific)
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if cfg!(target_os = "linux") || cfg!(target_os = "freebsd") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    // Windows/MSVC: no explicit C++ runtime link needed (handled by cmake)
}

#[cfg(not(feature = "active-set"))]
fn build_active_set() {}
