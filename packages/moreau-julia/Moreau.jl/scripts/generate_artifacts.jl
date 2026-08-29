using Base.BinaryPlatforms
using Pkg.Artifacts: bind_artifact!, create_artifact
using SHA

function bind_release_artifact!(
    artifacts_toml::String,
    artifact_name::String,
    tarball::String,
    url::String,
    platform::Platform;
    lazy::Bool=false,
)
    sha256 = bytes2hex(open(SHA.sha256, tarball))
    hash = create_artifact() do directory
        run(`tar xzf $tarball -C $directory`)
    end
    bind_artifact!(
        artifacts_toml,
        artifact_name,
        hash;
        platform=platform,
        download_info=[(url, sha256)],
        lazy=lazy,
        force=true,
    )
    return
end

function main(args::Vector{String})
    if length(args) != 2
        println(stderr, "usage: julia scripts/generate_artifacts.jl RELEASE_TAG DIST_DIR")
        return 2
    end

    release_tag, dist = args
    artifacts_toml = normpath(joinpath(@__DIR__, "..", "Artifacts.toml"))
    release_url = "https://github.com/moreau-project/moreau/releases/download/$release_tag"
    rm(artifacts_toml; force=true)

    artifacts = [
        ("moreau_cpu", "moreau-cpu-linux-x86_64.tar.gz", Platform("x86_64", "linux"; libc="glibc"), false),
        ("moreau_cpu", "moreau-cpu-linux-aarch64.tar.gz", Platform("aarch64", "linux"; libc="glibc"), false),
        ("moreau_cpu", "moreau-cpu-macos-arm64.tar.gz", Platform("aarch64", "macos"), false),
        ("moreau_cuda12", "moreau-cuda12-linux-x86_64.tar.gz", Platform("x86_64", "linux"; libc="glibc"), true),
        ("moreau_cuda12", "moreau-cuda12-linux-aarch64.tar.gz", Platform("aarch64", "linux"; libc="glibc"), true),
        ("moreau_cuda13", "moreau-cuda13-linux-x86_64.tar.gz", Platform("x86_64", "linux"; libc="glibc"), true),
        ("moreau_cuda13", "moreau-cuda13-linux-aarch64.tar.gz", Platform("aarch64", "linux"; libc="glibc"), true),
    ]

    cpu_count = 0
    for (name, filename, platform, lazy) in artifacts
        tarball = joinpath(dist, filename)
        if !isfile(tarball)
            @info "Skipping unavailable artifact" filename
            continue
        end
        bind_release_artifact!(
            artifacts_toml,
            name,
            tarball,
            "$release_url/$filename",
            platform;
            lazy=lazy,
        )
        cpu_count += name == "moreau_cpu"
    end

    cpu_count > 0 || error("No CPU artifacts found in $dist")
    print(read(artifacts_toml, String))
    return 0
end

exit(main(ARGS))
