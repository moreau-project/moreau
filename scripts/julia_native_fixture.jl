# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0
# Let source/prerelease tests load the native library built by the same job.
using TOML

package, output, library = ARGS
library = realpath(library)
info = TOML.parsefile(joinpath(package, "Project.toml"))
name = "Moreau_CPU_jll"
mkpath(joinpath(output, "src"))
open(joinpath(output, "Project.toml"), "w") do io
    TOML.print(io, Dict("name" => name, "uuid" => info["deps"][name],
        "version" => first(split(info["version"], '-')) * "+0"))
end
write(joinpath(output, "src", "$name.jl"), """
module $name
const libmoreau = $(repr(library))
is_available() = isfile(libmoreau)
end
""")
