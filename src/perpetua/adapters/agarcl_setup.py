"""Build AgarCL's pinned native runtime for ``perpetua.adapters.agarcl``.

AgarCL needs clang, CMake and GLFW and declares Gymnasium 0.29, so it is built
in a separate Python 3.12 environment rather than Perpetua's. The tested build
uses CMake 3.31, a local GLFW 3.4 (null platform, rendering through EGL on
Linux) and ``pip --no-deps``; no sudo, Homebrew or system package installation
is performed. On Linux, missing Clang/OpenGL files are extracted locally from
apt packages (see ``linux_toolchain``). Network access is required for the
source clones and the build's own Google Benchmark/GoogleTest download.

    python -m perpetua.adapters.agarcl_setup [--root DIR]

Override the adapter with ``PERPETUA_AGARCL_PYTHON`` and
``PERPETUA_AGARCL_SOURCE``.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from perpetua.adapters.agarcl import AGARCL_COMMIT, DEFAULT_ROOT, GLFW_COMMIT

__all__ = ["main"]

AGARCL_REPOSITORY = "https://github.com/AgarCL/AgarCL.git"
GLFW_REPOSITORY = "https://github.com/glfw/glfw.git"
REQUIREMENTS = (
    "gymnasium==1.4.0",
    "numpy==2.2.5",
    "opencv-python-headless==4.11.0.86",
    "cmake==3.31.10",
    "setuptools",
    "packaging",
)
# Debian multiarch directory of the extracted packages, e.g. x86_64-linux-gnu.
MULTIARCH = f"{platform.machine()}-linux-gnu"
GL_PACKAGES = (
    "libgl-dev",
    "libegl-dev",
    "libgles-dev",
    "libglx-dev",
    "libglu1-mesa-dev",
    "libglu1-mesa",
    "libegl1",
    "libgl1",
    "libglx0",
    # libGLU links against libOpenGL.so.0 (absent on e.g. stock Ubuntu 24.04).
    "libopengl0",
    "libopengl-dev",
)
GL_HEADERS = ("GL/gl.h", "GL/glu.h", "EGL/egl.h")
GL_LIBRARIES = ("libGL.so", "libGLU.so", "libEGL.so", "libGLX.so", "libOpenGL.so")


def run(*command: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def usable(python: Path) -> bool:
    """Whether an environment's interpreter is there and starts."""
    try:
        return subprocess.run([str(python), "-c", ""], capture_output=True).returncode == 0
    except OSError:
        return False


def ensure_venv(venv: Path) -> None:
    """Create the runtime's environment, replacing one whose interpreter no longer runs.

    An environment outlives the Python it was made from (a removed or moved
    uv-managed interpreter leaves ``bin/python`` dangling), and ``uv venv``
    refuses an existing directory without ``--clear``.
    """
    if not usable(venv / "bin" / "python"):
        run("uv", "venv", "--quiet", "--clear", "--python", "3.12", str(venv))


def system_gl() -> bool:
    """Whether OpenGL/GLU/EGL development files are already installed."""
    library = Path("/usr/lib") / MULTIARCH
    return all((Path("/usr/include") / header).is_file() for header in GL_HEADERS) and all(
        (library / name).exists() for name in GL_LIBRARIES
    )


def apt(*arguments: str) -> str:
    return subprocess.run(arguments, check=True, capture_output=True, text=True).stdout


def default_clang() -> int:
    """The distribution's default Clang (what ``clang`` depends on), else the newest."""
    for line in apt("apt-cache", "depends", "clang").splitlines():
        name = line.strip().removeprefix("Depends: ")
        if name.startswith("clang-") and name.removeprefix("clang-").isdigit():
            return int(name.removeprefix("clang-"))
    listing = apt("apt-cache", "search", "--names-only", "^clang-[0-9]+$")
    versions = [int(line.split()[0].removeprefix("clang-")) for line in listing.splitlines()]
    if not versions:
        raise RuntimeError("apt knows no clang-N package; run `apt-get update` or set CXX.")
    return max(versions)


def missing_closure(packages: list[str]) -> list[str]:
    """The packages plus their dependency closure, minus what is already installed."""
    listing = apt(
        "apt-cache",
        "depends",
        "--recurse",
        "--no-recommends",
        "--no-suggests",
        "--no-conflicts",
        "--no-breaks",
        "--no-replaces",
        "--no-enhances",
        *packages,
    )
    # Unindented lines are package names; "<name>" marks virtual packages.
    names = {
        line.split(":")[0]
        for line in listing.splitlines()
        if line and not line[0].isspace() and not line.startswith("<")
    }
    status = apt("dpkg-query", "-W", "-f", "${Package} ${db:Status-Abbrev}\n")
    installed = {parts[0] for parts in map(str.split, status.splitlines()) if parts[1:2] == ["ii"]}
    return sorted(names - installed)


def link_installed_libraries(prefix: Path) -> None:
    """Point extracted ``-dev`` symlinks at runtime libraries that are installed.

    A ``-dev`` package's ``libGL.so -> libGL.so.1`` is relative; when the
    runtime package is already installed it is not extracted, so the link would
    dangle inside ``prefix``.
    """
    local, system = prefix / "usr" / "lib" / MULTIARCH, Path("/usr/lib") / MULTIARCH
    for link in local.glob("*.so*"):
        if link.is_symlink() and not link.exists():
            target = system / os.readlink(link)
            if target.exists():
                link.unlink()
                link.symlink_to(target)


def linux_toolchain(root: Path, prefix: Path, venv: Path) -> dict[str, str]:
    """Find or locally extract Clang and OpenGL files; never install system packages.

    AgarCL's sources need Clang (GCC 11 rejects them). Existing system files are
    preferred; anything missing is fetched, with its uninstalled dependency
    closure, by ``apt-get download`` and extracted into ``prefix``. A C/C++ base
    toolchain (libc and libstdc++ headers, a linker), apt package lists and, on
    machines without a GPU driver, Mesa's EGL driver must already be present.
    """
    compiler = os.environ.get("CXX") or shutil.which("clang++")
    packages: list[str] = []
    gl_local = not system_gl()
    if gl_local:
        packages += GL_PACKAGES
    if compiler is None:
        version = default_clang()
        packages += [
            f"clang-{version}",
            f"libclang-common-{version}-dev",
            f"libclang-cpp{version}",
            f"libllvm{version}",
            f"libclang1-{version}",
        ]
        compiler = str(prefix / "usr" / "bin" / f"clang++-{version}")
    if packages:
        if not shutil.which("apt-get") or not shutil.which("dpkg"):
            raise RuntimeError(
                f"Missing {', '.join(packages)}; without apt-get/dpkg, install Clang and "
                "OpenGL/GLU/EGL development files yourself (or set CXX)."
            )
        debs = root / "debs"
        shutil.rmtree(debs, ignore_errors=True)
        debs.mkdir()
        run("apt-get", "download", *missing_closure(packages), cwd=debs)
        for package in sorted(debs.glob("*.deb")):
            run("dpkg", "-x", str(package), str(prefix))
        link_installed_libraries(prefix)
    environment = {
        "CXX": compiler,
        # AgarCL's CMake does not link GLU explicitly on Linux.
        "LDFLAGS": f"{os.environ.get('LDFLAGS', '')} -Wl,--no-as-needed -lGLU".strip(),
    }
    if gl_local:
        cmake_wrapper(venv, prefix)
        library = str(prefix / "usr" / "lib" / MULTIARCH)
        environment |= {
            "CPATH": str(prefix / "usr" / "include"),
            "LIBRARY_PATH": library,
            "LD_LIBRARY_PATH": library,
        }
    return environment


def cmake_wrapper(venv: Path, prefix: Path) -> None:
    """Inject the private OpenGL paths into setup.py's CMake invocations."""
    real = venv / "bin" / "cmake-real"
    if not real.exists():
        (venv / "bin" / "cmake").rename(real)
    library = prefix / "usr" / "lib" / MULTIARCH
    include = prefix / "usr" / "include"
    (venv / "bin" / "cmake").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--build" ]; then\n'
        f'  exec "{real}" "$@"\n'
        "fi\n"
        f'exec "{real}" '
        f'-DOPENGL_INCLUDE_DIR="{include}" '
        f'-DOPENGL_opengl_LIBRARY="{library / "libGL.so"}" '
        f'-DOPENGL_glx_LIBRARY="{library / "libGLX.so"}" '
        f'-DOPENGL_EGL_INCLUDE_DIR="{include}" '
        f'-DOPENGL_egl_LIBRARY="{library / "libEGL.so"}" "$@"\n'
    )
    (venv / "bin" / "cmake").chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    root = parser.parse_args().root
    source, glfw, venv = root / "src", root / "glfw", root / "venv"
    prefix = root / "prefix"
    root.mkdir(parents=True, exist_ok=True)
    if not (source / ".git").is_dir():
        run("git", "clone", "--recursive", AGARCL_REPOSITORY, str(source))
    run("git", "-C", str(source), "checkout", "--quiet", AGARCL_COMMIT)
    if not (glfw / ".git").is_dir():
        run("git", "clone", GLFW_REPOSITORY, str(glfw))
    run("git", "-C", str(glfw), "checkout", "--quiet", GLFW_COMMIT)
    ensure_venv(venv)
    run("uv", "pip", "install", "--quiet", "--python", str(venv / "bin" / "python"), *REQUIREMENTS)
    toolchain = linux_toolchain(root, prefix, venv) if sys.platform == "linux" else {}
    run(
        str(venv / "bin" / "cmake"),
        "-S",
        str(glfw),
        "-B",
        str(root / "glfw-build"),
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        "-DGLFW_BUILD_EXAMPLES=OFF",
        "-DGLFW_BUILD_TESTS=OFF",
        "-DGLFW_BUILD_DOCS=OFF",
        "-DGLFW_BUILD_WAYLAND=OFF",
        "-DGLFW_BUILD_X11=OFF",
    )
    run(
        str(venv / "bin" / "cmake"),
        "--build",
        str(root / "glfw-build"),
        "--target",
        "install",
        "-j2",
    )
    environment = os.environ.copy()
    environment["PATH"] = f"{venv / 'bin'}:{environment['PATH']}"
    environment["CMAKE_PREFIX_PATH"] = str(prefix)
    environment["CXX"] = os.environ.get("CXX") or "clang++"
    environment |= toolchain
    shutil.rmtree(source / "build", ignore_errors=True)  # do not reuse a previous compiler
    run(
        "uv",
        "pip",
        "install",
        "--quiet",
        "--python",
        str(venv / "bin" / "python"),
        "--no-build-isolation",
        "--no-deps",
        str(source),
        env=environment,
    )
    print(f"PERPETUA_AGARCL_PYTHON={venv / 'bin' / 'python'}")
    print(f"PERPETUA_AGARCL_SOURCE={source}")


if __name__ == "__main__":
    main()
