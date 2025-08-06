extern crate cpp_build;

use std::env;
use std::path::{Path, PathBuf};

struct HEXLLocation {
    include_dir: PathBuf,
    lib_dir: PathBuf
}

fn find_hexl() -> HEXLLocation {
    let mut include_dir = None;
    let mut lib_dir = None;
    let default_dir = Path::new("/usr/local");
    if default_dir.exists() {
        include_dir = Some(default_dir.join("include"));
        lib_dir = Some(default_dir.join("lib"));
    }
    if let Some(dir) = env::var_os("HEXL_DIR") {
        let dir = PathBuf::from(dir);
        include_dir = Some(dir.join("include"));
        lib_dir = Some(dir.join("lib"));
    }
    if let Some(dir) = env::var_os("HEXL_INCLUDE_DIR") {
        include_dir = Some(PathBuf::from(dir));
    }
    if let Some(dir) = env::var_os("HEXL_LIB_DIR") {
        lib_dir = Some(PathBuf::from(dir));
    }
    let include_dir = include_dir.expect("Could not locate HEXL include directory");
    let lib_dir = lib_dir.expect("Could not locate HEXL lib directory");
    assert!(include_dir.exists(), "Invalid HEXL include directory: `{}`", include_dir.display());
    assert!(include_dir.join("hexl").exists(), "Invalid HEXL include directory: `{}`", include_dir.display());
    assert!(lib_dir.exists(), "Invalid HEXL lib directory: `{}`", lib_dir.display());
    return HEXLLocation {
        include_dir: include_dir,
        lib_dir: lib_dir
    };
}

fn main() {
    // don't try to actually link HEXL when building documentation
    if std::env::var("DOCS_RS").is_err() {
        let hexl_location = find_hexl();
        cpp_build::Config::new().include(hexl_location.include_dir).build("src/lib.rs");
        println!("cargo:rustc-link-search={}", hexl_location.lib_dir.display());
        println!("cargo:rustc-link-lib=hexl");
        println!("cargo:rerun-if-changed=src/hexl.rs");
    }
}