extern crate cpp_build;

fn main() {
    let include_path = "/mnt/f/Users/Simon/Documents/Projekte/hexl/hexl/include";
    let lib_path = "/mnt/f/Users/Simon/Documents/Projekte/hexl/build/hexl/lib";
    cpp_build::Config::new().include(include_path).build("src/lib.rs");
    println!("cargo:rustc-link-search={}", lib_path);
    println!("cargo:rustc-link-lib=hexl");
    println!("cargo:rerun-if-changed=src/bindings.rs")
}