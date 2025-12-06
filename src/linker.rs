// linker.rs
use std::path::Path;
use std::process::Command;
use crate::target::TargetMetrics;

/// 检查 clang 是否存在
fn check_clang_installed() -> bool {
    Command::new("clang")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

pub fn link_executable(
    obj_path: &Path,
    output_path: &Path,
    target: &TargetMetrics,
    verbose: bool,
) -> Result<(), String> {
    let linker_cmd = "clang";

    if !check_clang_installed() {
        return Err(format!(
            "Linker 'clang' not found in PATH.\n\
            \n\
            Please install clang to build executables:\n\
            - Ubuntu/Debian: sudo apt install clang\n\
            - Fedora/CentOS: sudo dnf install clang\n\
            - macOS: xcode-select --install\n\
            - Windows: choco install llvm\n\
            \n\
            Or wait for the upcoming '9lang-up' tool to handle this for you! ;)"
        ));
    }
    let target_triple = target.triple.to_string();

    if verbose {
        println!("[Linker] Using {} for target {}", linker_cmd, target_triple);
    }

    let mut cmd = Command::new(linker_cmd);
    
    cmd.arg(obj_path)
       .arg("-o")
       .arg(output_path);

    cmd.arg("--target").arg(&target_triple);

    //? LTO (Link Time Optimization)
    //? -flto -O3
    //? cmd.arg("-O2"); 
    
    //? linking to lib
    //? cmd.arg("-lm"); 

    if verbose {
        println!("[Linker] Executing: {:?}", cmd);
    }

    let output = cmd.output().map_err(|e| format!("Failed to run linker: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Linking failed:\n{}", stderr.trim()));
    }

    Ok(())
}