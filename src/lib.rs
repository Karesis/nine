//    Copyright 2025 Karesis
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

pub mod analyzer;
pub mod ast;
pub mod codegen;
pub mod diagnostic;
pub mod driver;
pub mod lexer;
pub mod parser;
pub mod source;
pub mod target;
pub mod token;
pub mod linker; 

use crate::analyzer::Analyzer;
use crate::codegen::CodeGen;
use crate::diagnostic::emit_diagnostics;
use crate::driver::Driver;
use crate::linker::link_executable; 
use crate::target::TargetMetrics;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetTriple,
};
use inkwell::OptimizationLevel;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::fs; 

#[derive(Debug, Clone)]
pub enum EmitType {
    LlvmIr,
    Bitcode,
    Object,
    Executable,
}

#[derive(Debug, Clone)]
pub struct CompileConfig {
    pub source_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub emit_type: EmitType,
    pub target: TargetMetrics,
    pub verbose: bool,
}

/// 返回 Result<(), Box<dyn Error>> 以便于上层处理错误
pub fn compile(config: CompileConfig) -> Result<(), Box<dyn Error>> {
    let mut driver = Driver::new();

    let program = match driver.compile_project(config.source_path.clone()) {
        Ok(p) => p,
        Err(fatal_msg) => {
            eprintln!("Fatal Error: {}", fatal_msg);
            return Err("Parsing failed due to fatal error".into());
        }
    };

    if !driver.parse_errors.is_empty() {
        emit_diagnostics(&driver, &driver.parse_errors);
        return Err("Compilation failed due to syntax errors".into());
    }

    let mut analyzer = Analyzer::new(config.target.clone());
    analyzer.analyze_program(&program);

    if !analyzer.ctx.errors.is_empty() {
        emit_diagnostics(&driver, &analyzer.ctx.errors);
        return Err("Compilation failed due to semantic errors".into());
    }

    let context = Context::create();
    let module_name = config.source_path.file_stem().unwrap().to_str().unwrap();
    let module = context.create_module(module_name);
    let builder = context.create_builder();

    let mut codegen = CodeGen::new(&context, &module, &builder, &analyzer.ctx, &program);
    codegen.compile_program(&program);

    handle_output(&module, &config)?;

    Ok(())
}

fn handle_output(module: &Module, config: &CompileConfig) -> Result<(), Box<dyn Error>> {
    let output_path = match &config.output_path {
        Some(p) => p.clone(),
        None => {
            let mut p = config.source_path.clone();
            match config.emit_type {
                EmitType::LlvmIr => { p.set_extension("ll"); },
                EmitType::Bitcode => { p.set_extension("bc"); },
                EmitType::Object => { p.set_extension("o"); },
                EmitType::Executable => {
                    if cfg!(target_os = "windows") {
                        p.set_extension("exe");
                    } else {
                        p.set_extension(""); 
                    }
                }
            }
            p
        }
    };

    if config.verbose {
        println!("Emitting {:?} to {:?}", config.emit_type, output_path);
    }

    match config.emit_type {
        EmitType::LlvmIr => {
            module
                .print_to_file(&output_path)
                .map_err(|e| e.to_string())?;
        }
        EmitType::Bitcode => {
            module.write_bitcode_to_path(&output_path);
        }
        EmitType::Object => {
            emit_object_file(module, &output_path, &config.target)?;
        }
        EmitType::Executable => {
            let temp_obj_path = output_path.with_extension("o");

            if config.verbose {
                println!("Generating temporary object file: {:?}", temp_obj_path);
            }

            emit_object_file(module, &temp_obj_path, &config.target)?;

            if let Err(e) = link_executable(&temp_obj_path, &output_path, &config.target, config.verbose) {
                // 如果链接失败，保留 .o 文件可能有助于调试，但通常应该报错
                return Err(format!("Linking failed: {}", e).into());
            }
            if !config.verbose {
                let _ = fs::remove_file(&temp_obj_path);
            }
        }
    }

    Ok(())
}

fn emit_object_file(
    module: &Module,
    path: &Path,
    target_metrics: &TargetMetrics,
) -> Result<(), String> {
    Target::initialize_all(&InitializationConfig::default());

    let triple_str = target_metrics.triple.to_string();
    let triple = TargetTriple::create(&triple_str);
    module.set_triple(&triple);

    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    
    let target_machine = target
        .create_target_machine(
            &triple,
            "generic",
            "",
            OptimizationLevel::Default,
            RelocMode::Default,
            CodeModel::Default,
        )
        .ok_or("Could not create target machine")?;

    target_machine
        .write_to_file(module, FileType::Object, path)
        .map_err(|e| e.to_string())
}