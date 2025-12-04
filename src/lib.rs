pub mod ast;
pub mod driver;
pub mod lexer;
pub mod parser;
pub mod source;
pub mod token;
pub mod analyzer;
pub mod codegen;
pub mod diagnostic;

use std::path::PathBuf;
use std::error::Error;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::context::Context;
use inkwell::module::Module;
use crate::driver::Driver;
use crate::analyzer::Analyzer;
use crate::codegen::CodeGen;
use crate::diagnostic::emit_diagnostics;
use inkwell::OptimizationLevel;

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
    pub verbose: bool,
}

/// 返回 Result<(), Box<dyn Error>> 以便于上层处理错误
pub fn compile(config: CompileConfig) -> Result<(), Box<dyn Error>> {
    let mut driver = Driver::new();
    
    // 1. Driver/Parser 阶段
    let program = match driver.compile_project(config.source_path.clone()) {
        Ok(p) => p,
        Err(fatal_msg) => {
            // 这种是文件读不到之类的致命错误，直接打印 String
            eprintln!("Fatal Error: {}", fatal_msg);
            return Err("Parsing failed due to fatal error".into());
        }
    };

    // 检查语法错误
    // 此时 driver.parse_errors 里收集了所有模块的解析错误
    if !driver.parse_errors.is_empty() {
        // 使用漂亮打印！
        emit_diagnostics(&driver, &driver.parse_errors);
        return Err("Compilation failed due to syntax errors".into());
    }

    // 2. Analyzer 阶段
    let mut analyzer = Analyzer::new();
    analyzer.analyze_program(&program);

    if !analyzer.ctx.errors.is_empty() {
        emit_diagnostics(&driver, &analyzer.ctx.errors);
        return Err("Compilation failed due to semantic errors".into());
    }

    // 3. Codegen 
    let context = Context::create();
    let module_name = config.source_path.file_stem().unwrap().to_str().unwrap();
    let module = context.create_module(module_name);
    let builder = context.create_builder();

    let mut codegen = CodeGen::new(&context, &module, &builder, &analyzer.ctx);
    codegen.compile_program(&program);

    // 4. 根据配置输出产物
    handle_output(&module, &config)?;

    Ok(())
}

fn handle_output(module: &Module, config: &CompileConfig) -> Result<(), Box<dyn Error>> {
    // 确定输出路径
    let output_path = match &config.output_path {
        Some(p) => p.clone(),
        None => {
            let ext = match config.emit_type {
                EmitType::LlvmIr => "ll",
                EmitType::Bitcode => "bc",
                EmitType::Object => "o",
                _ => "out",
            };
            config.source_path.with_extension(ext)
        }
    };

    if config.verbose {
        println!("Emitting {:?} to {:?}", config.emit_type, output_path);
    }

    match config.emit_type {
        EmitType::LlvmIr => {
            module.print_to_file(&output_path).map_err(|e| e.to_string())?;
        }
        EmitType::Object => {
            // 将原来的 emit_object_file 逻辑移到这里或 codegen 模块中
            emit_object_file(module, &output_path)?;
        }
        _ => {
            eprintln!("Warning: Output type {:?} not yet fully implemented", config.emit_type);
        }
    }
    
    Ok(())
}

// 辅助：生成 .o 文件
fn emit_object_file(module: &inkwell::module::Module, path: &std::path::Path) -> Result<(), String> {
    // 1. 初始化 Native Target
    Target::initialize_all(&InitializationConfig::default());

    // 2. 获取 Target Triple
    let triple = TargetMachine::get_default_triple();
    module.set_triple(&triple);

    // 3. 创建 Target Machine
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    let target_machine = target
        .create_target_machine(
            &triple,
            "generic", // CPU
            "",        // Features
            OptimizationLevel::Default,
            RelocMode::Default,
            CodeModel::Default,
        )
        .ok_or("Could not create target machine")?;

    // 4. 写文件
    target_machine
        .write_to_file(module, FileType::Object, path)
        .map_err(|e| e.to_string())
}