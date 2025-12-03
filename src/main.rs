use std::env;
use std::path::PathBuf;
use std::process::exit;

use inkwell::context::Context;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::OptimizationLevel;

mod ast;
mod driver;
mod lexer;
mod parser;
mod source;
mod token;
mod analyzer;
mod codegen;

use driver::Driver;
use analyzer::Analyzer;
use codegen::CodeGen;

fn main() {
    // 1. 参数解析
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <entry_file.9>", args[0]);
        exit(1);
    }
    let entry_path = PathBuf::from(&args[1]);

    // 2. Driver: 解析 AST (Parse)
    let mut driver = Driver::new();
    let program = match driver.compile_project(entry_path.clone()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Driver Error: {}", e);
            exit(1);
        }
    };

    // 3. Analyzer: 语义分析 (Type Check)
    let mut analyzer = Analyzer::new();
    analyzer.analyze_program(&program);

    if !analyzer.ctx.errors.is_empty() {
        eprintln!("Analysis failed with {} errors:", analyzer.ctx.errors.len());
        for err in &analyzer.ctx.errors {
            // 这里可以利用 source_manager 打印更详细的报错 (行号、代码片段)
            // 简单版本：
            if let Some((file, line, col)) = driver.source_manager.lookup_source(err.span) {
                eprintln!("  File: {}, Line: {}, Col: {}", file.name, line, col);
                // 打印源码行
                // ...
            }
            eprintln!("  Error: {}", err.message);
            eprintln!();
        }
        exit(1);
    }

    // 4. Codegen: 生成 LLVM IR
    let context = Context::create();
    let module = context.create_module("main");
    let builder = context.create_builder();

    let mut codegen = CodeGen::new(&context, &module, &builder, &analyzer.ctx);
    
    // 开始编译
    codegen.compile_program(&program);

    // 5. 输出结果
    // A. 打印 IR 到标准输出 (调试用)
    // module.print_to_stderr(); 

    // B. 输出到文件 (entry_file.ll)
    let output_ll = entry_path.with_extension("ll");
    if let Err(e) = module.print_to_file(&output_ll) {
        eprintln!("Error writing LLVM IR: {:?}", e);
        exit(1);
    }
    println!("LLVM IR written to: {:?}", output_ll);

    // C. (可选) 编译为目标机器代码 (.o)
    // 这需要初始化 TargetMachine
    if let Err(e) = emit_object_file(&module, &entry_path.with_extension("o")) {
        eprintln!("Error emitting object file: {:?}", e);
        // 不退出，因为 IR 可能已经生成成功了
    } else {
        println!("Object file written to: {:?}", entry_path.with_extension("o"));
    }
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