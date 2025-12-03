mod source;
mod token;
mod lexer;
mod ast;
mod parser;
mod driver;
mod analyzer;

use std::env;
use std::path::PathBuf;
use driver::Driver;
use analyzer::Analyzer;

fn main() {
    // 1. 获取命令行参数，或者默认使用 "test.9"
    let args: Vec<String> = env::args().collect();
    let entry_file = if args.len() > 1 {
        args[1].clone()
    } else {
        "test.9".to_string()
    };

    println!("=== Compiling [{}] ===", entry_file);
    let entry_path = PathBuf::from(entry_file);

    // 2. 初始化 Driver
    let mut driver = Driver::new();

    // 3. 运行 Parser (构建 AST)
    println!("--- Phase 1: Parsing ---");
    let program = match driver.compile_project(entry_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Panic/Error during parsing: {}", e);
            return;
        }
    };
    println!("> Parsing complete. Found {} top-level items.", program.items.len());

    // 4. 运行 Analyzer (语义分析 & 类型检查)
    println!("--- Phase 2: Analyzing ---");
    let mut analyzer = Analyzer::new();
    analyzer.analyze_program(&program);

    // 5. 检查结果与报错打印
    if analyzer.ctx.errors.is_empty() {
        println!("> Analysis Complete: \x1b[32mSUCCESS\x1b[0m"); // 绿色 SUCCESS
        println!("> Inferred Types: {}", analyzer.ctx.types.len());
        println!("> Resolved Paths: {}", analyzer.ctx.path_resolutions.len());
        
        // 这里可以进行下一步：CodeGen
        // codegen.generate(&program, &analyzer.ctx);
    } else {
        println!("> Analysis Complete: \x1b[31mFAILED\x1b[0m with {} errors:", analyzer.ctx.errors.len()); // 红色 FAILED
        
        for (i, err) in analyzer.ctx.errors.iter().enumerate() {
            // 利用 SourceManager 将全局 Span 转换为 文件名:行:列
            if let Some((file, line, col)) = driver.source_manager.lookup_source(err.span) {
                // 打印格式： [1] Error in main.9:10:5 -> Type mismatch...
                println!("[{}] Error in {}:{}:{} -> {}", i + 1, file.name, line, col, err.message);
                
                // 可选：打印出错的那一行代码作为上下文
                // (这需要一些额外的切片逻辑，暂时先不加，防止 index out of bounds)
            } else {
                println!("[{}] Error at unknown location ({:?}) -> {}", i + 1, err.span, err.message);
            }
        }
    }
}