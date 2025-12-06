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

use ninec::compile;
use ninec::target::TargetMetrics;
use ninec::{CompileConfig, EmitType};
use std::env;
use std::path::PathBuf;
use std::process::exit;

fn print_help() {
    println!("Usage: ninec [OPTIONS] <INPUT>");
    println!();
    println!("Options:");
    println!("  -o, --output <FILE>   Specify output file path");
    println!("  --emit <TYPE>         Emit type [ir|bc|obj|exe] (default: exe)");
    println!("  --target <TRIPLE>     Target triple (e.g., x86_64-unknown-linux-gnu)");
    println!("  -v, --verbose         Enable verbose logging");
    println!("  -h, --help            Print this help message");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    // 默认配置
    let mut source_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    // 默认改为 Executable，因为这是大多数用户想要的
    let mut emit_type = EmitType::Executable; 
    let mut verbose = false;
    let mut target_str: Option<String> = None;

    // 手动解析参数
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                return;
            }
            "-v" | "--verbose" => {
                verbose = true;
            }
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: Missing value for --output");
                    exit(1);
                }
                output_path = Some(PathBuf::from(&args[i]));
            }
            "--target" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: Missing value for --target");
                    exit(1);
                }
                target_str = Some(args[i].clone());
            }
            "--emit" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: Missing value for --emit");
                    exit(1);
                }
                emit_type = match args[i].as_str() {
                    "ir" => EmitType::LlvmIr,
                    "bc" => EmitType::Bitcode,
                    "obj" => EmitType::Object,
                    "exe" => EmitType::Executable,
                    _ => {
                        eprintln!("Error: Unknown emit type '{}'", args[i]);
                        exit(1);
                    }
                };
            }
            arg => {
                if arg.starts_with('-') {
                    eprintln!("Error: Unknown option '{}'", arg);
                    print_help();
                    exit(1);
                }
                if source_path.is_some() {
                    eprintln!("Error: Multiple input files specified");
                    exit(1);
                }
                source_path = Some(PathBuf::from(arg));
            }
        }
        i += 1;
    }

    let source_path = match source_path {
        Some(p) => p,
        None => {
            eprintln!("Error: No input file specified");
            print_help();
            exit(1);
        }
    };

    let target_metrics = match target_str {
        Some(s) => TargetMetrics::from_str(&s).unwrap_or_else(|e| {
            eprintln!("Error parsing target triple: {}", e);
            exit(1);
        }),
        None => TargetMetrics::host(),
    };

    let config = CompileConfig {
        source_path,
        output_path,
        emit_type,
        target: target_metrics,
        verbose,
    };

    if let Err(e) = compile(config) {
        eprintln!("Error: {}", e);
        exit(1);
    }
}