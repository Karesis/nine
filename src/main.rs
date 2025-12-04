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

use clap::{Parser, ValueEnum};
use ninec::compile;
use ninec::{CompileConfig, EmitType};
use std::path::PathBuf;
use std::process::exit;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input source file
    #[arg(required = true)]
    input: PathBuf,

    /// Output file path
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Emit type
    #[arg(long, value_enum, default_value_t = ArgEmitType::Obj)]
    emit: ArgEmitType,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum ArgEmitType {
    Ir,
    Bc,
    Obj,
}

fn main() {
    let args = Args::parse();

    // 将 CLI 参数转换为内部 Config
    let config = CompileConfig {
        source_path: args.input,
        output_path: args.output,
        emit_type: match args.emit {
            ArgEmitType::Ir => EmitType::LlvmIr,
            ArgEmitType::Bc => EmitType::Bitcode,
            ArgEmitType::Obj => EmitType::Object,
        },
        verbose: args.verbose,
    };

    // 调用编译器核心
    if let Err(e) = compile(config) {
        eprintln!("Error: {}", e);
        exit(1);
    }

    if args.verbose {
        println!("Build success.");
    }
}
