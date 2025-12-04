# Nine Lang (9-lang) 🚀

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Version](https://img.shields.io/badge/version-0.1.0--alpha-orange)

**Nine Lang** is a modern, statically-typed systems programming language focusing on simplicity, explicit semantics, and module-based organization. Built with Rust and LLVM.

> "A language that bridges the gap between C's simplicity and modern structural needs."

## ✨ Features

- **Module System**: File-system based module resolution (`.9` extension).
- **Type System**: Strong static typing with explicit `set`/`mut` mutability semantics.
- **LLVM Backend**: Compiles to native binary via LLVM IR (high performance).
- **C Interop**: Seamlessly call C standard library functions via `extern`.
- **Clean Syntax**: No semi-colon noise for blocks, explicit keywords (`ret`, `imp`, `fn`).

## 🛠️ Quick Start

### Prerequisites
- LLVM 20+ installed.
- Rust toolchain (cargo).

### Installation

```bash
git clone https://github.com/yourname/nine.git
cd nine-lang
cargo build --release
# Add target/release/ninec to your PATH
```

### Writing Your First Program

Create a file named `main.9`:

```rust
extern fn printf(fmt: ^u8, ...) -> i32;

fn main() -> i32 {
    set greeting: ^u8 = "Hello, Nine Lang!\n";
    printf(greeting);
    ret 0;
}
```

Compile and run:

```bash
ninec main.9
# ninec need clang to link libc
clang main.o -o main
./main
# Output: Hello, Nine Lang!
```

## 📂 Project Structure

- `src/`: Compiler source code (Rust).
  - `lexer.rs` / `parser.rs`: Frontend.
  - `analyzer.rs`: Semantic analysis & Type checking.
  - `codegen.rs`: LLVM IR generation.
  - `driver.rs`: Compilation orchestration.
- `demo/`: Example programs written in Nine Lang.

## 📝 Language Tour

### Modules
Use `mod` to declare declarations and file paths imply hierarchy.

```rust
// src/math.9
pub fn add(a: i32, b: i32) -> i32 {
    ret a + b;
}

// src/main.9
mod math;
fn main() {
    math::add(1, 2);
}
```

### Mutability
By default, variables are explicit.

```rust
set x: i32 = 10; // Immutable
mut y: i32 = 20; // Mutable
y = 30;          // OK
// x = 11;       // Error!
```

## 🗺️ Roadmap

- [x] Basic types & Control flow
- [x] Functions & Modules
- [x] Structs & Enums
- [ ] Generics
- [ ] Standard Library (fluf integration)

## 📄 License

This project is licensed under the Apache-2.0 License.