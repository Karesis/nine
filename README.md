# Nine Lang (9-lang)

![Build Status](https://img.shields.io/github/actions/workflow/status/Karesis/nine/rust.yml?branch=main)
![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Version](https://img.shields.io/badge/version-0.1.0--alpha-orange)

> [\!WARNING]
> **This project is in early development (Pre-Alpha).**
> Features are missing, bugs are expected, and the syntax is subject to change.

**Nine** (or **9-lang**) is a modern, statically-typed systems programming language designed for simplicity, explicit semantics, and performance. Built with **Rust** and **LLVM**, it aims to provide modern structural capabilities (like generics and methods) without hiding the underlying machine details.

> "A language that bridges the gap between C's transparency and modern type system needs."

## Key Features

  - **Zero-Cost Generics**: Monomorphized generics (`struct#<T>`, `fn#<T>`) allow for powerful abstractions without runtime overhead.
  - **Integrated Toolchain**: Compiles directly to native executables by orchestrating LLVM and Clang automatically.
  - **Explicit Memory Control**: No Garbage Collection. Manual memory management via `malloc`/`free` with clear pointer semantics (`^T` for raw pointers, `*T` for pointers to structs).
  - **Method Extensions**: Define methods for types using `imp` blocks, supporting both static (`Type::new`) and instance (`obj.method`) syntax.
  - **Predictable Layout**: Precise control over memory layout with `@sizeof` introspection, making it suitable for low-level systems programming.
  - **C Interop**: Seamless FFI with C standard libraries.
  - **Module System**: File-system based organization with `mod` and `use`.

## Quick Start

### Prerequisites

  - **LLVM 18+** (LLVM 20 recommended).
  - **Rust Toolchain** (Cargo).
  - **Clang** (Required in `PATH` for linking executables).

### Installation

```bash
git clone https://github.com/Karesis/nine.git
cd nine
cargo build --release
# Optional: Add target/release to your PATH
```

### Compile & Run

Create `main.9`:

```rust
extern fn printf(fmt: ^u8, ...) -> i32;

fn main() -> i32 {
    printf("Hello, Nine Lang!\n");
    ret 0;
}
```

Compile it (Nine automatically handles object generation and linking):

```bash
# Compile directly to an executable
ninec main.9

# Run the generated binary
./main
# (On Windows: .\main.exe)
```

## CLI Usage

Nine comes with a lightweight, built-in CLI driver:

```text
Usage: ninec [OPTIONS] <INPUT>

Options:
  -o, --output <FILE>   Specify output file path (default: same as input name)
  --emit <TYPE>         Emit type [ir|bc|obj|exe] (default: exe)
  --target <TRIPLE>     Target triple (e.g., x86_64-unknown-linux-gnu)
  -v, --verbose         Enable verbose logging (shows linker commands)
  -h, --help            Print help message
```

**Cross-Compilation Example:**

```bash
# Cross-compile for ARM64 Linux (requires clang cross-compilation support)
ninec main.9 --target aarch64-unknown-linux-gnu -o main_arm
```

## Language Tour

### 1\. Generics & Data Structures

Nine supports defining generic structs and implementing methods for them. This allows for type-safe data structures like Linked Lists.

```rust
extern fn malloc(size: u64) -> ^u8;
extern fn free(ptr: ^u8);

struct Node#<T> {
    data: T;
    next: *Node#<T>;
}

struct List#<T> {
    head: *Node#<T>;
}

// Implementation block for a generic type
imp#<T> for *List#<T> {
    pub fn push(mut self, val: T) {
        // Explicit casting and memory allocation
        set new_node: *Node#<T> = malloc(@sizeof(Node#<T>)) as *Node#<T>;
        new_node^.data = val;
        new_node^.next = self^.head;
        self^.head = new_node;
    }
}
```

### 2\. OOP-Style Method Calls

Organize code logic using `imp` blocks. Nine supports syntax sugar for method calls (`object.method()`), making code readable while keeping C-like semantics under the hood.

```rust
// src/entity.9
pub struct Player {
    hp: i32;
    ad: i32;
    
    // Static method (Constructor)
    pub fn new(hp: i32, dmg: i32) -> *Player {
        set p: *Player = malloc(@sizeof(Player)) as *Player;
        p^.hp = hp; 
        p^.ad = dmg;
        ret p;
    }
}

imp for *Player {
    // Instance method
    pub fn attack(self, mut target: *Player) {
        target^.hp = target^.hp - self^.ad;
    }
}

// src/main.9
mod entity;
use entity::Player;

fn main() {
    set hero: *Player = Player::new(100, 10);
    set goblin: *Player = Player::new(50, 5);

    hero.attack(goblin); // Syntactic sugar for Player::attack(hero, goblin)
}
```

### 3\. Functional Patterns

Nine supports function pointers and generic functions, enabling high-order logic.

```rust
struct Transformer#<T> {
    val: T;
    mapper: fn(T) -> T; 
}

fn square(v: i32) -> i32 { ret v * v; }

fn main() {
    set t: Transformer#<i32> = Transformer#<i32> {
        val: 10,
        mapper: square
    };
    // Call the function pointer
    set res: i32 = t.mapper(t.val); // 100
}
```

### 4\. Memory Layout Transparency

Nine provides introspection keywords like `@sizeof` to ensure you know exactly how your data is laid out in memory—critical for kernel or driver development.

```rust
struct Wrapper#<T> {
    head: u8;   // 1 byte
    payload: T; // generic payload
    tail: u8;   // 1 byte
}

fn check_layout() {
    // Compile-time size calculation
    // Layout: u8(1) + [padding 7] + i64(8) + u8(1) + [padding 7] = 24 bytes 
    set size: u64 = @sizeof(Wrapper#<i64>); 
}
```

## Roadmap

  - [x] Basic Types & Control Flow
  - [x] Functions & Recursion
  - [x] Module System (`mod`, `use`, `pub`)
  - [x] **Generics** (`struct#<T>`, `fn#<T>`)
  - [x] **Methods & OOP Syntax** (`imp`, `obj.method()`)
  - [x] Struct Layout & Padding
  - [x] **Compiler Driver & Linker** (Output Executables directly)
  - [ ] Standard Library (FLUF integration)
  - [ ] Self-hosting (writing the compiler in Nine)

## License

This project is licensed under the Apache-2.0 License.