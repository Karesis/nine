mod source;
mod token;
mod ast;
mod lexer;
mod parser;

use lexer::Lexer;
use parser::Parser;
use source::Span;
use parser::ParseError;
use std::time::Instant;

// ==========================================
// 辅助工具：行号映射
// ==========================================
struct LineMap {
    /// 每一行开始的字节索引。
    /// lines[0] 总是 0。
    /// 如果源码有 3 行，lines 长度可能为 3 或 4 (取决于末尾是否有换行)
    line_starts: Vec<usize>,
}

impl LineMap {
    fn new(source: &str) -> Self {
        let mut line_starts = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                line_starts.push(i + 1);
            }
        }
        Self { line_starts }
    }

    /// 将字节偏移量转换为 (行号, 列号)
    /// 行号和列号都从 1 开始
    fn translate(&self, offset: usize) -> (usize, usize) {
        // 二分查找：找到第一个大于 offset 的行首，减 1 就是当前行
        let line_idx = match self.line_starts.binary_search(&offset) {
            Ok(idx) => idx,              // 精确匹配行首
            Err(idx) => idx - 1,         // 在两行之间
        };

        let line_start = self.line_starts[line_idx];
        let col_idx = offset - line_start;
        
        (line_idx + 1, col_idx + 1)
    }
}

// ==========================================
// 错误报告打印函数 (仿 Rustc 风格)
// ==========================================
fn report_error(err: &ParseError, source: &str, line_map: &LineMap) {
    let (line, col) = line_map.translate(err.span.start);
    
    // 提取出错的那一行文本
    let line_start_idx = line_map.line_starts[line - 1];
    let line_end_idx = if line < line_map.line_starts.len() {
        line_map.line_starts[line] - 1 // 减去换行符
    } else {
        source.len()
    };
    
    // 安全切片
    let line_text = &source[line_start_idx..line_end_idx];
    
    // 计算波浪线的长度
    let len = std::cmp::max(1, err.span.end - err.span.start);

    eprintln!("❌ \x1b[31mError:\x1b[0m {}", err.message);
    eprintln!("   \x1b[34m-->\x1b[0m input:{}:{}", line, col);
    eprintln!("    \x1b[34m|\x1b[0m");
    eprintln!("{:3} \x1b[34m|\x1b[0m {}", line, line_text);
    
    // 打印波浪线指引
    // 先打印前面的空格
    print!("    \x1b[34m|\x1b[0m ");
    for _ in 0..(col - 1) {
        print!(" ");
    }
    // 打印红色波浪线
    print!("\x1b[31m");
    for _ in 0..len {
        print!("^");
    }
    println!("\x1b[0m");
    
    eprintln!("    \x1b[34m=\x1b[0m Expected: {}", err.expected);
    eprintln!("    \x1b[34m=\x1b[0m Found:    {:?}\n", err.found);
}

// ==========================================
// 主入口
// ==========================================
fn main() {
    // 测试代码：我在里面故意埋了一个错误用来测试报错效果
    // 把 Switch 里的分号删掉试试？或者把 struct(8) 改成 struct(0)
    let source_code = r#"
pub mod math;
use std::io::print;

struct(8) Point {
    x: i32;
    y: i32;
}

enum Color : u8 {
    Red = 1;
    Blue;
    Green;
}

imp for Point {
    pub fn new(x: i32, y: i32) -> Point {
        ret Point { x: x, y: y };
    }

    pub fn dist_sq(const self) -> i32 {
        ret self.x * self.x + self.y * self.y;
    }
    
    fn move_by(mut self, dx: i32, dy: i32) {
        self.x = self.x + dx;
        self.y = self.y + dy;
    }
}

pub fn main() -> i32 {
    mut p: Point = Point::new(10, 20);
    const LIMIT: i32 = 100;

    if (p.x < LIMIT) {
        p.move_by(1, 1);
    } else {
        ret 1;
    }

    switch (p.y) {
        10 | 20 -> {
            print(p.y);
        }
        30 -> ret 3;
        default -> ret 0;
    }

    while (p.x > 0) : mut i: i32 = 0; {
        p.x = p.x - 1;
        if (i > 10) { break; }
    }

    ret 0;
}
"#;

    println!("=== Nine Language Parser Test ===\n");

    let line_map = LineMap::new(source_code);
    let lexer = Lexer::new(source_code);
    let mut parser = Parser::new(source_code, lexer);

    let start_time = Instant::now();
    let program = parser.parse_program();
    let duration = start_time.elapsed();

    if parser.errors.is_empty() {
        println!("✅ Parse Successful! (Time: {:?})", duration);
        println!("--------------------------------------------------");
        println!("{:#?}", program);
    } else {
        println!("❌ Parse Failed with {} errors:", parser.errors.len());
        println!("--------------------------------------------------");
        
        for err in &parser.errors {
            report_error(err, source_code, &line_map);
        }
    }
}