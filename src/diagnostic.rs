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

use crate::driver::Driver;
use crate::source::{SourceFile, Span};
use std::cmp;

pub trait Diagnosable {
    fn span(&self) -> Span;
    fn message(&self) -> &str;
}

use crate::parser::ParseError;
impl Diagnosable for ParseError {
    fn span(&self) -> Span {
        self.span
    }
    fn message(&self) -> &str {
        &self.message
    }
}

use crate::analyzer::AnalyzeError;
impl Diagnosable for AnalyzeError {
    fn span(&self) -> Span {
        self.span
    }
    fn message(&self) -> &str {
        &self.message
    }
}

pub fn emit_diagnostics<E: Diagnosable>(driver: &Driver, errors: &[E]) {
    for err in errors {
        print_one_error(driver, err);
    }
}

fn print_one_error<E: Diagnosable>(driver: &Driver, err: &E) {
    let span = err.span();
    let msg = err.message();

    // 1. 获取位置信息
    // lookup_source 返回 (SourceFile引用, 行号1-based, 列号1-based)
    let location = driver.source_manager.lookup_source(span);

    // 颜色代码
    let color_red = "\x1b[31;1m";
    let color_blue = "\x1b[34;1m";
    let color_reset = "\x1b[0m";
    let color_bold = "\x1b[1m";

    eprintln!("{}error{}: {}", color_red, color_reset, msg);

    if let Some((file, line_num, col_num)) = location {
        // 打印文件路径:  --> src/main.9:10:5
        eprintln!(
            "  {}-->{} {}:{}:{}",
            color_blue,
            color_reset,
            file.path.display(),
            line_num,
            col_num
        );

        // 2. 获取源码行内容
        if let Some(line_str) = get_line_slice(file, line_num) {
            let line_num_str = line_num.to_string();
            let padding = " ".repeat(line_num_str.len());

            // 装饰行 1: "  |"
            eprintln!("  {} {} |{}", color_blue, padding, color_reset);

            // 源码行: "10 |     let x = 1;"
            let (safe_line, col_offset_fix) = sanitize_line(line_str, col_num);
            eprintln!(
                "  {} {} |{} {}",
                color_blue, line_num_str, color_reset, safe_line
            );

            // 指针行: "   |             ^"
            eprint!("  {} {} |{}", color_blue, padding, color_reset);

            // 计算前导空格
            // col_num 是 1-based，所以前导空格是 col_num - 1 + tab带来的修正
            let spaces = " ".repeat(col_num.saturating_sub(1) + col_offset_fix);
            eprint!("{}", spaces);

            // 计算波浪线长度
            let raw_len = span.end.saturating_sub(span.start);
            let line_remain_len = line_str.len().saturating_sub(col_num - 1);
            let draw_len = cmp::max(1, cmp::min(raw_len, line_remain_len));

            let underline = "^".repeat(draw_len);
            eprintln!("{}{}{}", color_red, underline, color_reset);
        }
    } else {
        eprintln!("  (Location unavailable)");
    }
    eprintln!(); // 空一行
}

/// 辅助函数：根据 line_starts 高效切片获取行内容
fn get_line_slice(file: &SourceFile, line_1based: usize) -> Option<&str> {
    if line_1based == 0 {
        return None;
    }
    let idx = line_1based - 1; // 0-based index for vector

    if idx >= file.line_starts.len() {
        return None;
    }

    let start_offset = file.line_starts[idx];

    // 结束 offset 是下一行的开始，或者是文件末尾
    let end_offset = if idx + 1 < file.line_starts.len() {
        file.line_starts[idx + 1]
    } else {
        file.src.len()
    };

    if start_offset > end_offset {
        return None;
    } // 安全检查

    // 获取切片
    let slice = &file.src[start_offset..end_offset];

    // 去掉行末的换行符 (\n 或 \r\n)
    Some(slice.trim_end_matches(&['\n', '\r'][..]))
}

/// 辅助函数：处理 Tab 字符
fn sanitize_line(raw: &str, target_col: usize) -> (String, usize) {
    let mut result = String::new();
    let mut offset_fix = 0;

    // 遍历字符，如果在 target_col 之前遇到 tab，需要增加 offset_fix
    for (i, c) in raw.chars().enumerate() {
        if c == '\t' {
            result.push_str("    "); // 4空格
            // 如果这个 tab 在报错点之前，让下划线右移 3 格 (4-1)
            if i < target_col - 1 {
                offset_fix += 3;
            }
        } else {
            result.push(c);
        }
    }
    (result, offset_fix)
}
