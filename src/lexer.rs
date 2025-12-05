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

use crate::token::{Token, TokenKind};

pub struct Lexer<'a> {
    src: &'a str,
    chars: std::iter::Peekable<std::str::Chars<'a>>,

    /// 当前字符的字节偏移量
    current_pos: usize,
    /// 当前 Token 开始的字节偏移量
    start_pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Self {
            src,
            chars: src.chars().peekable(),
            current_pos: 0,
            start_pos: 0,
        }
    }

    /// 获取下一个 Token
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();

        self.start_pos = self.current_pos;

        let c = match self.advance() {
            Some(c) => c,
            None => return self.make_token(TokenKind::EOF),
        };

        match c {
            // 1. 标识符与关键字 (Start with a-z, A-Z, _)
            c if is_ident_start(c) => self.scan_identifier(),

            // 2. 数字 (Start with 0-9)
            c if c.is_ascii_digit() => self.scan_number(),

            // 3. 字符串与字符
            '"' => self.scan_string(),
            '\'' => self.scan_char(),

            // 4. 单字符与双字符符号
            '(' => self.make_token(TokenKind::LParen),
            ')' => self.make_token(TokenKind::RParen),
            '{' => self.make_token(TokenKind::LBrace),
            '}' => self.make_token(TokenKind::RBrace),
            '[' => self.make_token(TokenKind::LBracket),
            ']' => self.make_token(TokenKind::RBracket),
            ',' => self.make_token(TokenKind::Comma),
            ';' => self.make_token(TokenKind::Semi),
            '+' => self.make_token(TokenKind::Plus),
            '*' => self.make_token(TokenKind::Star),
            '/' => self.make_token(TokenKind::Slash),
            '%' => self.make_token(TokenKind::Percent),
            '^' => self.make_token(TokenKind::Caret), // Ptr / Deref
            '&' => self.make_token(TokenKind::Ampersand), // AddrOf
            '|' => self.make_token(TokenKind::Pipe),  // Switch pattern
            '@' => self.make_token(TokenKind::At),
            '#' => self.make_token(TokenKind::Hash),

            // 需要向前看的符号
            '.' => {
                // 检查是否是 '...'
                let mut lookahead = self.chars.clone();
                let next1 = lookahead.next();
                let next2 = lookahead.next();

                if let (Some('.'), Some('.')) = (next1, next2) {
                    // 确实是 '...'，吞噬掉后两个点
                    self.advance(); // second dot
                    self.advance(); // third dot
                    self.make_token(TokenKind::DotDotDot)
                } else {
                    // 只是单个点 '.'
                    self.make_token(TokenKind::Dot)
                }
            }

            ':' => {
                if self.match_char(':') {
                    self.make_token(TokenKind::ColonColon)
                } else {
                    self.make_token(TokenKind::Colon)
                }
            }

            '=' => {
                if self.match_char('=') {
                    self.make_token(TokenKind::EqEq)
                } else {
                    self.make_token(TokenKind::Eq)
                }
            }

            '!' => {
                if self.match_char('=') {
                    self.make_token(TokenKind::NeEq)
                } else {
                    self.make_token(TokenKind::Bang)
                }
            }

            '<' => {
                if self.match_char('<') {
                    self.make_token(TokenKind::Shl) // <<
                } else if self.match_char('=') {
                    self.make_token(TokenKind::LtEq)
                } else {
                    self.make_token(TokenKind::Lt)
                }
            }

            '>' => {
                if self.match_char('>') {
                    self.make_token(TokenKind::Shr) // >>
                } else if self.match_char('=') {
                    self.make_token(TokenKind::GtEq)
                } else {
                    self.make_token(TokenKind::Gt)
                }
            }

            '-' => {
                if self.match_char('>') {
                    self.make_token(TokenKind::Arrow)
                } else {
                    self.make_token(TokenKind::Minus)
                }
            }

            // 未知字符
            _ => self.make_token(TokenKind::ERROR),
        }
    }

    // === 具体的扫描函数 ===

    fn scan_identifier(&mut self) -> Token {
        while let Some(&c) = self.chars.peek() {
            if is_ident_continue(c) {
                self.advance();
            } else {
                break;
            }
        }

        // 提取文本
        let text = &self.src[self.start_pos..self.current_pos];

        // 检查是否是关键字
        let kind = TokenKind::lookup_keyword(text).unwrap_or(TokenKind::Identifier);

        self.make_token(kind)
    }

    fn scan_number(&mut self) -> Token {
        // 1. 检查起始字符是否为 '0'
        let start_char = self.src.as_bytes()[self.start_pos] as char;

        if start_char == '0' {
            // 此时 self.chars.peek() 指向的是 '0' 后面的那个字符 (例如 'x')
            if let Some(&c) = self.chars.peek() {
                match c {
                    'x' | 'X' => {
                        self.advance(); // 消耗 x
                        // 扫描 Hex 数字 (0-9, a-f, A-F, _)
                        while let Some(&c) = self.chars.peek() {
                            if c.is_ascii_hexdigit() || c == '_' {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        return self.make_token(TokenKind::Integer);
                    }
                    'b' | 'B' => {
                        self.advance(); // 消耗 b
                        // 扫描 Bin 数字 (0-1, _)
                        while let Some(&c) = self.chars.peek() {
                            if c == '0' || c == '1' || c == '_' {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        return self.make_token(TokenKind::Integer);
                    }
                    'o' | 'O' => {
                        self.advance(); // 消耗 o
                        // 扫描 Oct 数字 (0-7, _)
                        while let Some(&c) = self.chars.peek() {
                            if (c >= '0' && c <= '7') || c == '_' {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        return self.make_token(TokenKind::Integer);
                    }
                    _ => {} // 后面不是 x/b/o，是普通的 0 或者 0.123
                }
            }
        }

        // 2. 普通十进制扫描
        // 如果是 0xB8...，上面的分支会返回，不会走到这里
        // 如果是 123，会走到这里
        while let Some(&c) = self.chars.peek() {
            if c.is_ascii_digit() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        // 3. 扫描小数部分
        if let Some(&'.') = self.chars.peek() {
            let mut iter_clone = self.chars.clone();
            iter_clone.next();
            if let Some(next_c) = iter_clone.next() {
                if next_c.is_ascii_digit() {
                    self.advance(); // .
                    while let Some(&c) = self.chars.peek() {
                        if c.is_ascii_digit() || c == '_' {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    return self.make_token(TokenKind::Float);
                }
            }
        }

        self.make_token(TokenKind::Integer)
    }

    fn scan_string(&mut self) -> crate::token::Token {
        // 已消耗了开头的 "
        while let Some(&c) = self.chars.peek() {
            match c {
                '"' => {
                    self.advance(); // 闭合引号
                    return self.make_token(TokenKind::StringLit);
                }
                '\\' => {
                    // 转义：吞掉 \ 和下一个字符
                    self.advance();
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }
        // 如果循环结束还没遇到 "，说明 EOF 了，报错
        self.make_token(TokenKind::ERROR)
    }

    fn scan_char(&mut self) -> crate::token::Token {
        // 已消耗了开头的 '
        // 简单处理：'a', '\n', '\t'
        if let Some(&c) = self.chars.peek() {
            if c == '\\' {
                self.advance(); // \
                self.advance(); // n, t, etc.
            } else {
                self.advance();
            }
        }

        if self.match_char('\'') {
            self.make_token(TokenKind::CharLit)
        } else {
            self.make_token(TokenKind::ERROR)
        }
    }

    // === 辅助函数 ===

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.next()?;
        self.current_pos += c.len_utf8();
        Some(c)
    }

    fn match_char(&mut self, expected: char) -> bool {
        if let Some(&c) = self.chars.peek() {
            if c == expected {
                self.advance();
                return true;
            }
        }
        false
    }

    /// 跳过空白和注释
    fn skip_whitespace_and_comments(&mut self) {
        while let Some(&c) = self.chars.peek() {
            match c {
                ' ' | '\t' | '\r' | '\n' => {
                    self.advance();
                }
                '/' => {
                    // 预读一位，看是不是 //
                    let mut lookahead = self.chars.clone();
                    lookahead.next();
                    if let Some('/') = lookahead.next() {
                        // 是注释，吞掉当前行
                        self.advance(); // /
                        self.advance(); // /
                        while let Some(&c) = self.chars.peek() {
                            if c == '\n' {
                                break;
                            }
                            self.advance();
                        }
                    } else {
                        // 只是一个除号 /，不是空白，停止跳过
                        break;
                    }
                }
                _ => break,
            }
        }
    }

    fn make_token(&self, kind: TokenKind) -> crate::token::Token {
        crate::token::Token::new(kind, self.start_pos, self.current_pos)
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}
