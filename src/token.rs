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

use crate::source::Span;

#[derive(Debug, Clone, Copy)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, start: usize, end: usize) -> Self {
        Self {
            kind,
            span: Span::new(start, end),
        }
    }

    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.span.start..self.span.end]
    }
}

macro_rules! define_tokens {
    (
        dynamic { $($dyn_variant:ident),* $(,)? }
        keywords { $($kw_text:literal => $kw_variant:ident),* $(,)? }
        symbols { $($sym_text:literal => $sym_variant:ident),* $(,)? }
    ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum TokenKind {
            EOF,
            ERROR,

            $($dyn_variant),*,
            $($kw_variant),*,
            $($sym_variant),*,
        }

        impl TokenKind {
            /// 获取 Token 的字面量表示
            pub fn as_str(&self) -> &'static str {
                match self {
                    TokenKind::EOF => "EOF",
                    TokenKind::ERROR => "ERROR",
                    $(TokenKind::$dyn_variant => stringify!($dyn_variant)),*,
                    $(TokenKind::$kw_variant => $kw_text),*,
                    $(TokenKind::$sym_variant => $sym_text),*,
                }
            }

            /// 辅助函数：根据标识符文本查找是否为关键字
            pub fn lookup_keyword(text: &str) -> Option<TokenKind> {
                match text {
                    $($kw_text => Some(TokenKind::$kw_variant),)*
                    _ => None,
                }
            }
        }

    };
}

define_tokens! {
    dynamic {
        Identifier,
        Integer,
        Float,
        StringLit,
        CharLit
    }

    keywords {

        "i8"    => I8,    "u8"    => U8,
        "i16"   => I16,   "u16"   => U16,
        "i32"   => I32,   "u32"   => U32,
        "i64"   => I64,   "u64"   => U64,
        "iz"    => Isize, "uz"    => Usize,
        "f32"   => F32,   "f64"   => F64,
        "bool"  => Bool,


        "true"  => True,  "false" => False,
        "self"  => SelfVal,


        "and"   => And,
        "or"    => Or,
        "band"  => BitAnd,
        "bor"   => BitOr,
        "xor"   => Xor,
        "as"    => As,


        "fn"     => Fn,
        "struct" => Struct,
        "union"  => Union,
        "enum"   => Enum,
        "imp"    => Imp,
        "mod"    => Mod,
        "use"    => Use,
        "pub"    => Pub,
        "set"    => Set,
        "mut"    => Mut,
        "const"  => Const,
        "typedef"   => Typedef,
        "typealias" => Typealias,
        "extern" => Extern,
        "cap" => Cap,


        "if"       => If,
        "else"     => Else,
        "while"    => While,
        "break"    => Break,
        "continue" => Continue,
        "ret"      => Ret,
        "switch"   => Switch,
        "default"  => Default,
        "for"      => For,


        "sizeof" => SizeOf,
        "alignof" => AlignOf,
    }

    symbols {

        "+" => Plus,
        "-" => Minus,
        "*" => Star,
        "/" => Slash,
        "%" => Percent,
        "<<" => Shl,
        ">>" => Shr,


        "==" => EqEq,
        "!=" => NeEq,
        "<"  => Lt,
        "<=" => LtEq,
        ">"  => Gt,
        ">=" => GtEq,


        "="  => Eq,
        "!"  => Bang,
        "^"  => Caret,
        "&"  => Ampersand,

        "("  => LParen,
        ")"  => RParen,
        "["  => LBracket,
        "]"  => RBracket,
        "{"  => LBrace,
        "}"  => RBrace,

        ","  => Comma,
        ";"  => Semi,
        "."  => Dot,
        "::" => ColonColon,
        ":"  => Colon,
        "->" => Arrow,
        "|"  => Pipe,
        "..." => DotDotDot,
        "@" => At,
        "#" => Hash,
    }
}
