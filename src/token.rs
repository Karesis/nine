use crate::source::Span;

// 完整的 Token
#[derive(Debug, Clone, Copy)] // Copy 很关键，让 Token 传递极其廉价
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

    // 辅助方法：从源码中提取文本
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.span.start..self.span.end]
    }
}

// 使用 macro_rules! 复刻 X-Macro 模式
macro_rules! define_tokens {
    (
        // 1. 动态 Token (需要携带数据或单纯作为占位)
        dynamic { $($dyn_variant:ident),* $(,)? }

        // 2. 关键字 (Keywords) - 自动生成字符串匹配
        // 格式: "text" => VariantName
        keywords { $($kw_text:literal => $kw_variant:ident),* $(,)? }

        // 3. 符号 (Symbols)
        // 格式: "text" => VariantName
        symbols { $($sym_text:literal => $sym_variant:ident),* $(,)? }
    ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum TokenKind {
            // 特殊 Token
            EOF,
            ERROR, // 词法错误可以暂存这里

            // 宏展开部分
            $($dyn_variant),*,
            $($kw_variant),*,
            $($sym_variant),*,
        }

        impl TokenKind {
            /// 获取 Token 的字面量表示 (用于调试或报错)
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

// ==========================================
// 在这里统一填写所有的 Token 定义
// ==========================================
define_tokens! {
    dynamic {
        Identifier,
        Integer,      // 123, 0xFF
        Float,        // 3.14
        StringLit,    // "hello"
        CharLit       // 'a'
    }

    keywords {
        // 类型
        "i8"    => I8,    "u8"    => U8,
        "i16"   => I16,   "u16"   => U16,
        "i32"   => I32,   "u32"   => U32,
        "i64"   => I64,   "u64"   => U64,
        "iz"    => Isize, "uz"    => Usize,
        "f32"   => F32,   "f64"   => F64,
        "bool"  => Bool,

        // 值
        "true"  => True,  "false" => False,
        "self"  => SelfVal, // 小写的 self

        // 逻辑与位运算 (关键字形式)
        "and"   => And,
        "or"    => Or,
        "band"  => BitAnd, // 按位与
        "bor"   => BitOr,  // 按位或
        "xor"   => Xor,
        "as"    => As,

        // 声明
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

        // 流程控制
        "if"       => If,
        "else"     => Else,
        "while"    => While,
        "break"    => Break,
        "continue" => Continue,
        "ret"      => Ret,
        "switch"   => Switch,
        "default"  => Default,
        "for"      => For // imp for ...
    }

    symbols {
        // 算术
        "+" => Plus,
        "-" => Minus,
        "*" => Star,
        "/" => Slash,
        "%" => Percent,
        "<<" => Shl, // Shift Left
        ">>" => Shr, // Shift Right

        // 比较
        "==" => EqEq,
        "!=" => NeEq,
        "<"  => Lt,
        "<=" => LtEq,
        ">"  => Gt,
        ">=" => GtEq,

        // 标点 & 其他
        "="  => Eq,
        "!"  => Bang,
        "^"  => Caret,      // 指针/解引用
        "&"  => Ampersand,  // 取地址后缀

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
        "|"  => Pipe,        // 用于 switch case 模式多选
        "..." => DotDotDot, // ...
    }
}
