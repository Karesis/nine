use std::collections::VecDeque;
use crate::token::{Token, TokenKind};
use crate::lexer::Lexer;
use crate::source::Span;
use crate::ast::*; 

#[derive(Debug, Clone)]
pub struct ParseError {
    pub expected: String,
    pub found: TokenKind,
    pub span: Span,
    pub message: String,
}

pub type ParseResult<T> = Result<T, ParseError>;

/// 负责 Token 流的管理：缓冲、预读、过滤错误
struct TokenStream<'a> {
    lexer: Lexer<'a>,
    /// 环形缓冲区：buffer[0] 是 current, buffer[1] 是 next...
    buffer: VecDeque<Token>,
    /// 记录上一个被消费的 Token 的 Span (用于 EOF 时的报错定位)
    last_span: Span,
}

impl<'a> TokenStream<'a> {
    fn new(lexer: Lexer<'a>) -> Self {
        Self {
            lexer,
            buffer: VecDeque::new(),
            // 初始 span，可以是 0..0
            last_span: Span::new(0, 0), 
        }
    }

    /// 核心：确保缓冲区里至少有 n+1 个 Token (用于 peek(n))
    fn fill(&mut self, n: usize) {
        while self.buffer.len() <= n {
            let tok = self.lexer.next_token();
            
            // 无论是正常的 Token，还是 ERROR，还是 EOF，统统放进缓冲区。
            // Parser 会在 consume/expect 时处理它们。
            self.buffer.push_back(tok);

            if tok.kind == TokenKind::EOF {
                // EOF 也要放进去之后再 break，这样 buffer 尾部才会有 EOF 哨兵
                break;
            }
        }
    }

    /// 查看第 n 个 Token (不消耗)
    fn peek(&mut self, n: usize) -> Token {
        self.fill(n);
        // 如果 n 越界（说明已经全是 EOF 了），返回 buffer 里的最后一个（必定是 EOF）
        //？问题同上，没有放进去会导致这里不是eof？
        *self.buffer.get(n).unwrap_or_else(|| self.buffer.back().unwrap())
    }

    /// 消耗当前 Token 并前进
    fn advance(&mut self) -> Token {
        self.fill(0);
        let tok = self.buffer.pop_front().unwrap_or_else(|| {
            Token::new(TokenKind::EOF, self.last_span.end, self.last_span.end) 
        });
        self.last_span = tok.span;
        tok
    }
}

// ==========================================
// Parser (High Level Logic)
// ==========================================

pub struct Parser<'a> {
    source: &'a str,      
    stream: TokenStream<'a>,
    pub errors: Vec<ParseError>,
    // 专门用于错误恢复：记录上一个被 consume 的 token 类型
    previous_kind: TokenKind,
    node_id_counter: u32,
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str, lexer: Lexer<'a>) -> Self {
        Self {
            source,
            stream: TokenStream::new(lexer),
            errors: Vec::new(),
            // 初始化为 EOF，表示“还没有任何历史”
            // 只要不是 TokenKind::Semi 就可以，EOF 是最无害的
            previous_kind: TokenKind::EOF,
            node_id_counter: 0,
        }
    }

    // === Level 0: Stream Wrapper ===

    fn peek(&mut self) -> Token {
        self.stream.peek(0)
    }

    fn check(&mut self, kind: TokenKind) -> bool {
        self.peek().kind == kind
    }
    
    // 支持无限向前看：check_nth(1) == next, check_nth(2) == next next
    fn check_nth(&mut self, n: usize, kind: TokenKind) -> bool {
        self.stream.peek(n).kind == kind
    }

    fn is_at_end(&mut self) -> bool {
        self.peek().kind == TokenKind::EOF
    }

    fn advance(&mut self) -> Token {
        let tok = self.stream.advance();
        self.previous_kind = tok.kind; // <--- 关键点
        tok
    }
    
    fn previous_span(&self) -> Span {
        self.stream.last_span
    }

    // === Level 1: Expect / Consume ===

    pub fn consume(&mut self, kind: TokenKind) -> Option<Token> {
        if self.check(kind) {
            Some(self.advance())
        } else {
            None
        }
    }

    /// 多选尝试消费：尝试匹配列表中的任意一个。
    /// 场景：处理运算符 ( + | - ) 或者可选符号
    /// 成功 -> 消耗 Token，返回 true
    /// 失败 -> 不动指针，返回 false
    pub fn match_token(&mut self, kinds: &[TokenKind]) -> bool {
        for &kind in kinds {
            if self.check(kind) {
                self.advance();
                return true;
            }
        }
        false
    }

    pub fn expect(&mut self, kind: TokenKind) -> ParseResult<Token> {
        if let Some(token) = self.consume(kind) {
            Ok(token)
        } else {
            let current = self.peek();
            Err(ParseError {
                expected: kind.as_str().to_string(),
                found: current.kind,
                span: current.span,
                message: format!("Expected '{}'", kind.as_str()),
            })
        }
    }
    
    pub fn synchronize(&mut self) {
        // 1. 必须先吃掉那个导致报错的“坏”Token，打破死循环
        self.advance();

        while !self.is_at_end() {
            // A. 检查上一位：如果刚吃掉的是分号，说明上一句结束了，这里是安全的开始
            if self.previous_kind == TokenKind::Semi {
                return;
            }

            // B. 检查当前位：如果是这些关键字，说明新的一句/声明开始了
            match self.peek().kind {
                // --- 顶层定义 (Top Level) ---
                TokenKind::Struct | 
                TokenKind::Union | 
                TokenKind::Enum | 
                TokenKind::Fn | 
                TokenKind::Mod | 
                TokenKind::Use | 
                TokenKind::Imp | 
                TokenKind::Typedef | 
                TokenKind::Typealias |
                TokenKind::Pub | // pub 也是一个强烈的声明开始信号

                // --- 变量声明 (DeclStmt) ---
                TokenKind::Set | 
                TokenKind::Mut | 
                TokenKind::Const |

                // --- 流程控制语句 (Statements) ---
                TokenKind::If | 
                TokenKind::While | 
                TokenKind::Switch | 
                TokenKind::Ret | 
                TokenKind::Break | 
                TokenKind::Continue 
                => {
                    return;
                }
                
                // 其他情况继续跳过
                _ => {}
            }

            self.advance();
        }
    }

    /// 获取 Token 对应的源码文本
    pub fn text(&self, token: Token) -> &'a str {
        token.text(self.source)
    }

    /// 核心辅助函数：分配下一个 ID
    fn next_id(&mut self) -> NodeId {
        let id = self.node_id_counter;
        self.node_id_counter += 1;
        NodeId(id)
    }

}

/// ======================================================
/// 解析类型
/// ======================================================

impl<'a> Parser<'a> {
    /// EBNF: Type -> PointerType | ArrayOrAtom
    pub fn parse_type(&mut self) -> ParseResult<Type> {
        // 利用向前看 (Lookahead) 来决定分支
        // PointerType 以 '*' 或 '^' 开头
        if self.check(TokenKind::Star) || self.check(TokenKind::Caret) {
            return self.parse_pointer_type();
        }
        
        // 否则进入 ArrayOrAtom 分支
        self.parse_array_or_atom()
    }

    /// EBNF: PointerType -> ( "*" | "^" ) Type
    fn parse_pointer_type(&mut self) -> ParseResult<Type> {
        let op_token = self.advance(); // 吃掉 '*' 或 '^'
        let start = op_token.span.start;
        
        // 递归调用 parse_type，因为指针可以多层 (**T)
        let inner_type = self.parse_type()?;
        let end = inner_type.span.end;

        let mutability = if op_token.kind == TokenKind::Star {
            Mutability::Mutable // *T
        } else {
            Mutability::Constant // ^T
        };

        Ok(Type {
            kind: TypeKind::Pointer {
                inner: Box::new(inner_type),
                mutability,
            },
            span: Span::new(start, end),
        })
    }

    /// EBNF: ArrayOrAtom -> AtomType { "[" INT "]" }
    fn parse_array_or_atom(&mut self) -> ParseResult<Type> {
        // 1. 先解析基础原子类型
        let mut ty = self.parse_atom_type()?;

        // 2. 循环处理后缀：只要后面是 '['，就说明是数组
        while self.match_token(&[TokenKind::LBracket]) {
            // EBNF: "[" INT "]"
            let size_token = self.expect(TokenKind::Integer)?;
            
            // 解析数组大小
            // 这里我们暂时只支持字面量 INT。
            // 如果你的语言支持编译时常量表达式 (const generics)，这里应该改为 parse_expression
            let size_str = self.text(size_token);
            let size = size_str.parse::<u64>().map_err(|_| ParseError {
                expected: "Integer that fits in u64".into(),
                found: size_token.kind,
                span: size_token.span,
                message: "Array size is too large".into(),
            })?;

            let end_token = self.expect(TokenKind::RBracket)?;
            
            // 构造新的 ArrayType，包裹住之前的 type
            // 例如 i32[10] -> Array(inner: i32, size: 10)
            let new_span = Span::new(ty.span.start, end_token.span.end);
            ty = Type {
                kind: TypeKind::Array {
                    inner: Box::new(ty),
                    size,
                },
                span: new_span,
            };
        }

        Ok(ty)
    }

    /// EBNF: AtomType -> PrimitiveType | FunctionType | Path | "(" Type ")"
    fn parse_atom_type(&mut self) -> ParseResult<Type> {
        let token = self.peek();
        let span = token.span;

        match token.kind {
            // 1. 基础类型 (PrimitiveType)
            TokenKind::I8 | TokenKind::U8 | TokenKind::I16 | TokenKind::U16 |
            TokenKind::I32 | TokenKind::U32 | TokenKind::I64 | TokenKind::U64 |
            TokenKind::Isize | TokenKind::Usize |
            TokenKind::F32 | TokenKind::F64 | TokenKind::Bool => {
                self.advance();
                let prim = self.token_kind_to_primitive(token.kind);
                Ok(Type {
                    kind: TypeKind::Primitive(prim),
                    span,
                })
            }

            // 2. 函数指针类型 (FunctionType) -> "fn" ...
            TokenKind::Fn => self.parse_function_type(),

            // 3. 命名类型 (Path) -> Identifier ...
            TokenKind::Identifier => {
                let path = self.parse_path()?;
                let end = path.span.end; // path 包含自己的 span
                Ok(Type {
                    kind: TypeKind::Named(path),
                    span: Span::new(span.start, end),
                })
            }

            // 4. 括号类型 "(" Type ")"
            TokenKind::LParen => {
                self.advance(); // 吃掉 '('
                let inner = self.parse_type()?;
                let end_token = self.expect(TokenKind::RParen)?;
                Ok(Type {
                    kind: inner.kind, // 括号只是改变解析优先级，不改变 AST 结构（除非你想保留 ParenType）
                    span: Span::new(span.start, end_token.span.end),
                })
            }

            _ => Err(ParseError {
                expected: "Type (primitive, identifier, fn, or pointer)".into(),
                found: token.kind,
                span,
                message: "Expected a type here".into(),
            }),
        }
    }

    /// EBNF: FunctionType -> "fn" "(" [ TypeList ] ")" [ "->" Type ]
    fn parse_function_type(&mut self) -> ParseResult<Type> {
        let start_token = self.expect(TokenKind::Fn)?; // 必须是 fn
        self.expect(TokenKind::LParen)?;

        let mut params = Vec::new();
        
        // 解析参数列表 TypeList -> Type { "," Type }
        if !self.check(TokenKind::RParen) {
            loop {
                params.push(self.parse_type()?);
                if !self.match_token(&[TokenKind::Comma]) {
                    break;
                }
            }
        }

        self.expect(TokenKind::RParen)?;

        // 解析可选的返回值 [ "->" Type ]
        let mut ret_type = None;
        let mut end_pos = self.previous_span().end; // 默认为 ')' 的位置

        if self.match_token(&[TokenKind::Arrow]) {
            let ret = self.parse_type()?;
            end_pos = ret.span.end;
            ret_type = Some(Box::new(ret));
        }

        Ok(Type {
            kind: TypeKind::Function {
                params,
                ret_type,
            },
            span: Span::new(start_token.span.start, end_pos),
        })
    }

    // === 辅助函数 ===

    /// 简单的 Path 解析：Ident { :: Ident }
    pub fn parse_path(&mut self) -> ParseResult<Path> {
        let mut segments = Vec::new();
        let start_pos = self.peek().span.start;
        
        // 第一个必然是 Ident
        let first = self.expect(TokenKind::Identifier)?;
        segments.push(Identifier { 
            name: self.text(first).to_string(), 
            span: first.span 
        });

        // 循环解析 :: Ident
        while self.match_token(&[TokenKind::ColonColon]) {
            let seg = self.expect(TokenKind::Identifier)?;
            segments.push(Identifier { 
                name: self.text(seg).to_string(), 
                span: seg.span 
            });
        }

        let end_pos = segments.last().unwrap().span.end;
        Ok(Path {
            segments,
            span: Span::new(start_pos, end_pos),
        })
    }

    /// 将 TokenKind 转换为 PrimitiveType 枚举
    fn token_kind_to_primitive(&self, kind: TokenKind) -> PrimitiveType {
        match kind {
            TokenKind::I8 => PrimitiveType::I8,
            TokenKind::U8 => PrimitiveType::U8,
            TokenKind::I16 => PrimitiveType::I16,
            TokenKind::U16 => PrimitiveType::U16,
            TokenKind::I32 => PrimitiveType::I32,
            TokenKind::U32 => PrimitiveType::U32,
            TokenKind::I64 => PrimitiveType::I64,
            TokenKind::U64 => PrimitiveType::U64,
            TokenKind::Isize => PrimitiveType::ISize,
            TokenKind::Usize => PrimitiveType::USize,
            TokenKind::F32 => PrimitiveType::F32,
            TokenKind::F64 => PrimitiveType::F64,
            TokenKind::Bool => PrimitiveType::Bool,
            // 理论上这里不会 panic，因为调用前 match 已经过滤了
            _ => unreachable!("Not a primitive token"),
        }
    }
}

impl<'a> Parser<'a> {
    // ==========================================
    // 表达式解析 (从低优先级到高优先级)
    // ==========================================

    /// 入口：Expression -> LogicOr
    pub fn parse_expr(&mut self) -> ParseResult<Expression> {
        self.parse_logic_or()
    }

    /// Level 1: LogicOr -> LogicAnd { "or" LogicAnd }
    fn parse_logic_or(&mut self) -> ParseResult<Expression> {
        let mut lhs = self.parse_logic_and()?;

        while self.match_token(&[TokenKind::Or]) {
            let op = BinaryOperator::LogicalOr;
            let rhs = self.parse_logic_and()?;
            
            let span = Span::new(lhs.span.start, rhs.span.end);
            lhs = Expression {
                id: self.next_id(),
                kind: ExpressionKind::Binary { lhs: Box::new(lhs), op, rhs: Box::new(rhs) },
                span,
            };
        }
        Ok(lhs)
    }

    /// Level 2: LogicAnd -> Equality { "and" Equality }
    fn parse_logic_and(&mut self) -> ParseResult<Expression> {
        let mut lhs = self.parse_equality()?;

        while self.match_token(&[TokenKind::And]) {
            let op = BinaryOperator::LogicalAnd;
            let rhs = self.parse_equality()?;
            
            let span = Span::new(lhs.span.start, rhs.span.end);
            lhs = Expression {
                id:self.next_id(),
                kind: ExpressionKind::Binary { lhs: Box::new(lhs), op, rhs: Box::new(rhs) },
                span,
            };
        }
        Ok(lhs)
    }

    /// Level 3: Equality -> Relational { ("=="|"!=") Relational }
    fn parse_equality(&mut self) -> ParseResult<Expression> {
        let mut lhs = self.parse_relational()?;

        while self.match_token(&[TokenKind::EqEq, TokenKind::NeEq]) {
            let op = match self.previous_kind {
                TokenKind::EqEq => BinaryOperator::Equal,
                TokenKind::NeEq => BinaryOperator::NotEqual,
                _ => unreachable!(),
            };
            let rhs = self.parse_relational()?;
            
            let span = Span::new(lhs.span.start, rhs.span.end);
            lhs = Expression {
                id: self.next_id(),
                kind: ExpressionKind::Binary { lhs: Box::new(lhs), op, rhs: Box::new(rhs) },
                span,
            };
        }
        Ok(lhs)
    }

    /// Level 4: Relational -> Additive { ("<"|"<="|">"|">=") Additive }
    fn parse_relational(&mut self) -> ParseResult<Expression> {
        let mut lhs = self.parse_additive()?;

        while self.match_token(&[TokenKind::Lt, TokenKind::LtEq, TokenKind::Gt, TokenKind::GtEq]) {
            let op = match self.previous_kind {
                TokenKind::Lt => BinaryOperator::Less,
                TokenKind::LtEq => BinaryOperator::LessEqual,
                TokenKind::Gt => BinaryOperator::Greater,
                TokenKind::GtEq => BinaryOperator::GreaterEqual,
                _ => unreachable!(),
            };
            let rhs = self.parse_additive()?;
            
            let span = Span::new(lhs.span.start, rhs.span.end);
            lhs = Expression {
                id: self.next_id(),
                kind: ExpressionKind::Binary { lhs: Box::new(lhs), op, rhs: Box::new(rhs) },
                span,
            };
        }
        Ok(lhs)
    }

    /// Level 5: Additive -> Multiplicative { ("+"| "-" | "bor" | "xor") Multiplicative }
    fn parse_additive(&mut self) -> ParseResult<Expression> {
        let mut lhs = self.parse_multiplicative()?;

        // 注意：bor 和 xor 是关键字 Token，不是符号
        while self.match_token(&[TokenKind::Plus, TokenKind::Minus, TokenKind::BitOr, TokenKind::Xor]) {
            let op = match self.previous_kind {
                TokenKind::Plus => BinaryOperator::Add,
                TokenKind::Minus => BinaryOperator::Subtract,
                TokenKind::BitOr => BinaryOperator::BitwiseOr,
                TokenKind::Xor => BinaryOperator::BitwiseXor,
                _ => unreachable!(),
            };
            let rhs = self.parse_multiplicative()?;
            
            let span = Span::new(lhs.span.start, rhs.span.end);
            lhs = Expression {
                id: self.next_id(),
                kind: ExpressionKind::Binary { lhs: Box::new(lhs), op, rhs: Box::new(rhs) },
                span,
            };
        }
        Ok(lhs)
    }

    /// Level 6: Multiplicative -> Unary { ("*" | "/" | "%" | "band") Unary }
    fn parse_multiplicative(&mut self) -> ParseResult<Expression> {
        let mut lhs = self.parse_unary()?;

        while self.match_token(&[TokenKind::Star, TokenKind::Slash, TokenKind::Percent, TokenKind::BitAnd]) {
            let op = match self.previous_kind {
                TokenKind::Star => BinaryOperator::Multiply,
                TokenKind::Slash => BinaryOperator::Divide,
                TokenKind::Percent => BinaryOperator::Modulo,
                TokenKind::BitAnd => BinaryOperator::BitwiseAnd,
                _ => unreachable!(),
            };
            let rhs = self.parse_unary()?;
            
            let span = Span::new(lhs.span.start, rhs.span.end);
            lhs = Expression {
                id: self.next_id(),
                kind: ExpressionKind::Binary { lhs: Box::new(lhs), op, rhs: Box::new(rhs) },
                span,
            };
        }
        Ok(lhs)
    }

    /// Level 7: Unary -> ("-" | "!") Unary | Postfix
    fn parse_unary(&mut self) -> ParseResult<Expression> {
        // 检查前缀运算符
        if self.match_token(&[TokenKind::Minus, TokenKind::Bang]) {
            let start = self.previous_span().start;
            let op = match self.previous_kind {
                TokenKind::Minus => UnaryOperator::Negate,
                TokenKind::Bang => UnaryOperator::Not,
                _ => unreachable!(),
            };
            
            // 递归解析右侧 (允许连续: !!x, --x)
            let operand = self.parse_unary()?;
            let span = Span::new(start, operand.span.end);
            
            return Ok(Expression {
                id: self.next_id(),
                kind: ExpressionKind::Unary { op, operand: Box::new(operand) },
                span,
            });
        }

        // 如果不是前缀，解析后缀/Primary
        self.parse_postfix()
    }

    /// Level 8: Postfix -> Primary { ... }
    /// 这是最复杂的部分，处理调用、索引、字段访问、Cast、后缀解引用
    fn parse_postfix(&mut self) -> ParseResult<Expression> {
        let mut expr = self.parse_primary()?;

        loop {
            if self.match_token(&[TokenKind::LParen]) {
                // 1. 函数调用: expr(args)
                let args = self.parse_arguments()?; // 辅助函数
                let end_token = self.expect(TokenKind::RParen)?;
                let span = Span::new(expr.span.start, end_token.span.end);
                
                expr = Expression {
                    id: self.next_id(),
                    kind: ExpressionKind::Call { callee: Box::new(expr), arguments: args },
                    span,
                };

            } else if self.match_token(&[TokenKind::LBracket]) {
                // 2. 索引: expr[index]
                let index = self.parse_expr()?;
                let end_token = self.expect(TokenKind::RBracket)?;
                let span = Span::new(expr.span.start, end_token.span.end);

                expr = Expression {
                    id: self.next_id(),
                    kind: ExpressionKind::Index { target: Box::new(expr), index: Box::new(index) },
                    span,
                };

            } else if self.match_token(&[TokenKind::Dot]) {
                // 3. 字段访问: expr.ident
                let name = self.expect(TokenKind::Identifier)?;
                let ident = Identifier { name: self.text(name).to_string(), span: name.span };
                
                // 检查是否是方法调用: expr.method(...)
                // EBNF: Postfix -> "." Identifier. 但 AST 里区分了 MethodCall 和 FieldAccess
                // 我们可以 Lookahead 看下一个是不是 '('
                if self.check(TokenKind::LParen) {
                    self.advance(); // 吃掉 '('
                    let args = self.parse_arguments()?;
                    let end_token = self.expect(TokenKind::RParen)?;
                    let span = Span::new(expr.span.start, end_token.span.end);

                    expr = Expression {
                        id: self.next_id(),
                        kind: ExpressionKind::MethodCall { 
                            receiver: Box::new(expr), 
                            method_name: ident, 
                            arguments: args 
                        },
                        span,
                    };
                } else {
                    // 只是字段访问
                    let span = Span::new(expr.span.start, name.span.end);
                    expr = Expression {
                        id: self.next_id(),
                        kind: ExpressionKind::FieldAccess { receiver: Box::new(expr), field_name: ident },
                        span,
                    };
                }

            } else if self.match_token(&[TokenKind::As]) {
                // 4. 类型转换: expr as Type
                let target_type = self.parse_type()?;
                let span = Span::new(expr.span.start, target_type.span.end);
                
                expr = Expression {
                    id: self.next_id(),
                    kind: ExpressionKind::Cast { expr: Box::new(expr), target_type },
                    span,
                };

            } else if self.match_token(&[TokenKind::Caret]) {
                // 5. 后缀解引用: ptr^
                let span = Span::new(expr.span.start, self.previous_span().end);
                expr = Expression {
                    id: self.next_id(),
                    kind: ExpressionKind::Unary { op: UnaryOperator::Dereference, operand: Box::new(expr) },
                    span,
                };

            } else if self.match_token(&[TokenKind::Ampersand]) {
                // 6. 后缀取地址: val&
                let span = Span::new(expr.span.start, self.previous_span().end);
                expr = Expression {
                    id: self.next_id(),
                    kind: ExpressionKind::Unary { op: UnaryOperator::AddressOf, operand: Box::new(expr) },
                    span,
                };
            
            } else if self.match_token(&[TokenKind::ColonColon]) {
                // 7. 命名空间访问 (EBNF: :: Identifier)
                let name = self.expect(TokenKind::Identifier)?;
                let ident = Identifier { name: self.text(name).to_string(), span: name.span };
                let span = Span::new(expr.span.start, name.span.end);
                
                // 明确使用 StaticAccess
                expr = Expression {
                    id: self.next_id(),
                    kind: ExpressionKind::StaticAccess { 
                        target: Box::new(expr), 
                        member: ident 
                     },
                     span,
                };
            } else {
                break;
            }
        }
        Ok(expr)
    }

    /// Level 9: Primary -> Literal | Path | StructLiteral | "(" Expr ")"
    fn parse_primary(&mut self) -> ParseResult<Expression> {
        let token = self.peek();

        match token.kind {
            // 字面量
            TokenKind::Integer | TokenKind::Float | TokenKind::StringLit | 
            TokenKind::CharLit | TokenKind::True | TokenKind::False => {
                self.advance();
                let lit = self.token_to_literal(token)?;
                Ok(Expression {
                    id: self.next_id(),
                    kind: ExpressionKind::Literal(lit),
                    span: token.span,
                })
            }

            // 括号表达式
            TokenKind::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                // Span 应该包含括号吗？通常是的
                Ok(expr) // 这里返回内部 expr，也可以包裹一个 ParenExpr 节点
            }

            // 路径 OR 结构体初始化
            // 两者都以 Identifier 开头：
            // Path:   std::io::File
            // Struct: std::io::File { fd: 1 }
            TokenKind::Identifier => {
                let path = self.parse_path()?; // 先解析完整的 Path
                
                // 歧义解决：如果后面跟着 '{'，则是结构体初始化
                if self.check(TokenKind::LBrace) {
                    self.parse_struct_literal(path)
                } else {
                    // 否则就是纯路径（变量名/枚举值）
                    Ok(Expression {
                        id: self.next_id(),
                        kind: ExpressionKind::Path(path.clone()),
                        span: path.span,
                    })
                }
            }

            //  self 作为表达式 
            TokenKind::SelfVal => {
                let tok = self.advance();
                Ok(Expression {
                    id: self.next_id(),
                    // 将 self 视为名为 "self" 的 Path
                    kind: ExpressionKind::Path(Path { 
                        segments: vec![Identifier { name: "self".to_string(), span: tok.span }], 
                        span: tok.span 
                    }),
                    span: tok.span,
                })
            }

            _ => Err(ParseError {
                expected: "Expression".into(),
                found: token.kind,
                span: token.span,
                message: "Expected expression".into(),
            }),
        }
    }

    // ==========================================
    // 表达式辅助函数
    // ==========================================

    /// 解析函数参数列表: expr, expr, expr (不包含括号)
    fn parse_arguments(&mut self) -> ParseResult<Vec<Expression>> {
        let mut args = Vec::new();
        if !self.check(TokenKind::RParen) {
            loop {
                args.push(self.parse_expr()?);
                if !self.match_token(&[TokenKind::Comma]) {
                    break;
                }
            }
        }
        Ok(args)
    }

    /// 解析结构体字面量: Path { field: val, ... }
    fn parse_struct_literal(&mut self, type_name: Path) -> ParseResult<Expression> {
        let start = type_name.span.start;
        self.expect(TokenKind::LBrace)?;
        
        let mut fields = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let field_name_tok = self.expect(TokenKind::Identifier)?;
            self.expect(TokenKind::Colon)?;
            let value = self.parse_expr()?;
            
            fields.push(StructFieldInit {
                field_name: Identifier { 
                    name: self.text(field_name_tok).to_string(), 
                    span: field_name_tok.span 
                },
                value,
                span: Span::new(field_name_tok.span.start, self.previous_span().end),
            });

            if !self.match_token(&[TokenKind::Comma]) {
                break;
            }
        }
        
        let end_tok = self.expect(TokenKind::RBrace)?;
        
        Ok(Expression {
            id: self.next_id(),
            kind: ExpressionKind::StructLiteral { type_name, fields },
            span: Span::new(start, end_tok.span.end),
        })
    }

    fn token_to_literal(&self, token: Token) -> ParseResult<Literal> {
        let text = self.text(token);
        match token.kind {
            TokenKind::True => Ok(Literal::Boolean(true)),
            TokenKind::False => Ok(Literal::Boolean(false)),
            TokenKind::Integer => {
                // 处理 0xFF, 1_000 等，Rust parse 支持部分，
                let val = text.replace('_', "").parse::<u64>().map_err(|_| ParseError{
                    expected: "Integer".into(), found: token.kind, span: token.span, message: "Invalid integer".into()
                })?;
                Ok(Literal::Integer(val))
            },
            TokenKind::Float => {
                let val = text.replace('_', "").parse::<f64>().unwrap();
                Ok(Literal::Float(val))
            },
            TokenKind::StringLit => {
                // 去掉引号
                let content = &text[1..text.len()-1]; 
                Ok(Literal::String(content.to_string()))
            },
            TokenKind::CharLit => {
                let content = text.chars().nth(1).unwrap_or('\0');
                Ok(Literal::Char(content))
            },
            _ => unreachable!(),
        }
    }
}

impl<'a> Parser<'a> {
    // ==========================================
    // 语句解析 (Statement Parsing)
    // ==========================================

    /// Statement -> Decl | Assign | Call | If | While | Break | Continue | Ret | Block | Switch
    pub fn parse_statement(&mut self) -> ParseResult<Statement> {
        // 1. 变量声明 (Set / Mut / Const)
        if self.match_token(&[TokenKind::Set, TokenKind::Mut, TokenKind::Const]) {
            let modifier = self.previous_kind; // match_token 已经 advance 了
            return self.parse_decl_stmt(modifier);
        }

        // 2. 块 (Block)
        if self.check(TokenKind::LBrace) {
            // parse_block 内部已经处理了 { } 的消耗和 Span 的计算
            let block = self.parse_block()?;
            
            // 直接取出 block.span
            let span = block.span; 
            
            return Ok(Statement {
                kind: StatementKind::Block(block), // 这里的 kind 变成了 Block(Block)
                span,
            });
        }

        // 3. 流程控制
        if self.match_token(&[TokenKind::If]) { return self.parse_if_stmt(); }
        if self.match_token(&[TokenKind::While]) { return self.parse_while_stmt(); }
        if self.match_token(&[TokenKind::Switch]) { return self.parse_switch_stmt(); }
        if self.match_token(&[TokenKind::Ret]) { return self.parse_return_stmt(); }
        
        if self.match_token(&[TokenKind::Break]) {
            let start = self.previous_span().start;
            let end_tok = self.expect(TokenKind::Semi)?;
            return Ok(Statement {
                kind: StatementKind::Break,
                span: Span::new(start, end_tok.span.end),
            });
        }
        
        if self.match_token(&[TokenKind::Continue]) {
            let start = self.previous_span().start;
            let end_tok = self.expect(TokenKind::Semi)?;
            return Ok(Statement {
                kind: StatementKind::Continue,
                span: Span::new(start, end_tok.span.end),
            });
        }

        // 4. 剩下的歧义：AssignStmt vs CallStmt
        // 两者都以 Postfix 开头：
        // Assign: Postfix "=" Expr ";"
        // Call:   Postfix ";"
        self.parse_assign_or_call_stmt()
    }

    /// Block -> "{" { Statement } "}"
    /// 返回 Vec<Statement> 供其他结构（如函数体）复用
    fn parse_block(&mut self) -> ParseResult<Block> { // 返回类型变了
        let start = self.expect(TokenKind::LBrace)?.span.start;
        
        let mut stmts = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            match self.parse_statement() {
                Ok(stmt) => stmts.push(stmt),
                Err(e) => {
                    self.errors.push(e);
                    self.synchronize();
                }
            }
        }
        
        let end = self.expect(TokenKind::RBrace)?.span.end;
        
        Ok(Block {
            stmts,
            span: Span::new(start, end),
        })
    }

    // DeclStmt -> VarModifier Identifier ":" Type [ "=" Expression ] ";"
    fn parse_decl_stmt(&mut self, modifier_kind: TokenKind) -> ParseResult<Statement> {
        let start = self.previous_span().start;
        
        // 1. Modifier
        let modifier = match modifier_kind {
            TokenKind::Set => Mutability::Immutable, // set x: T;
            TokenKind::Mut => Mutability::Mutable,
            TokenKind::Const => Mutability::Constant,
            _ => unreachable!(),
        };

        // 2. Identifier
        let name_tok = self.expect(TokenKind::Identifier)?;
        let name = Identifier { name: self.text(name_tok).to_string(), span: name_tok.span };

        // 3. Type
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type()?;

        // 4. Initializer (Optional)
        let mut initializer = None;
        if self.match_token(&[TokenKind::Eq]) {
            initializer = Some(self.parse_expr()?);
        }

        // 5. Semicolon
        let end_tok = self.expect(TokenKind::Semi)?;

        Ok(Statement {
            kind: StatementKind::VariableDeclaration {
                modifier,
                name,
                type_annotation: ty,
                initializer,
            },
            span: Span::new(start, end_tok.span.end),
        })
    }

    // IfStmt -> "if" "(" Expression ")" Block [ "else" ( Block | IfStmt ) ]
    fn parse_if_stmt(&mut self) -> ParseResult<Statement> {
        let start = self.previous_span().start;

        self.expect(TokenKind::LParen)?;
        let condition = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;

        // === 修正点 1：直接获取 Block 结构 ===
        // parse_block 返回 ParseResult<Block>
        let then_block = self.parse_block()?; 

        let mut else_branch = None;
        if self.match_token(&[TokenKind::Else]) {
            if self.check(TokenKind::If) {
                self.advance(); // consume 'if'
                else_branch = Some(Box::new(self.parse_if_stmt()?));
            } else {
                // else { ... }
                // === 修正点 2：else 后面如果是块，也需要包成 Statement ===
                // 因为 AST 里 else_branch 是 Option<Box<Statement>>
                let block = self.parse_block()?;
                let span = block.span;
                else_branch = Some(Box::new(Statement {
                    kind: StatementKind::Block(block),
                    span,
                }));
            }
        }

        // 计算整个 if 的 span
        let end = else_branch.as_ref().map(|s| s.span.end).unwrap_or(then_block.span.end);

        Ok(Statement {
            kind: StatementKind::If {
                condition,
                then_block, // 类型匹配：Block
                else_branch,
            },
            span: Span::new(start, end),
        })
    }

    // WhileStmt -> "while" "(" Expression ")" [ ":" DeclStmt ] Block
    fn parse_while_stmt(&mut self) -> ParseResult<Statement> {
        let start = self.previous_span().start;
        
        self.expect(TokenKind::LParen)?;
        let condition = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;

        // Optional Loop Variable: [ ":" DeclStmt ]
        let mut init_statement = None;
        if self.match_token(&[TokenKind::Colon]) {
            // EBNF: DeclStmt -> VarModifier ...
            // 我们需要 check 下一个是不是 VarModifier
            if self.match_token(&[TokenKind::Set, TokenKind::Mut, TokenKind::Const]) {
                let modifier = self.previous_kind;
                init_statement = Some(Box::new(self.parse_decl_stmt(modifier)?));
            } else {
                return Err(ParseError {
                    expected: "VarModifier (set/mut/const)".into(),
                    found: self.peek().kind,
                    span: self.peek().span,
                    message: "Expected declaration after ':' in while loop".into(),
                });
            }
        }

        let body = self.parse_block()?;
        
        // 结束位置直接取 body.span.end
        let end = body.span.end;
        
        Ok(Statement {
            kind: StatementKind::While {
                condition,
                init_statement,
                body, // 类型匹配：Block
            },
            span: Span::new(start, end),
        })
    }

    // ReturnStmt -> "ret" [ Expression ] ";"
    fn parse_return_stmt(&mut self) -> ParseResult<Statement> {
        let start = self.previous_span().start;
        
        let mut value = None;
        if !self.check(TokenKind::Semi) {
            value = Some(self.parse_expr()?);
        }
        
        let end_tok = self.expect(TokenKind::Semi)?;
        
        Ok(Statement {
            kind: StatementKind::Return(value),
            span: Span::new(start, end_tok.span.end),
        })
    }

    // SwitchStmt -> "switch" "(" Expression ")" "{" { SwitchCase } [ DefaultCase ] "}"
    fn parse_switch_stmt(&mut self) -> ParseResult<Statement> {
        let start = self.previous_span().start;
        
        self.expect(TokenKind::LParen)?;
        let target = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;
        
        self.expect(TokenKind::LBrace)?;
        
        let mut cases = Vec::new();
        let mut default_case = None;

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            // 检查是不是 default 分支
            if self.match_token(&[TokenKind::Default]) {
                if default_case.is_some() {
                    return Err(ParseError {
                        expected: "End of switch".into(), 
                        found: TokenKind::Default, 
                        span: self.previous_span(),
                        message: "Multiple default cases found".into() 
                    });
                }
                self.expect(TokenKind::Arrow)?;
                
                // 解析语句 (可能是 Block，也可能是单行)
                let stmt = self.parse_statement()?;
                
                // === 关键修正：归一化为 Block 结构 ===
                let block = self.statement_to_block(stmt);
                default_case = Some(block);

            } else {
                // 普通 Case
                let mut patterns = Vec::new();
                let first = self.parse_expr()?;
                patterns.push(first.clone()); // 这里的 clone 视 Expression 实现而定，或者用 span 记录起点
                let pattern_start = first.span.start;

                while self.match_token(&[TokenKind::Pipe]) {
                    patterns.push(self.parse_expr()?);
                }
                
                self.expect(TokenKind::Arrow)?;
                let stmt = self.parse_statement()?;

                // === 关键修正：归一化为 Block 结构 ===
                let block = self.statement_to_block(stmt);
                
                // Case 的 Span 包含 patterns + body
                let span = Span::new(pattern_start, block.span.end);
                
                cases.push(SwitchCase {
                    patterns,
                    body: block, // 现在是 Block 类型
                    span,
                });
            }
        }
        
        let end_tok = self.expect(TokenKind::RBrace)?;
        
        Ok(Statement {
            kind: StatementKind::Switch { target, cases, default_case },
            span: Span::new(start, end_tok.span.end),
        })
    }

    // === 辅助工具：将任意 Statement 转换为 Block ===
    // 如果它本来就是 Block，直接拆包取出；
    // 如果它是单行语句，包装成 Block。
    fn statement_to_block(&self, stmt: Statement) -> Block {
        match stmt.kind {
            // 如果本身就是 Block (即用户写了 { ... })，直接用里面的 Block 结构
            StatementKind::Block(b) => b,
            
            // 否则 (用户写了 -> ret 0;)，人工构造成 Block
            _ => Block {
                span: stmt.span, // Block 的范围就是这单行语句的范围
                stmts: vec![stmt],
            }
        }
    }

    // 处理 AssignStmt 或 CallStmt
    // 共同前缀：Postfix
    fn parse_assign_or_call_stmt(&mut self) -> ParseResult<Statement> {
        let start = self.peek().span.start;
        
        // 1. 严格按照 EBNF: Postfix
        // 注意：parse_postfix 只解析到 Primary + Suffixes。
        // 它不会解析 1+1 (Binary)。
        // 这符合 EBNF：AssignStmt -> Postfix "=" Expr ";"
        // 这意味着 `a + b = 1;` 在语法层面就会报错，这很好。
        let expr = self.parse_postfix()?;
        
        // 2. 分歧判断
        if self.match_token(&[TokenKind::Eq]) {
            // 是赋值: Postfix "=" Expression ";"
            let rhs = self.parse_expr()?;
            let end_tok = self.expect(TokenKind::Semi)?;
            
            Ok(Statement {
                kind: StatementKind::Assignment { lhs: expr, rhs },
                span: Span::new(start, end_tok.span.end),
            })
        } else {
            // 是调用/表达式语句: Postfix ";"
            let end_tok = self.expect(TokenKind::Semi)?;
            
            // 语义检查预告：
            // 虽然语法允许 `1;` 或 `a.b;`，但通常我们只允许 Call 类型的表达式作为语句。
            // 这可以在 Analyzer 阶段检查：
            // "Error: Statement must be an assignment or a function call."
            
            Ok(Statement {
                kind: StatementKind::ExpressionStatement(expr),
                span: Span::new(start, end_tok.span.end),
            })
        }
    }
}

impl<'a> Parser<'a> {
    // ==========================================
    // 顶层定义解析 (Top Level Parsing)
    // ==========================================

    /// Program -> { TopLevelDecl }
    pub fn parse_program(&mut self) -> Program {
        let mut items = Vec::new();
        while !self.is_at_end() {
            match self.parse_top_level() {
                Ok(item) => items.push(item),
                Err(e) => {
                    self.errors.push(e);
                    self.synchronize();
                }
            }
        }
        Program { items }
    }

    /// TopLevelDecl 分发逻辑
    fn parse_top_level(&mut self) -> ParseResult<Item> {
        // 1. 处理 Visibility (pub)
        // 注意：imp 不带 pub，其他大多可以带
        let is_pub = self.match_token(&[TokenKind::Pub]);
        let start_span = if is_pub { self.previous_span() } else { self.peek().span };

        // 2. 根据关键字分发
        let token = self.peek();
        match token.kind {
            TokenKind::Mod => self.parse_module_decl(is_pub, start_span.start),
            TokenKind::Use => self.parse_import_decl(is_pub, start_span.start),
            TokenKind::Struct => self.parse_struct_decl(is_pub, start_span.start),
            TokenKind::Union => self.parse_union_decl(is_pub, start_span.start),
            TokenKind::Enum => self.parse_enum_decl(is_pub, start_span.start),
            TokenKind::Fn => self.parse_fn_decl(is_pub, start_span.start),
            TokenKind::Typedef => self.parse_typedef_decl(is_pub, start_span.start),
            TokenKind::Typealias => self.parse_typealias_decl(is_pub, start_span.start),
            
            // "imp" 不能带 pub (根据 EBNF)，如果这里前面解析到了 pub，是语法错误
            TokenKind::Imp => {
                if is_pub {
                    return Err(ParseError {
                        expected: "imp without pub".into(), found: TokenKind::Imp, span: start_span,
                        message: "'imp' blocks cannot be public, only their methods can.".into()
                    });
                }
                self.parse_imp_decl(start_span.start)
            }
            
            _ => Err(ParseError {
                expected: "Top level declaration".into(),
                found: token.kind,
                span: token.span,
                message: format!("Unexpected token at top level: {:?}", token.kind),
            }),
        }
    }

    // ModuleDecl -> [ "pub" ] "mod" Identifier ";"
    fn parse_module_decl(&mut self, is_pub: bool, start: usize) -> ParseResult<Item> {
        self.expect(TokenKind::Mod)?;
        let name = self.parse_identifier()?;
        let end = self.expect(TokenKind::Semi)?.span.end;
        
        Ok(Item {
            kind: ItemKind::ModuleDecl { name, is_pub, items: None},
            span: Span::new(start, end),
        })
    }

    // ImportDecl -> [ "pub" ] "use" Path [ "as" Identifier ] ";"
    fn parse_import_decl(&mut self, is_pub: bool, start: usize) -> ParseResult<Item> {
        self.expect(TokenKind::Use)?;
        let path = self.parse_path()?;
        
        let mut alias = None;
        if self.match_token(&[TokenKind::As]) {
            alias = Some(self.parse_identifier()?);
        }
        
        let end = self.expect(TokenKind::Semi)?.span.end;
        
        Ok(Item {
            kind: ItemKind::Import { path, alias, is_pub },
            span: Span::new(start, end),
        })
    }

    // TypedefDecl -> "typedef" Identifier "=" Type ";"
    fn parse_typedef_decl(&mut self, _is_pub: bool, start: usize) -> ParseResult<Item> {
        self.expect(TokenKind::Typedef)?;
        let name = self.parse_identifier()?;
        self.expect(TokenKind::Eq)?;
        let target_type = self.parse_type()?;
        let end = self.expect(TokenKind::Semi)?.span.end;

        Ok(Item {
            kind: ItemKind::Typedef { name, target_type },
            span: Span::new(start, end),
        })
    }

    // TypeAliasDecl -> "typealias" Identifier "=" Type ";"
    fn parse_typealias_decl(&mut self, _is_pub: bool, start: usize) -> ParseResult<Item> {
        self.expect(TokenKind::Typealias)?;
        let name = self.parse_identifier()?;
        self.expect(TokenKind::Eq)?;
        let target_type = self.parse_type()?;
        let end = self.expect(TokenKind::Semi)?.span.end;

        Ok(Item {
            kind: ItemKind::TypeAlias { name, target_type },
            span: Span::new(start, end),
        })
    }

    // FnDecl -> ["pub"] "fn" Identifier "(" [ ParamList ] ")" [ "->" Type ] Block 
    fn parse_fn_decl(&mut self, is_pub: bool, start: usize) -> ParseResult<Item> {
        // 调用通用的函数解析器，allow_self = false
        let func_def = self.parse_function_definition(is_pub, start, None)?; 
        
        Ok(Item {
            span: func_def.span,
            kind: ItemKind::FunctionDecl(func_def),
        })
    }

    // ==========================================
    // 结构体/联合体解析 (Refactored)
    // ==========================================

    // StructDecl -> "struct" [ "(" INT ")" ] Identifier "{" ... "}"
    fn parse_struct_decl(&mut self, _is_pub: bool, start: usize) -> ParseResult<Item> {
        // 调用通用解析逻辑，传入 Struct 关键字
        let def = self.parse_record_definition(TokenKind::Struct, start)?;
        let end = def.span.end;

        Ok(Item {
            kind: ItemKind::StructDecl(def),
            span: Span::new(start, end),
        })
    }

    // UnionDecl -> "union" [ "(" INT ")" ] Identifier "{" ... "}"
    // 现在 Union 也完全支持对齐语法了！
    fn parse_union_decl(&mut self, _is_pub: bool, start: usize) -> ParseResult<Item> {
        // 调用通用解析逻辑，传入 Union 关键字
        let def = self.parse_record_definition(TokenKind::Union, start)?;
        let end = def.span.end;

        Ok(Item {
            kind: ItemKind::UnionDecl(def),
            span: Span::new(start, end),
        })
    }

    /// 通用的“混合体”解析器
    /// 适用场景：Struct, Union, Enum
    /// 逻辑：解析 `{` -> 循环(静态方法 OR 数据成员) -> `}`
    ///
    /// 参数 `parse_item`: 一个闭包，负责解析具体的数据成员 (Field 或 Variant)
    /// 返回值: (数据成员列表, 静态方法列表, 整个块的 Span)
    fn parse_mixed_body<T, F>(
        &mut self,
        parse_item: F,
    ) -> ParseResult<(Vec<T>, Vec<FunctionDefinition>, Span)>
    where
        F: Fn(&mut Self) -> ParseResult<T>,
    {
        let start = self.expect(TokenKind::LBrace)?.span.start;

        let mut items = Vec::new();
        let mut static_methods = Vec::new();

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            // 1. 先检查是不是静态方法 (fn 或 pub fn)
            // 逻辑完全复用：Struct/Union/Enum 里的静态方法长得都一样
            let is_fn = self.check(TokenKind::Fn) || 
                        (self.check(TokenKind::Pub) && self.check_nth(1, TokenKind::Fn));

            if is_fn {
                let is_method_pub = self.match_token(&[TokenKind::Pub]);
                let fn_start = if is_method_pub { self.previous_span().start } else { self.peek().span.start };
                // 静态方法不允许 self
                static_methods.push(self.parse_function_definition(is_method_pub, fn_start, None)?);
            } else {
                // 2. 如果不是方法，那就是数据成员 (Field 或 Variant)
                // 调用传入的闭包来解析它
                items.push(parse_item(self)?);
            }
        }

        let end = self.expect(TokenKind::RBrace)?.span.end;
        Ok((items, static_methods, Span::new(start, end)))
    }

    // === 核心公共逻辑：解析类似 Struct 的定义 ===
    // 负责解析: Keyword [Alignment] Identifier "{" { Members } "}"
    // 负责解析: Keyword [Alignment] Identifier "{" ... "}"
    fn parse_record_definition(&mut self, keyword: TokenKind, start: usize) -> ParseResult<StructDefinition> {
        self.expect(keyword)?;

        // 1. Header 部分
        let alignment = self.parse_optional_alignment()?;
        let name = self.parse_identifier()?;
        
        // 2. Body 部分 (使用通用解析器)
        // 闭包负责解析: Identifier ":" Type ";"
        let (fields, static_methods, body_span) = self.parse_mixed_body(|p| {
            let field_name = p.parse_identifier()?;
            p.expect(TokenKind::Colon)?;
            let ty = p.parse_type()?;
            let semi = p.expect(TokenKind::Semi)?;
            Ok(FieldDefinition {
                name: field_name.clone(),
                ty,
                span: Span::new(field_name.span.start, semi.span.end),
            })
        })?;

        Ok(StructDefinition {
            name,
            fields,
            static_methods,
            alignment,
            span: Span::new(start, body_span.end),
        })
    }

    // === 辅助：解析可选对齐 ===
    // 格式: [ "(" INT ")" ]
    fn parse_optional_alignment(&mut self) -> ParseResult<Option<u32>> {
        if self.match_token(&[TokenKind::LParen]) {
            let int_tok = self.expect(TokenKind::Integer)?;
            let val_str = self.text(int_tok);
            
            let val = val_str.parse::<u32>().map_err(|_| ParseError {
                expected: "Integer (u32)".into(),
                found: TokenKind::Integer,
                span: int_tok.span,
                message: "Alignment value is too large".into()
            })?;

            if val == 0 {
                return Err(ParseError {
                    expected: "Positive Integer".into(), found: TokenKind::Integer, span: int_tok.span,
                    message: "Alignment cannot be 0".into()
                });
            }
            
            // 额外检查：是否是 2 的幂 (位运算技巧: x & (x-1) == 0 表示只有一位是1)
            if (val & (val - 1)) != 0 {
                 return Err(ParseError {
                    expected: "Power of 2".into(), found: TokenKind::Integer, span: int_tok.span,
                    message: format!("Alignment must be a power of 2, but found {}", val)
                });
            }

            self.expect(TokenKind::RParen)?;
            Ok(Some(val))
        } else {
            Ok(None)
        }
    }

    // EnumDecl -> "enum" Identifier [ ":" IntType ] "{" { EnumMember } "}"
    fn parse_enum_decl(&mut self, _is_pub: bool, start: usize) -> ParseResult<Item> {
        self.expect(TokenKind::Enum)?;
        let name = self.parse_identifier()?;
        
        // 1. Header 部分: [ ":" IntType ]
        let mut underlying_type = None;
        if self.match_token(&[TokenKind::Colon]) {
            let ty = self.parse_type()?;
            if let TypeKind::Primitive(p) = ty.kind {
                 underlying_type = Some(p);
            } else {
                 return Err(ParseError { 
                     expected: "Integer Type".into(), found: TokenKind::Identifier, span: ty.span,
                     message: "Enum backing type must be primitive integer".into() 
                 });
            }
        }

        // 2. Body 部分 (使用通用解析器)
        // 闭包负责解析: Identifier [ "=" INT ] ";"
        let (variants, static_methods, body_span) = self.parse_mixed_body(|p| {
            let v_name = p.parse_identifier()?;
            let mut value = None;
            if p.match_token(&[TokenKind::Eq]) {
                value = Some(p.parse_expr()?);
            }
            let semi = p.expect(TokenKind::Semi)?;
            
            Ok(EnumVariant {
                name: v_name.clone(),
                value,
                span: Span::new(v_name.span.start, semi.span.end),
            })
        })?;

        Ok(Item {
            kind: ItemKind::EnumDecl(EnumDefinition {
                name, 
                underlying_type, 
                variants, 
                static_methods, 
                span: Span::new(start, body_span.end)
            }),
            span: Span::new(start, body_span.end),
        })
    }

    // TypeImpDecl -> "imp" "for" Type "{" { MethodDecl } "}"
    fn parse_imp_decl(&mut self, start: usize) -> ParseResult<Item> {
        self.expect(TokenKind::Imp)?;
        self.expect(TokenKind::For)?;
        
        // 1. 解析目标类型 (此时拥有所有权)
        let target_type = self.parse_type()?;
        
        self.expect(TokenKind::LBrace)?;
        
        let mut methods = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let is_pub = self.match_token(&[TokenKind::Pub]);
            let fn_start = if is_pub { self.previous_span().start } else { self.peek().span.start };
            methods.push(self.parse_function_definition(is_pub, fn_start, Some(&target_type))?);
        }
        
        let end = self.expect(TokenKind::RBrace)?.span.end;

        Ok(Item {
            // 3. 最后这里 move target_type
            kind: ItemKind::Implementation { target_type, methods },
            span: Span::new(start, end),
        })
    }

    // ==========================================
    // 函数解析核心 (复用)
    // ==========================================

    /// 解析函数定义
    /// ctx_type: 如果是 imp 块的方法，这里包含 impl 的目标类型；否则为 None
    fn parse_function_definition(
        &mut self, 
        is_pub: bool, 
        start: usize, 
        ctx_type: Option<&Type> 
    ) -> ParseResult<FunctionDefinition> {
        self.expect(TokenKind::Fn)?;
        let name = self.parse_identifier()?;
        
        self.expect(TokenKind::LParen)?;
        
        // 传递上下文类型给参数列表解析器
        let params = self.parse_param_list(ctx_type)?;
        
        self.expect(TokenKind::RParen)?;
        
        let mut return_type = None;
        if self.match_token(&[TokenKind::Arrow]) {
            return_type = Some(self.parse_type()?);
        }
        
        let body = self.parse_block()?; 
        let body_span_end = body.span.end;

        Ok(FunctionDefinition {
            name,
            params,
            return_type,
            body,
            is_pub,
            span: Span::new(start, body_span_end),
        })
    }

    /// 解析参数列表 (Param | SelfParam)
    /// ctx_type: Some(Type) 表示允许 self 且类型为 Type；None 表示不允许 self。
    fn parse_param_list(&mut self, ctx_type: Option<&Type>) -> ParseResult<Vec<Parameter>> {
        let mut params = Vec::new();
        if self.check(TokenKind::RParen) {
            return Ok(params);
        }

        loop {
            // 检查 self
            let is_self_start = self.check(TokenKind::SelfVal) || 
                                ((self.check(TokenKind::Mut) || self.check(TokenKind::Const)) && self.check_nth(1, TokenKind::SelfVal));

            if is_self_start {
                // 1. 检查是否允许 self (通过 ctx_type 是否为 Some)
                let target_type = if let Some(ty) = ctx_type {
                    ty
                } else {
                    return Err(ParseError { 
                        expected: "Parameter".into(), 
                        found: TokenKind::SelfVal, 
                        span: self.peek().span, 
                        message: "self parameter is only allowed in 'imp' blocks".into() 
                    });
                };

                if !params.is_empty() {
                    return Err(ParseError { 
                        expected: "Parameter".into(), 
                        found: TokenKind::SelfVal, 
                        span: self.peek().span, 
                        message: "self must be the first parameter".into() 
                    });
                }

                // 2. 解析修饰符
                let mut modifier = Mutability::Immutable;
                if self.match_token(&[TokenKind::Mut]) { modifier = Mutability::Mutable; }
                else if self.match_token(&[TokenKind::Const]) { modifier = Mutability::Constant; }
                
                let self_tok = self.expect(TokenKind::SelfVal)?;
                
                // 3. 【核心修改】直接 Clone 传入的类型！
                // 这样 AST 里的 self 参数就直接拥有了正确的类型（如 Point, *Point）
                let real_type = target_type.clone(); 

                params.push(Parameter {
                    name: Identifier { name: "self".into(), span: self_tok.span },
                    ty: real_type, // <--- 注入真实类型
                    is_mutable: modifier == Mutability::Mutable,
                    is_self: true,
                });

            } else {
                // 普通参数解析保持不变
                let mut modifier = Mutability::Immutable;
                 if self.match_token(&[TokenKind::Mut]) { modifier = Mutability::Mutable; }
                else if self.match_token(&[TokenKind::Const]) { modifier = Mutability::Constant; }

                let p_name = self.parse_identifier()?;
                self.expect(TokenKind::Colon)?;
                let p_type = self.parse_type()?;
                
                params.push(Parameter {
                    name: p_name,
                    ty: p_type,
                    is_mutable: modifier == Mutability::Mutable,
                    is_self: false,
                });
            }

            if !self.match_token(&[TokenKind::Comma]) {
                break;
            }
        }
        Ok(params)
    }
    
    // 辅助: parse_identifier (之前可能还没写，这里补上)
    fn parse_identifier(&mut self) -> ParseResult<Identifier> {
        let tok = self.expect(TokenKind::Identifier)?;
        Ok(Identifier {
            name: self.text(tok).to_string(),
            span: tok.span,
        })
    }
}