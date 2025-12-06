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

use crate::ast::*;
use crate::lexer::Lexer;
use crate::source::Span;
use crate::token::{Token, TokenKind};
use std::collections::VecDeque;

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
    base_offset: usize,
}

impl<'a> TokenStream<'a> {
    fn new(lexer: Lexer<'a>, base_offset: usize) -> Self {
        Self {
            lexer,
            buffer: VecDeque::new(),

            last_span: Span::new(0, 0),
            base_offset,
        }
    }

    /// 确保缓冲区里至少有 n+1 个 Token (用于 peek(n))
    fn fill(&mut self, n: usize) {
        while self.buffer.len() <= n {
            let mut tok = self.lexer.next_token();

            if tok.kind != TokenKind::EOF {
                tok.span.start += self.base_offset;
                tok.span.end += self.base_offset;
            } else {
                tok.span.start += self.base_offset;
                tok.span.end += self.base_offset;
            }

            self.buffer.push_back(tok);
            if tok.kind == TokenKind::EOF {
                break;
            }
        }
    }

    /// 查看第 n 个 Token (不消耗)
    fn peek(&mut self, n: usize) -> Token {
        self.fill(n);

        *self
            .buffer
            .get(n)
            .unwrap_or_else(|| self.buffer.back().unwrap())
    }

    /// 消耗当前 Token 并前进
    fn advance(&mut self) -> Token {
        self.fill(0);
        let tok = self
            .buffer
            .pop_front()
            .unwrap_or_else(|| Token::new(TokenKind::EOF, self.last_span.end, self.last_span.end));
        self.last_span = tok.span;
        tok
    }
}

pub struct Parser<'a> {
    source: &'a str,
    stream: TokenStream<'a>,
    pub errors: Vec<ParseError>,

    previous_kind: TokenKind,
    node_id_counter: &'a mut u32,
}

impl<'a> Parser<'a> {
    pub fn new(
        source: &'a str,
        lexer: Lexer<'a>,
        base_offset: usize,
        id_counter: &'a mut u32,
    ) -> Self {
        Self {
            source,
            stream: TokenStream::new(lexer, base_offset),
            errors: Vec::new(),
            previous_kind: TokenKind::EOF,
            node_id_counter: id_counter,
        }
    }

    fn peek(&mut self) -> Token {
        self.stream.peek(0)
    }

    fn check(&mut self, kind: TokenKind) -> bool {
        self.peek().kind == kind
    }

    fn check_nth(&mut self, n: usize, kind: TokenKind) -> bool {
        self.stream.peek(n).kind == kind
    }

    fn is_at_end(&mut self) -> bool {
        self.peek().kind == TokenKind::EOF
    }

    fn advance(&mut self) -> Token {
        let tok = self.stream.advance();
        self.previous_kind = tok.kind;
        tok
    }

    fn previous_span(&self) -> Span {
        self.stream.last_span
    }

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
        self.advance();

        while !self.is_at_end() {
            if self.previous_kind == TokenKind::Semi {
                return;
            }

            match self.peek().kind {
                TokenKind::Struct
                | TokenKind::Union
                | TokenKind::Enum
                | TokenKind::Fn
                | TokenKind::Mod
                | TokenKind::Use
                | TokenKind::Imp
                | TokenKind::Typedef
                | TokenKind::Typealias
                | TokenKind::Pub
                | TokenKind::Set
                | TokenKind::Mut
                | TokenKind::Const
                | TokenKind::If
                | TokenKind::While
                | TokenKind::Switch
                | TokenKind::Ret
                | TokenKind::Break
                | TokenKind::Continue => {
                    return;
                }

                _ => {}
            }

            self.advance();
        }
    }

    /// 获取 Token 对应的源码文本
    pub fn text(&self, token: Token) -> &'a str {
        let offset = self.stream.base_offset;

        if token.span.start < offset {
            panic!(
                "Token start {} is smaller than base offset {}",
                token.span.start, offset
            );
        }

        let local_start = token.span.start - offset;
        let local_end = token.span.end - offset;

        &self.source[local_start..local_end]
    }

    /// 辅助函数：分配下一个 ID
    fn next_id(&mut self) -> NodeId {
        let id = *self.node_id_counter;
        *self.node_id_counter += 1;
        NodeId(id)
    }

    fn unescape_string(&self, raw: &str) -> String {
        let mut result = String::new();
        let mut chars = raw.chars();

        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    Some('0') => result.push('\0'),
                    Some(other) => {
                        result.push('\\');
                        result.push(other);
                    }
                    None => {
                        break;
                    }
                }
            } else {
                result.push(c);
            }
        }
        result
    }
}

/// ======================================================
/// 解析类型
/// ======================================================

impl<'a> Parser<'a> {
    /// EBNF: Type -> PointerType | ArrayOrAtom
    pub fn parse_type(&mut self) -> ParseResult<Type> {
        if self.check(TokenKind::Star) || self.check(TokenKind::Caret) {
            return self.parse_pointer_type();
        }

        self.parse_array_or_atom()
    }

    /// EBNF: PointerType -> ( "*" | "^" ) Type
    fn parse_pointer_type(&mut self) -> ParseResult<Type> {
        let op_token = self.advance();
        let start = op_token.span.start;

        let inner_type = self.parse_type()?;
        let end = inner_type.span.end;

        let mutability = if op_token.kind == TokenKind::Star {
            Mutability::Mutable
        } else {
            Mutability::Constant
        };

        Ok(Type {
            id: self.next_id(),
            kind: TypeKind::Pointer {
                inner: Box::new(inner_type),
                mutability,
            },
            span: Span::new(start, end),
        })
    }

    /// EBNF: ArrayOrAtom -> AtomType { "[" INT "]" }
    fn parse_array_or_atom(&mut self) -> ParseResult<Type> {
        let mut ty = self.parse_atom_type()?;

        while self.match_token(&[TokenKind::LBracket]) {
            let size_token = self.expect(TokenKind::Integer)?;

            let size_str = self.text(size_token);
            let size = size_str.parse::<u64>().map_err(|_| ParseError {
                expected: "Integer that fits in u64".into(),
                found: size_token.kind,
                span: size_token.span,
                message: "Array size is too large".into(),
            })?;

            let end_token = self.expect(TokenKind::RBracket)?;

            let new_span = Span::new(ty.span.start, end_token.span.end);
            ty = Type {
                id: self.next_id(),
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
            TokenKind::I8
            | TokenKind::U8
            | TokenKind::I16
            | TokenKind::U16
            | TokenKind::I32
            | TokenKind::U32
            | TokenKind::I64
            | TokenKind::U64
            | TokenKind::Isize
            | TokenKind::Usize
            | TokenKind::F32
            | TokenKind::F64
            | TokenKind::Bool => {
                self.advance();
                let prim = self.token_kind_to_primitive(token.kind);
                Ok(Type {
                    id: self.next_id(),
                    kind: TypeKind::Primitive(prim),
                    span,
                })
            }

            TokenKind::Fn => self.parse_function_type(),

            TokenKind::Identifier => {
                let path = self.parse_path()?;
                let end = path.span.end;
                Ok(Type {
                    id: self.next_id(),
                    kind: TypeKind::Named(path),
                    span: Span::new(span.start, end),
                })
            }

            TokenKind::LParen => {
                self.advance();
                let inner = self.parse_type()?;
                let end_token = self.expect(TokenKind::RParen)?;
                Ok(Type {
                    id: self.next_id(),
                    kind: inner.kind,
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
        let start_token = self.expect(TokenKind::Fn)?;
        self.expect(TokenKind::LParen)?;

        let mut params = Vec::new();

        if !self.check(TokenKind::RParen) {
            loop {
                params.push(self.parse_type()?);
                if !self.match_token(&[TokenKind::Comma]) {
                    break;
                }
            }
        }

        self.expect(TokenKind::RParen)?;

        let mut ret_type = None;
        let mut end_pos = self.previous_span().end;

        if self.match_token(&[TokenKind::Arrow]) {
            let ret = self.parse_type()?;
            end_pos = ret.span.end;
            ret_type = Some(Box::new(ret));
        }

        Ok(Type {
            id: self.next_id(),
            kind: TypeKind::Function { params, ret_type },
            span: Span::new(start_token.span.start, end_pos),
        })
    }

    /// Path 解析：Ident [Generics] { :: Ident [Generics] }
    /// 修改后支持: List#<i32> 或 Map#<String, i32>::Entry
    pub fn parse_path(&mut self) -> ParseResult<Path> {
        let mut segments = Vec::new();
        let start_pos = self.peek().span.start;

        let first = self.expect(TokenKind::Identifier)?;
        let first_args = self.parse_generic_arguments()?;

        let mut last_end = first.span.end;

        if let Some(ref args) = first_args {
            if let Some(last_arg) = args.last() {
                last_end = self.previous_span().end;
            }
        }

        segments.push(PathSegment {
            name: self.text(first).to_string(),
            generic_args: first_args,
            span: Span::new(first.span.start, last_end),
        });

        while self.match_token(&[TokenKind::ColonColon]) {
            let seg = self.expect(TokenKind::Identifier)?;
            let seg_args = self.parse_generic_arguments()?;

            let mut seg_end = seg.span.end;
            if let Some(_) = seg_args {
                seg_end = self.previous_span().end;
            }

            segments.push(PathSegment {
                name: self.text(seg).to_string(),
                generic_args: seg_args,
                span: Span::new(seg.span.start, seg_end),
            });
        }

        let end_pos = segments.last().unwrap().span.end;
        Ok(Path {
            id: self.next_id(),
            segments,
            span: Span::new(start_pos, end_pos),
        })
    }

    /// 解析泛型参数定义 (Definition Site)
    /// 语法: "#" "<" GenericParam { "," GenericParam } ">"
    /// 例如: #<T, U: Printable>

    fn parse_generic_params(&mut self) -> ParseResult<Vec<GenericParam>> {
        if !self.check(TokenKind::Hash) {
            return Ok(Vec::new());
        }
        if !self.check_nth(1, TokenKind::Lt) {
            return Ok(Vec::new());
        }

        self.advance();
        self.expect(TokenKind::Lt)?;

        let mut params = Vec::new();

        while !self.check(TokenKind::Gt) && !self.check(TokenKind::Shr) && !self.is_at_end() {
            let name_tok = self.expect(TokenKind::Identifier)?;
            let name = Identifier {
                name: self.text(name_tok).to_string(),
                span: name_tok.span,
            };

            let mut constraints = Vec::new();
            if self.match_token(&[TokenKind::Colon]) {
                loop {
                    let cap_path = self.parse_path()?;
                    constraints.push(cap_path);

                    if !self.match_token(&[TokenKind::Plus]) {
                        break;
                    }
                }
            }

            let span = if let Some(last_cap) = constraints.last() {
                Span::new(name.span.start, last_cap.span.end)
            } else {
                name.span
            };

            params.push(GenericParam {
                id: self.next_id(),
                name,
                constraints,
                span,
            });

            if !self.match_token(&[TokenKind::Comma]) {
                break;
            }
        }

        self.consume_generic_closer()?;
        Ok(params)
    }

    /// 解析泛型实参 (Call Site / Type Usage)
    /// 语法: "#" "<" Type { "," Type } ">"
    /// 例如: #<i32, String>
    fn parse_generic_arguments(&mut self) -> ParseResult<Option<Vec<Type>>> {
        if !self.check(TokenKind::Hash) || !self.check_nth(1, TokenKind::Lt) {
            return Ok(None);
        }

        self.advance();
        self.expect(TokenKind::Lt)?;

        let mut args = Vec::new();

        while !self.check(TokenKind::Gt) && !self.check(TokenKind::Shr) && !self.is_at_end() {
            args.push(self.parse_type()?);

            if !self.match_token(&[TokenKind::Comma]) {
                break;
            }
        }

        self.consume_generic_closer()?;

        Ok(Some(args))
    }

    /// 专门用于消耗泛型结束符 '>'
    /// 如果遇到 '>>' (Shr)，会将其拆分为两个 '>'，消耗一个，把另一个塞回流中
    fn consume_generic_closer(&mut self) -> ParseResult<Token> {
        if self.check(TokenKind::Gt) {
            return self.advance_with_check();
        }

        if self.check(TokenKind::Shr) {
            let mut token = self.advance_with_check()?;

            token.kind = TokenKind::Gt;

            let original_end = token.span.end;
            token.span.end -= 1;

            let second_gt = Token {
                kind: TokenKind::Gt,
                span: Span::new(token.span.end, original_end),
            };

            self.stream.buffer.push_front(second_gt);

            self.previous_kind = TokenKind::Gt;

            return Ok(token);
        }

        self.expect(TokenKind::Gt)
    }

    fn advance_with_check(&mut self) -> ParseResult<Token> {
        Ok(self.advance())
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
            _ => unreachable!("Not a primitive token"),
        }
    }
}

impl<'a> Parser<'a> {
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
                kind: ExpressionKind::Binary {
                    lhs: Box::new(lhs),
                    op,
                    rhs: Box::new(rhs),
                },
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
                id: self.next_id(),
                kind: ExpressionKind::Binary {
                    lhs: Box::new(lhs),
                    op,
                    rhs: Box::new(rhs),
                },
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
                kind: ExpressionKind::Binary {
                    lhs: Box::new(lhs),
                    op,
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    /// Level 4: Relational -> Shift
    fn parse_relational(&mut self) -> ParseResult<Expression> {
        let mut lhs = self.parse_shift()?;

        while self.match_token(&[
            TokenKind::Lt,
            TokenKind::LtEq,
            TokenKind::Gt,
            TokenKind::GtEq,
        ]) {
            let op = match self.previous_kind {
                TokenKind::Lt => BinaryOperator::Less,
                TokenKind::LtEq => BinaryOperator::LessEqual,
                TokenKind::Gt => BinaryOperator::Greater,
                TokenKind::GtEq => BinaryOperator::GreaterEqual,
                _ => unreachable!(),
            };
            let rhs = self.parse_shift()?;

            let span = Span::new(lhs.span.start, rhs.span.end);
            lhs = Expression {
                id: self.next_id(),
                kind: ExpressionKind::Binary {
                    lhs: Box::new(lhs),
                    op,
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_shift(&mut self) -> ParseResult<Expression> {
        let mut lhs = self.parse_additive()?;

        while self.match_token(&[TokenKind::Shl, TokenKind::Shr]) {
            let op = match self.previous_kind {
                TokenKind::Shl => BinaryOperator::ShiftLeft,
                TokenKind::Shr => BinaryOperator::ShiftRight,
                _ => unreachable!(),
            };
            let rhs = self.parse_additive()?;
            let span = Span::new(lhs.span.start, rhs.span.end);
            lhs = Expression {
                id: self.next_id(),
                kind: ExpressionKind::Binary {
                    lhs: Box::new(lhs),
                    op,
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    /// Level 6: Additive -> Multiplicative { ("+"| "-" | "bor" | "xor") Multiplicative }
    fn parse_additive(&mut self) -> ParseResult<Expression> {
        let mut lhs = self.parse_multiplicative()?;

        while self.match_token(&[
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::BitOr,
            TokenKind::Xor,
        ]) {
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
                kind: ExpressionKind::Binary {
                    lhs: Box::new(lhs),
                    op,
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    /// Level 7: Multiplicative -> Unary { ("*" | "/" | "%" | "band") Unary }
    fn parse_multiplicative(&mut self) -> ParseResult<Expression> {
        let mut lhs = self.parse_unary()?;

        while self.match_token(&[
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::Percent,
            TokenKind::BitAnd,
        ]) {
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
                kind: ExpressionKind::Binary {
                    lhs: Box::new(lhs),
                    op,
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    /// Level 8: Unary -> ("-" | "!") Unary | Postfix
    fn parse_unary(&mut self) -> ParseResult<Expression> {
        if self.match_token(&[TokenKind::Minus, TokenKind::Bang]) {
            let start = self.previous_span().start;
            let op = match self.previous_kind {
                TokenKind::Minus => UnaryOperator::Negate,
                TokenKind::Bang => UnaryOperator::Not,
                _ => unreachable!(),
            };

            let operand = self.parse_unary()?;
            let span = Span::new(start, operand.span.end);

            return Ok(Expression {
                id: self.next_id(),
                kind: ExpressionKind::Unary {
                    op,
                    operand: Box::new(operand),
                },
                span,
            });
        }

        self.parse_postfix()
    }

    /// Level 9: Postfix -> Primary { ... }
    fn parse_postfix(&mut self) -> ParseResult<Expression> {
        let mut expr = self.parse_primary()?;

        loop {
            if self.match_token(&[TokenKind::LParen]) {
                let args = self.parse_arguments()?;
                let end_token = self.expect(TokenKind::RParen)?;
                let span = Span::new(expr.span.start, end_token.span.end);

                expr = Expression {
                    id: self.next_id(),
                    kind: ExpressionKind::Call {
                        callee: Box::new(expr),
                        arguments: args,
                    },
                    span,
                };
            } else if self.match_token(&[TokenKind::LBracket]) {
                let index = self.parse_expr()?;
                let end_token = self.expect(TokenKind::RBracket)?;
                let span = Span::new(expr.span.start, end_token.span.end);

                expr = Expression {
                    id: self.next_id(),
                    kind: ExpressionKind::Index {
                        target: Box::new(expr),
                        index: Box::new(index),
                    },
                    span,
                };
            } else if self.match_token(&[TokenKind::Dot]) {
                let name = self.expect(TokenKind::Identifier)?;
                let ident = Identifier {
                    name: self.text(name).to_string(),
                    span: name.span,
                };

                if self.check(TokenKind::LParen) {
                    self.advance();
                    let args = self.parse_arguments()?;
                    let end_token = self.expect(TokenKind::RParen)?;
                    let span = Span::new(expr.span.start, end_token.span.end);

                    expr = Expression {
                        id: self.next_id(),
                        kind: ExpressionKind::MethodCall {
                            receiver: Box::new(expr),
                            method_name: ident,
                            arguments: args,
                        },
                        span,
                    };
                } else {
                    let span = Span::new(expr.span.start, name.span.end);
                    expr = Expression {
                        id: self.next_id(),
                        kind: ExpressionKind::FieldAccess {
                            receiver: Box::new(expr),
                            field_name: ident,
                        },
                        span,
                    };
                }
            } else if self.match_token(&[TokenKind::As]) {
                let target_type = self.parse_type()?;
                let span = Span::new(expr.span.start, target_type.span.end);

                expr = Expression {
                    id: self.next_id(),
                    kind: ExpressionKind::Cast {
                        expr: Box::new(expr),
                        target_type,
                    },
                    span,
                };
            } else if self.match_token(&[TokenKind::Caret]) {
                let span = Span::new(expr.span.start, self.previous_span().end);
                expr = Expression {
                    id: self.next_id(),
                    kind: ExpressionKind::Unary {
                        op: UnaryOperator::Dereference,
                        operand: Box::new(expr),
                    },
                    span,
                };
            } else if self.match_token(&[TokenKind::Ampersand]) {
                let span = Span::new(expr.span.start, self.previous_span().end);
                expr = Expression {
                    id: self.next_id(),
                    kind: ExpressionKind::Unary {
                        op: UnaryOperator::AddressOf,
                        operand: Box::new(expr),
                    },
                    span,
                };
            } else if self.match_token(&[TokenKind::ColonColon]) {
                let name = self.expect(TokenKind::Identifier)?;
                let ident = Identifier {
                    name: self.text(name).to_string(),
                    span: name.span,
                };
                let span = Span::new(expr.span.start, name.span.end);

                expr = Expression {
                    id: self.next_id(),
                    kind: ExpressionKind::StaticAccess {
                        target: Box::new(expr),
                        member: ident,
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
            TokenKind::Integer
            | TokenKind::Float
            | TokenKind::StringLit
            | TokenKind::CharLit
            | TokenKind::True
            | TokenKind::False => {
                self.advance();
                let lit = self.token_to_literal(token)?;
                Ok(Expression {
                    id: self.next_id(),
                    kind: ExpressionKind::Literal(lit),
                    span: token.span,
                })
            }

            TokenKind::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                Ok(expr)
            }

            TokenKind::Identifier => {
                let path = self.parse_path()?;

                if self.check(TokenKind::LBrace) {
                    self.parse_struct_literal(path)
                } else {
                    Ok(Expression {
                        id: self.next_id(),
                        kind: ExpressionKind::Path(path.clone()),
                        span: path.span,
                    })
                }
            }

            TokenKind::SelfVal => {
                let tok = self.advance();
                Ok(Expression {
                    id: self.next_id(),

                    kind: ExpressionKind::Path(Path {
                        id: self.next_id(),
                        segments: vec![PathSegment {
                            name: "self".to_string(),
                            generic_args: None,
                            span: tok.span,
                        }],
                        span: tok.span,
                    }),
                    span: tok.span,
                })
            }

            TokenKind::At => {
                let start = self.advance().span.start;

                if self.match_token(&[TokenKind::SizeOf]) {
                    self.expect(TokenKind::LParen)?;
                    let target_type = self.parse_type()?;
                    let end = self.expect(TokenKind::RParen)?.span.end;

                    Ok(Expression {
                        id: self.next_id(),
                        kind: ExpressionKind::SizeOf(target_type),
                        span: Span::new(start, end),
                    })
                } else if self.match_token(&[TokenKind::AlignOf]) {
                    self.expect(TokenKind::LParen)?;
                    let target_type = self.parse_type()?;
                    let end = self.expect(TokenKind::RParen)?.span.end;

                    Ok(Expression {
                        id: self.next_id(),
                        kind: ExpressionKind::AlignOf(target_type),
                        span: Span::new(start, end),
                    })
                } else {
                    return Err(ParseError {
                        expected: "sizeof or alignof".into(),
                        found: self.peek().kind,
                        span: self.peek().span,
                        message: "Expected 'sizeof' or 'alignof' after '@'".into(),
                    });
                }
            }
            _ => Err(ParseError {
                expected: "Expression".into(),
                found: token.kind,
                span: token.span,
                message: "Expected expression".into(),
            }),
        }
    }

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
                    span: field_name_tok.span,
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
                let clean_text = text.replace('_', "");

                let (num_str, radix) =
                    if clean_text.starts_with("0x") || clean_text.starts_with("0X") {
                        (&clean_text[2..], 16)
                    } else if clean_text.starts_with("0b") || clean_text.starts_with("0B") {
                        (&clean_text[2..], 2)
                    } else if clean_text.starts_with("0o") || clean_text.starts_with("0O") {
                        (&clean_text[2..], 8)
                    } else {
                        (clean_text.as_str(), 10)
                    };

                let val = u64::from_str_radix(num_str, radix).map_err(|_| ParseError {
                    expected: "Integer".into(),
                    found: token.kind,
                    span: token.span,
                    message: format!("Invalid integer literal '{}'", text),
                })?;

                Ok(Literal::Integer(val))
            }
            TokenKind::Float => {
                let val = text.replace('_', "").parse::<f64>().unwrap();
                Ok(Literal::Float(val))
            }
            TokenKind::StringLit => {
                let content = &text[1..text.len() - 1];

                let unescaped = self.unescape_string(content);

                Ok(Literal::String(unescaped))
            }
            TokenKind::CharLit => {
                let content = &text[1..text.len() - 1];

                let unescaped_str = self.unescape_string(content);

                let c = unescaped_str.chars().next().unwrap_or('\0');

                Ok(Literal::Char(c))
            }
            _ => unreachable!(),
        }
    }
}

impl<'a> Parser<'a> {
    /// Statement -> Decl | Assign | Call | If | While | Break | Continue | Ret | Block | Switch
    pub fn parse_statement(&mut self) -> ParseResult<Statement> {
        if self.match_token(&[TokenKind::Set, TokenKind::Mut, TokenKind::Const]) {
            let modifier = self.previous_kind;
            return self.parse_decl_stmt(modifier);
        }

        if self.check(TokenKind::LBrace) {
            let block = self.parse_block()?;
            let span = block.span;

            return Ok(Statement {
                id: self.next_id(),
                kind: StatementKind::Block(block),
                span,
            });
        }

        if self.match_token(&[TokenKind::If]) {
            return self.parse_if_stmt();
        }
        if self.match_token(&[TokenKind::While]) {
            return self.parse_while_stmt();
        }
        if self.match_token(&[TokenKind::Switch]) {
            return self.parse_switch_stmt();
        }
        if self.match_token(&[TokenKind::Ret]) {
            return self.parse_return_stmt();
        }

        if self.match_token(&[TokenKind::Break]) {
            let start = self.previous_span().start;
            let end_tok = self.expect(TokenKind::Semi)?;
            return Ok(Statement {
                id: self.next_id(),
                kind: StatementKind::Break,
                span: Span::new(start, end_tok.span.end),
            });
        }

        if self.match_token(&[TokenKind::Continue]) {
            let start = self.previous_span().start;
            let end_tok = self.expect(TokenKind::Semi)?;
            return Ok(Statement {
                id: self.next_id(),
                kind: StatementKind::Continue,
                span: Span::new(start, end_tok.span.end),
            });
        }

        self.parse_assign_or_call_stmt()
    }

    /// Block -> "{" { Statement } "}"
    /// 返回 Vec<Statement> 供其他结构（如函数体）复用
    fn parse_block(&mut self) -> ParseResult<Block> {
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
            id: self.next_id(),
            stmts,
            span: Span::new(start, end),
        })
    }

    fn parse_decl_stmt(&mut self, modifier_kind: TokenKind) -> ParseResult<Statement> {
        let start = self.previous_span().start;

        let modifier = match modifier_kind {
            TokenKind::Set => Mutability::Immutable,
            TokenKind::Mut => Mutability::Mutable,
            TokenKind::Const => Mutability::Constant,
            _ => unreachable!(),
        };

        let name_tok = self.expect(TokenKind::Identifier)?;
        let name = Identifier {
            name: self.text(name_tok).to_string(),
            span: name_tok.span,
        };

        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type()?;

        let mut initializer = None;
        if self.match_token(&[TokenKind::Eq]) {
            initializer = Some(self.parse_expr()?);
        }

        let end_tok = self.expect(TokenKind::Semi)?;

        Ok(Statement {
            id: self.next_id(),
            kind: StatementKind::VariableDeclaration {
                modifier,
                name,
                type_annotation: ty,
                initializer,
            },
            span: Span::new(start, end_tok.span.end),
        })
    }

    fn parse_if_stmt(&mut self) -> ParseResult<Statement> {
        let start = self.previous_span().start;

        self.expect(TokenKind::LParen)?;
        let condition = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;

        let then_block = self.parse_block()?;

        let mut else_branch = None;
        if self.match_token(&[TokenKind::Else]) {
            if self.check(TokenKind::If) {
                self.advance();
                else_branch = Some(Box::new(self.parse_if_stmt()?));
            } else {
                let block = self.parse_block()?;
                let span = block.span;
                else_branch = Some(Box::new(Statement {
                    id: self.next_id(),
                    kind: StatementKind::Block(block),
                    span,
                }));
            }
        }

        let end = else_branch
            .as_ref()
            .map(|s| s.span.end)
            .unwrap_or(then_block.span.end);

        Ok(Statement {
            id: self.next_id(),
            kind: StatementKind::If {
                condition,
                then_block,
                else_branch,
            },
            span: Span::new(start, end),
        })
    }

    fn parse_while_stmt(&mut self) -> ParseResult<Statement> {
        let start = self.previous_span().start;

        self.expect(TokenKind::LParen)?;
        let condition = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;

        let mut init_statement = None;
        if self.match_token(&[TokenKind::Colon]) {
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

        let end = body.span.end;

        Ok(Statement {
            id: self.next_id(),
            kind: StatementKind::While {
                condition,
                init_statement,
                body,
            },
            span: Span::new(start, end),
        })
    }

    fn parse_return_stmt(&mut self) -> ParseResult<Statement> {
        let start = self.previous_span().start;

        let mut value = None;
        if !self.check(TokenKind::Semi) {
            value = Some(self.parse_expr()?);
        }

        let end_tok = self.expect(TokenKind::Semi)?;

        Ok(Statement {
            id: self.next_id(),
            kind: StatementKind::Return(value),
            span: Span::new(start, end_tok.span.end),
        })
    }

    fn parse_switch_stmt(&mut self) -> ParseResult<Statement> {
        let start = self.previous_span().start;

        self.expect(TokenKind::LParen)?;
        let target = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;

        self.expect(TokenKind::LBrace)?;

        let mut cases = Vec::new();
        let mut default_case = None;

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            if self.match_token(&[TokenKind::Default]) {
                if default_case.is_some() {
                    return Err(ParseError {
                        expected: "End of switch".into(),
                        found: TokenKind::Default,
                        span: self.previous_span(),
                        message: "Multiple default cases found".into(),
                    });
                }
                self.expect(TokenKind::Arrow)?;

                let stmt = self.parse_statement()?;

                let block = self.statement_to_block(stmt);
                default_case = Some(block);
            } else {
                let mut patterns = Vec::new();
                let first = self.parse_expr()?;
                patterns.push(first.clone());
                let pattern_start = first.span.start;

                while self.match_token(&[TokenKind::Pipe]) {
                    patterns.push(self.parse_expr()?);
                }

                self.expect(TokenKind::Arrow)?;
                let stmt = self.parse_statement()?;
                let block = self.statement_to_block(stmt);

                let span = Span::new(pattern_start, block.span.end);

                cases.push(SwitchCase {
                    patterns,
                    body: block,
                    span,
                });
            }
        }

        let end_tok = self.expect(TokenKind::RBrace)?;

        Ok(Statement {
            id: self.next_id(),
            kind: StatementKind::Switch {
                target,
                cases,
                default_case,
            },
            span: Span::new(start, end_tok.span.end),
        })
    }

    fn statement_to_block(&mut self, stmt: Statement) -> Block {
        match stmt.kind {
            StatementKind::Block(b) => b,

            _ => Block {
                id: self.next_id(),
                span: stmt.span,
                stmts: vec![stmt],
            },
        }
    }

    fn parse_assign_or_call_stmt(&mut self) -> ParseResult<Statement> {
        let start = self.peek().span.start;

        let expr = self.parse_postfix()?;

        if self.match_token(&[TokenKind::Eq]) {
            let rhs = self.parse_expr()?;
            let end_tok = self.expect(TokenKind::Semi)?;

            Ok(Statement {
                id: self.next_id(),
                kind: StatementKind::Assignment { lhs: expr, rhs },
                span: Span::new(start, end_tok.span.end),
            })
        } else {
            let end_tok = self.expect(TokenKind::Semi)?;
            Ok(Statement {
                id: self.next_id(),
                kind: StatementKind::ExpressionStatement(expr),
                span: Span::new(start, end_tok.span.end),
            })
        }
    }
}

impl<'a> Parser<'a> {
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

    /// TopLevelDecl 分发
    fn parse_top_level(&mut self) -> ParseResult<Item> {
        let is_pub = self.match_token(&[TokenKind::Pub]);
        let start_span = if is_pub {
            self.previous_span()
        } else {
            self.peek().span
        };

        if self.match_token(&[TokenKind::Extern]) {
            let start = if is_pub {
                start_span.start
            } else {
                self.previous_span().start
            };

            if self.check(TokenKind::Fn) {
                let func_def = self.parse_function_definition(is_pub, true, start, None)?;
                return Ok(Item {
                    id: self.next_id(),
                    span: func_def.span,
                    kind: ItemKind::FunctionDecl(func_def),
                });
            } else if self.match_token(&[TokenKind::Set, TokenKind::Mut, TokenKind::Const]) {
                let modifier = match self.previous_kind {
                    TokenKind::Set => Mutability::Immutable,
                    TokenKind::Mut => Mutability::Mutable,
                    TokenKind::Const => Mutability::Constant,
                    _ => unreachable!(),
                };

                let name = self.parse_identifier()?;
                self.expect(TokenKind::Colon)?;
                let ty = self.parse_type()?;
                let end = self.expect(TokenKind::Semi)?.span.end;

                return Ok(Item {
                    id: self.next_id(),
                    kind: ItemKind::GlobalVariable(GlobalDefinition {
                        name,
                        ty,
                        modifier,
                        initializer: None,
                        is_extern: true,
                        span: Span::new(start, end),
                        is_pub,
                    }),
                    span: Span::new(start, end),
                });
            }

            return Err(ParseError {
                expected: "fn or variable declaration".into(),
                found: self.peek().kind,
                span: self.peek().span,
                message: "Expected 'fn' or 'set/mut/const' after extern".into(),
            });
        }

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

            TokenKind::Set | TokenKind::Mut | TokenKind::Const => {
                self.parse_global_variable(is_pub, start_span.start)
            }

            TokenKind::Imp => {
                if is_pub {
                    return Err(ParseError {
                        expected: "imp without pub".into(),
                        found: TokenKind::Imp,
                        span: start_span,
                        message: "'imp' blocks cannot be public, only their methods can.".into(),
                    });
                }
                self.parse_imp_decl(start_span.start)
            }
            TokenKind::Cap => self.parse_cap_decl(is_pub, start_span.start),

            _ => Err(ParseError {
                expected: "Top level declaration".into(),
                found: token.kind,
                span: token.span,
                message: format!("Unexpected token at top level: {:?}", token.kind),
            }),
        }
    }

    fn parse_module_decl(&mut self, is_pub: bool, start: usize) -> ParseResult<Item> {
        self.expect(TokenKind::Mod)?;
        let name = self.parse_identifier()?;
        let end = self.expect(TokenKind::Semi)?.span.end;

        Ok(Item {
            id: self.next_id(),
            kind: ItemKind::ModuleDecl {
                name,
                is_pub,
                items: None,
            },
            span: Span::new(start, end),
        })
    }

    fn parse_import_decl(&mut self, is_pub: bool, start: usize) -> ParseResult<Item> {
        self.expect(TokenKind::Use)?;
        let path = self.parse_path()?;

        let mut alias = None;
        if self.match_token(&[TokenKind::As]) {
            alias = Some(self.parse_identifier()?);
        }

        let end = self.expect(TokenKind::Semi)?.span.end;

        Ok(Item {
            id: self.next_id(),
            kind: ItemKind::Import {
                path,
                alias,
                is_pub,
            },
            span: Span::new(start, end),
        })
    }

    fn parse_typedef_decl(&mut self, is_pub: bool, start: usize) -> ParseResult<Item> {
        self.expect(TokenKind::Typedef)?;
        let name = self.parse_identifier()?;
        self.expect(TokenKind::Eq)?;
        let target_type = self.parse_type()?;
        let end = self.expect(TokenKind::Semi)?.span.end;

        Ok(Item {
            id: self.next_id(),
            kind: ItemKind::Typedef {
                name,
                target_type,
                is_pub,
            },
            span: Span::new(start, end),
        })
    }

    fn parse_typealias_decl(&mut self, is_pub: bool, start: usize) -> ParseResult<Item> {
        self.expect(TokenKind::Typealias)?;
        let name = self.parse_identifier()?;
        self.expect(TokenKind::Eq)?;
        let target_type = self.parse_type()?;
        let end = self.expect(TokenKind::Semi)?.span.end;

        Ok(Item {
            id: self.next_id(),
            kind: ItemKind::TypeAlias {
                name,
                target_type,
                is_pub,
            },
            span: Span::new(start, end),
        })
    }

    fn parse_fn_decl(&mut self, is_pub: bool, start: usize) -> ParseResult<Item> {
        let func_def = self.parse_function_definition(is_pub, false, start, None)?;

        Ok(Item {
            id: self.next_id(),
            span: func_def.span,
            kind: ItemKind::FunctionDecl(func_def),
        })
    }

    fn parse_struct_decl(&mut self, is_pub: bool, start: usize) -> ParseResult<Item> {
        let def = self.parse_record_definition(TokenKind::Struct, start, is_pub)?;
        let end = def.span.end;

        Ok(Item {
            id: self.next_id(),
            kind: ItemKind::StructDecl(def),
            span: Span::new(start, end),
        })
    }

    fn parse_union_decl(&mut self, is_pub: bool, start: usize) -> ParseResult<Item> {
        let def = self.parse_record_definition(TokenKind::Union, start, is_pub)?;
        let end = def.span.end;

        Ok(Item {
            id: self.next_id(),
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
            let is_fn = self.check(TokenKind::Fn)
                || (self.check(TokenKind::Pub) && self.check_nth(1, TokenKind::Fn));

            if is_fn {
                let is_method_pub = self.match_token(&[TokenKind::Pub]);
                let fn_start = if is_method_pub {
                    self.previous_span().start
                } else {
                    self.peek().span.start
                };

                static_methods.push(self.parse_function_definition(
                    is_method_pub,
                    false,
                    fn_start,
                    None,
                )?);
            } else {
                items.push(parse_item(self)?);
            }
        }

        let end = self.expect(TokenKind::RBrace)?.span.end;
        Ok((items, static_methods, Span::new(start, end)))
    }

    fn parse_record_definition(
        &mut self,
        keyword: TokenKind,
        start: usize,
        is_pub: bool,
    ) -> ParseResult<StructDefinition> {
        self.expect(keyword)?;

        let alignment = self.parse_optional_alignment()?;
        let name = self.parse_identifier()?;

        let generics = self.parse_generic_params()?;

        let (fields, static_methods, body_span) = self.parse_mixed_body(|p| {
            let field_name = p.parse_identifier()?;
            p.expect(TokenKind::Colon)?;
            let ty = p.parse_type()?;
            let semi = p.expect(TokenKind::Semi)?;
            Ok(FieldDefinition {
                id: p.next_id(),
                name: field_name.clone(),
                ty,
                span: Span::new(field_name.span.start, semi.span.end),
            })
        })?;

        Ok(StructDefinition {
            name,
            generics,
            fields,
            static_methods,
            alignment,
            span: Span::new(start, body_span.end),
            is_pub,
        })
    }

    fn parse_optional_alignment(&mut self) -> ParseResult<Option<u32>> {
        if self.match_token(&[TokenKind::LParen]) {
            let int_tok = self.expect(TokenKind::Integer)?;
            let val_str = self.text(int_tok);

            let val = val_str.parse::<u32>().map_err(|_| ParseError {
                expected: "Integer (u32)".into(),
                found: TokenKind::Integer,
                span: int_tok.span,
                message: "Alignment value is too large".into(),
            })?;

            if val == 0 {
                return Err(ParseError {
                    expected: "Positive Integer".into(),
                    found: TokenKind::Integer,
                    span: int_tok.span,
                    message: "Alignment cannot be 0".into(),
                });
            }

            if (val & (val - 1)) != 0 {
                return Err(ParseError {
                    expected: "Power of 2".into(),
                    found: TokenKind::Integer,
                    span: int_tok.span,
                    message: format!("Alignment must be a power of 2, but found {}", val),
                });
            }

            self.expect(TokenKind::RParen)?;
            Ok(Some(val))
        } else {
            Ok(None)
        }
    }

    fn parse_enum_decl(&mut self, is_pub: bool, start: usize) -> ParseResult<Item> {
        self.expect(TokenKind::Enum)?;
        let name = self.parse_identifier()?;

        let generics = self.parse_generic_params()?;

        let mut underlying_type = None;
        if self.match_token(&[TokenKind::Colon]) {
            let ty = self.parse_type()?;
            if let TypeKind::Primitive(p) = ty.kind {
                underlying_type = Some(p);
            } else {
                return Err(ParseError {
                    expected: "Integer Type".into(),
                    found: TokenKind::Identifier,
                    span: ty.span,
                    message: "Enum backing type must be primitive integer".into(),
                });
            }
        }

        let (variants, static_methods, body_span) = self.parse_mixed_body(|p| {
            let v_name = p.parse_identifier()?;
            let mut value = None;
            if p.match_token(&[TokenKind::Eq]) {
                value = Some(p.parse_expr()?);
            }
            let semi = p.expect(TokenKind::Semi)?;

            Ok(EnumVariant {
                id: p.next_id(),
                name: v_name.clone(),
                value,
                span: Span::new(v_name.span.start, semi.span.end),
            })
        })?;

        Ok(Item {
            id: self.next_id(),
            kind: ItemKind::EnumDecl(EnumDefinition {
                name,
                generics,
                underlying_type,
                variants,
                static_methods,
                span: Span::new(start, body_span.end),
                is_pub,
            }),
            span: Span::new(start, body_span.end),
        })
    }

    fn parse_cap_decl(&mut self, is_pub: bool, start: usize) -> ParseResult<Item> {
        self.expect(TokenKind::Cap)?;
        let name = self.parse_identifier()?;

        let generics = self.parse_generic_params()?;

        self.expect(TokenKind::LBrace)?;

        let mut methods = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let is_method_pub = self.match_token(&[TokenKind::Pub]);
            let fn_start = if is_method_pub {
                self.previous_span().start
            } else {
                self.peek().span.start
            };

            methods.push(self.parse_function_definition(is_method_pub, false, fn_start, None)?);
        }

        let end = self.expect(TokenKind::RBrace)?.span.end;

        Ok(Item {
            id: self.next_id(),
            kind: ItemKind::CapDecl(CapDefinition {
                name,
                generics,
                methods,
                is_pub,
                span: Span::new(start, end),
            }),
            span: Span::new(start, end),
        })
    }

    fn parse_imp_decl(&mut self, start: usize) -> ParseResult<Item> {
        self.expect(TokenKind::Imp)?;

        let generics = self.parse_generic_params()?;

        self.expect(TokenKind::For)?;

        let target_type = self.parse_type()?;

        self.expect(TokenKind::LBrace)?;

        let mut methods = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let is_pub = self.match_token(&[TokenKind::Pub]);
            let fn_start = if is_pub {
                self.previous_span().start
            } else {
                self.peek().span.start
            };

            methods.push(self.parse_function_definition(
                is_pub,
                false,
                fn_start,
                Some(&target_type),
            )?);
        }

        let end = self.expect(TokenKind::RBrace)?.span.end;

        Ok(Item {
            id: self.next_id(),
            kind: ItemKind::Implementation {
                generics,
                implements: None,
                target_type,
                methods,
            },
            span: Span::new(start, end),
        })
    }

    fn parse_global_variable(&mut self, is_pub: bool, start: usize) -> ParseResult<Item> {
        let modifier = if self.match_token(&[TokenKind::Mut]) {
            Mutability::Mutable
        } else if self.match_token(&[TokenKind::Set]) {
            Mutability::Immutable
        } else if self.match_token(&[TokenKind::Const]) {
            Mutability::Constant
        } else {
            unreachable!()
        };

        let name = self.parse_identifier()?;

        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type()?;

        let mut initializer = None;
        if self.match_token(&[TokenKind::Eq]) {
            initializer = Some(self.parse_expr()?);
        }

        let end_tok = self.expect(TokenKind::Semi)?;

        Ok(Item {
            id: self.next_id(),
            kind: ItemKind::GlobalVariable(GlobalDefinition {
                name,
                ty,
                modifier,
                initializer,
                span: Span::new(start, end_tok.span.end),
                is_extern: false,
                is_pub,
            }),
            span: Span::new(start, end_tok.span.end),
        })
    }

    /// 解析函数定义
    /// ctx_type: 如果是 imp 块的方法，这里包含 impl 的目标类型；否则为 None
    fn parse_function_definition(
        &mut self,
        is_pub: bool,
        is_extern: bool,
        start: usize,
        ctx_type: Option<&Type>,
    ) -> ParseResult<FunctionDefinition> {
        self.expect(TokenKind::Fn)?;
        let name = self.parse_identifier()?;

        let generics = self.parse_generic_params()?;

        self.expect(TokenKind::LParen)?;

        let (params, is_variadic) = self.parse_param_list(ctx_type)?;
        self.expect(TokenKind::RParen)?;

        let mut return_type = None;
        if self.match_token(&[TokenKind::Arrow]) {
            return_type = Some(self.parse_type()?);
        }

        let (body, end_pos) = if self.check(TokenKind::LBrace) {
            let block = self.parse_block()?;
            let end = block.span.end;
            (Some(block), end)
        } else if self.match_token(&[TokenKind::Semi]) {
            (None, self.previous_span().end)
        } else {
            return Err(ParseError {
                expected: "{ or ;".into(),
                found: self.peek().kind,
                span: self.peek().span,
                message: "Expected function body or semicolon".into(),
            });
        };

        Ok(FunctionDefinition {
            id: self.next_id(),
            name,
            generics,
            params,
            return_type,
            body,
            is_variadic,
            is_pub,
            is_extern,
            span: Span::new(start, end_pos),
        })
    }

    /// 解析参数列表 (Param | SelfParam | VarArgs)
    /// ctx_type: Some(Type) 表示允许 self 且类型为 Type；None 表示不允许 self。
    /// 返回值: (参数列表, 是否是变长参数)
    fn parse_param_list(&mut self, ctx_type: Option<&Type>) -> ParseResult<(Vec<Parameter>, bool)> {
        let mut params = Vec::new();
        let mut is_variadic = false;

        if self.check(TokenKind::RParen) {
            return Ok((params, false));
        }

        loop {
            if self.match_token(&[TokenKind::DotDotDot]) {
                is_variadic = true;

                break;
            }

            let is_self_start = self.check(TokenKind::SelfVal)
                || ((self.check(TokenKind::Mut) || self.check(TokenKind::Const))
                    && self.check_nth(1, TokenKind::SelfVal));

            if is_self_start {
                let target_type = if let Some(ty) = ctx_type {
                    ty
                } else {
                    return Err(ParseError {
                        expected: "Parameter".into(),
                        found: TokenKind::SelfVal,
                        span: self.peek().span,
                        message: "self only allowed in imp".into(),
                    });
                };
                if !params.is_empty() {
                    return Err(ParseError {
                        expected: "Parameter".into(),
                        found: TokenKind::SelfVal,
                        span: self.peek().span,
                        message: "self must be first".into(),
                    });
                }

                let mut modifier = Mutability::Immutable;
                if self.match_token(&[TokenKind::Mut]) {
                    modifier = Mutability::Mutable;
                } else if self.match_token(&[TokenKind::Const]) {
                    modifier = Mutability::Constant;
                }

                let self_tok = self.expect(TokenKind::SelfVal)?;
                let real_type = target_type.clone();

                params.push(Parameter {
                    id: self.next_id(),
                    name: Identifier {
                        name: "self".into(),
                        span: self_tok.span,
                    },
                    ty: real_type,
                    is_mutable: modifier == Mutability::Mutable,
                    is_self: true,
                });
            } else {
                let mut modifier = Mutability::Immutable;
                if self.match_token(&[TokenKind::Mut]) {
                    modifier = Mutability::Mutable;
                } else if self.match_token(&[TokenKind::Const]) {
                    modifier = Mutability::Constant;
                }

                let p_name = self.parse_identifier()?;
                self.expect(TokenKind::Colon)?;
                let p_type = self.parse_type()?;

                params.push(Parameter {
                    id: self.next_id(),
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

        Ok((params, is_variadic))
    }

    fn parse_identifier(&mut self) -> ParseResult<Identifier> {
        let tok = self.expect(TokenKind::Identifier)?;
        Ok(Identifier {
            name: self.text(tok).to_string(),
            span: tok.span,
        })
    }
}
