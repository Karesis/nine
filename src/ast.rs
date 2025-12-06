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

/// ======================================================
/// 基础定义
/// ======================================================

/// 带有位置信息的标识符
#[derive(Debug, Clone)]
pub struct Identifier {
    pub name: String,
    pub span: Span,
}

/// 泛型参数 <T: Cap + Cap>
#[derive(Debug, Clone)]
pub struct GenericParam {
    pub id: NodeId,
    pub name: Identifier,
    pub constraints: Vec<Path>,
    pub span: Span,
}

/// 能力定义 (Interface/Trait)
/// cap Printable { fn to_string(self) -> ^u8; }
#[derive(Debug, Clone)]
pub struct CapDefinition {
    pub name: Identifier,
    pub generics: Vec<GenericParam>,
    pub methods: Vec<FunctionDefinition>,
    pub is_pub: bool,
    pub span: Span,
}

/// 路径片段 (e.g. Vector#<i32>)
#[derive(Debug, Clone)]
pub struct PathSegment {
    pub name: String,
    pub generic_args: Option<Vec<Type>>,
    pub span: Span,
}

/// 路径
#[derive(Debug, Clone)]
pub struct Path {
    pub id: NodeId,
    pub segments: Vec<PathSegment>,
    pub span: Span,
}

/// 变量/参数的可变性修饰符
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mutability {
    Constant,
    Mutable,
    Immutable,
}

/// 二元运算符
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    LogicalOr,
    LogicalAnd,

    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,

    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,

    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
}

/// 一元运算符
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Negate,
    Not,

    Dereference,
    AddressOf,
}

/// ======================================================
/// 类型系统
/// ======================================================

#[derive(Debug, Clone)]
pub struct Type {
    pub id: NodeId,
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TypeKind {
    /// 基础类型 (i32, f64, bool...)
    Primitive(PrimitiveType),

    /// 命名类型 (MyStruct, std::io::File)
    Named(Path),

    /// 指针类型
    /// is_mutable = true  => *T (mut ptr)
    /// is_mutable = false => ^T (const ptr)
    Pointer {
        inner: Box<Type>,
        mutability: Mutability,
    },

    /// 数组类型 T[N] - EBNF: AtomType "[" INT "]"
    Array { inner: Box<Type>, size: u64 },

    /// 函数指针类型 fn(i32, i32) -> bool
    Function {
        params: Vec<Type>,
        ret_type: Option<Box<Type>>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    ISize,
    USize,
    F32,
    F64,
    Bool,

    Unit,
}

/// ======================================================
/// 表达式
/// ======================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

#[derive(Debug, Clone)]
pub struct Expression {
    pub id: NodeId,
    pub kind: ExpressionKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ExpressionKind {
    /// 字面量 (1, "abc", true)
    Literal(Literal),

    /// 路径作为值 (Enum Variant, Static Var)
    Path(Path),

    /// 二元运算 (a + b)
    Binary {
        lhs: Box<Expression>,
        op: BinaryOperator,
        rhs: Box<Expression>,
    },

    /// 一元运算 (-a, !b, ptr^, val&)
    /// 注意：我们将后缀的 ^ 和 & 也归一化到了这里
    Unary {
        op: UnaryOperator,
        operand: Box<Expression>,
    },

    /// 函数调用 func(arg1, arg2)
    Call {
        callee: Box<Expression>,
        arguments: Vec<Expression>,
    },

    /// 方法调用 obj.method(arg)
    MethodCall {
        receiver: Box<Expression>,
        method_name: Identifier,
        arguments: Vec<Expression>,
    },

    /// 字段访问 obj.field
    FieldAccess {
        receiver: Box<Expression>,
        field_name: Identifier,
    },

    /// 索引访问 arr[i]
    Index {
        target: Box<Expression>,
        index: Box<Expression>,
    },

    /// 类型转换 val as T
    Cast {
        expr: Box<Expression>,
        target_type: Type,
    },

    /// 结构体初始化 MyStruct { a: 1, b: 2 }
    /// EBNF: StructLiteral (需要补充内部细节)
    StructLiteral {
        type_name: Path,
        fields: Vec<StructFieldInit>,
    },

    /// 静态/命名空间访问: expr::member
    /// 对应 EBNF: Postfix -> "::" Identifier
    StaticAccess {
        target: Box<Expression>,
        member: Identifier,
    },

    SizeOf(Type),
    AlignOf(Type),
}

#[derive(Debug, Clone)]
pub struct StructFieldInit {
    pub field_name: Identifier,
    pub value: Expression,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Literal {
    Integer(u64),
    Float(f64),
    String(String),
    Char(char),
    Boolean(bool),
}

/// ======================================================
/// 语句
/// ======================================================

#[derive(Debug, Clone)]
pub struct Statement {
    pub id: NodeId,
    pub kind: StatementKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum StatementKind {
    /// 变量声明
    /// EBNF: DeclStmt -> VarModifier Identifier ":" Type [ "=" Expression ] ";"
    VariableDeclaration {
        modifier: Mutability,
        name: Identifier,
        type_annotation: Type,
        initializer: Option<Expression>,
    },

    /// 赋值
    /// EBNF: AssignStmt -> Postfix "=" Expression ";"
    Assignment {
        lhs: Expression,
        rhs: Expression,
    },

    /// 表达式语句 (函数调用等)
    ExpressionStatement(Expression),

    /// 块语句 { stmt... }
    Block(Block),

    /// If 语句
    If {
        condition: Expression,
        then_block: Block,
        else_branch: Option<Box<Statement>>,
    },

    /// While 语句
    /// EBNF: "while" "(" Expression ")" [ ":" DeclStmt ] Block
    While {
        condition: Expression,
        init_statement: Option<Box<Statement>>,
        body: Block,
    },

    /// Switch 语句
    Switch {
        target: Expression,
        cases: Vec<SwitchCase>,
        default_case: Option<Block>,
    },

    /// 流程控制
    Return(Option<Expression>),
    Break,
    Continue,
}

#[derive(Debug, Clone)]
pub struct SwitchCase {
    /// CasePatterns -> Expression { "|" Expression }
    pub patterns: Vec<Expression>,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub id: NodeId,
    pub stmts: Vec<Statement>,
    pub span: Span,
}

/// ======================================================
/// 顶层定义
/// ======================================================

#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub struct Item {
    pub id: NodeId,
    pub kind: ItemKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ItemKind {
    /// 模块声明: mod xxx;
    /// 解析初期：items 为 None
    /// Driver解析后：items 为 Some(Vec<Item>)，即子模块的内容被填入这里
    ModuleDecl {
        name: Identifier,
        is_pub: bool,
        items: Option<Vec<Item>>,
    },

    /// 导入: use path as alias;
    Import {
        path: Path,
        alias: Option<Identifier>,
        is_pub: bool,
    },

    /// 结构体定义
    StructDecl(StructDefinition),

    /// 联合体
    UnionDecl(StructDefinition),

    /// 枚举定义
    EnumDecl(EnumDefinition),

    /// 强类型定义 (typedef A = B)
    /// A 是一个全新的类型
    Typedef {
        name: Identifier,
        target_type: Type,
        is_pub: bool,
    },

    /// 类型别名 (typealias A = B)
    /// A 只是 B 的另一个名字
    TypeAlias {
        name: Identifier,
        target_type: Type,
        is_pub: bool,
    },

    /// 函数定义
    FunctionDecl(FunctionDefinition),

    CapDecl(CapDefinition),

    /// 实现块 imp for Type { ... }
    Implementation {
        generics: Vec<GenericParam>,
        implements: Option<Path>,
        target_type: Type,
        methods: Vec<FunctionDefinition>,
    },

    GlobalVariable(GlobalDefinition),
}

#[derive(Debug, Clone)]
pub struct GlobalDefinition {
    pub name: Identifier,
    pub ty: Type,
    pub modifier: Mutability,
    pub initializer: Option<Expression>,
    pub span: Span,
    pub is_extern: bool,
    pub is_pub: bool,
}

#[derive(Debug, Clone)]
pub struct StructDefinition {
    pub name: Identifier,
    pub generics: Vec<GenericParam>,
    pub fields: Vec<FieldDefinition>,
    /// 静态函数 (属于 struct 命名空间)
    /// 解析时：直接读取 Struct 内部的 fn
    /// 语义检查时：确保这里面的 fn 没有 self 参数
    pub static_methods: Vec<FunctionDefinition>,

    /// 内存对齐 (struct(N) Identifier ...)
    /// None 表示使用默认对齐，Some(N) 表示强制对齐到 N 字节
    pub alignment: Option<u32>,

    pub span: Span,
    pub is_pub: bool,
}

/// 枚举定义
#[derive(Debug, Clone)]
pub struct EnumDefinition {
    pub name: Identifier,
    pub generics: Vec<GenericParam>,
    /// 基础整数类型 (如 enum Color : u8)
    pub underlying_type: Option<PrimitiveType>,

    pub variants: Vec<EnumVariant>,

    /// 枚举也可以拥有静态方法 (如 Color::all())
    pub static_methods: Vec<FunctionDefinition>,

    pub span: Span,
    pub is_pub: bool,
}

#[derive(Debug, Clone)]
pub struct FieldDefinition {
    pub id: NodeId,
    pub name: Identifier,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub id: NodeId,
    pub name: Identifier,
    /// 显式赋值 (= INT)
    pub value: Option<Expression>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FunctionDefinition {
    pub id: NodeId,
    pub name: Identifier,
    pub generics: Vec<GenericParam>,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,

    pub body: Option<Block>,
    pub is_variadic: bool,
    pub is_pub: bool,
    pub is_extern: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub id: NodeId,
    pub name: Identifier,
    pub ty: Type,
    pub is_mutable: bool,
    pub is_self: bool,
}
