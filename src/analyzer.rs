use std::collections::HashMap;
use crate::ast::*;
use crate::source::Span;

/// 定义 ID：指向 AST 中的节点
pub type DefId = NodeId;

/// ======================================================
/// 1. 语义类型键 (TypeKey)
/// ======================================================
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeKey {
    Primitive(PrimitiveType),
    Named(DefId), 
    Pointer(Box<TypeKey>, Mutability),
    Array(Box<TypeKey>, u64),
    Function {
        params: Vec<TypeKey>,
        ret: Option<Box<TypeKey>>,
    },
    /// 未定类型的整数字面量
    IntegerLiteral(u64),
    /// 浮点字面量: 存 f64 的 to_bits()
    FloatLiteral(u64),
    Error, 
}

/// ======================================================
/// 2. 上下文与核心表结构
/// ======================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Block,
    Function,
    Module, 
    Loop,
}

#[derive(Debug, Clone)] 
pub struct Scope {
    pub kind: ScopeKind,
    pub symbols: HashMap<String, DefId>,
}

impl Scope {
    pub fn new(kind: ScopeKind) -> Self {
        Self { kind, symbols: HashMap::new() }
    }
}

#[derive(Debug, Clone)]
pub struct MethodInfo {
    pub name: String,
    pub def_id: DefId, 
    pub is_pub: bool,
    pub span: Span,
}

#[derive(Debug)]
pub struct AnalysisContext {
    pub namespace_scopes: HashMap<DefId, Scope>,
    pub method_registry: HashMap<TypeKey, HashMap<String, MethodInfo>>,
    pub path_resolutions: HashMap<NodeId, DefId>,
    pub types: HashMap<NodeId, TypeKey>,
    pub struct_fields: HashMap<DefId, HashMap<String, TypeKey>>,
    /// 记录变量/参数的可变性 (set/mut/const)
    /// Key: DefId (VariableDeclaration 或 Parameter 的 ID)
    /// Value: Mutability
    pub mutabilities: HashMap<DefId, Mutability>,
    pub errors: Vec<AnalyzeError>,
}

impl AnalysisContext {
    pub fn new() -> Self {
        Self {
            namespace_scopes: HashMap::new(),
            method_registry: HashMap::new(),
            path_resolutions: HashMap::new(),
            types: HashMap::new(),
            struct_fields: HashMap::new(),
            mutabilities: HashMap::new(),
            errors: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct AnalyzeError {
    pub message: String,
    pub span: Span,
}

/// ======================================================
/// 3. Analyzer 主逻辑
/// ======================================================

pub struct Analyzer {
    pub ctx: AnalysisContext,
    pub scopes: Vec<Scope>,
    pub current_return_type: Option<TypeKey>,
}

impl Analyzer {
    pub fn new() -> Self {
        Self {
            ctx: AnalysisContext::new(),
            scopes: Vec::new(),
            current_return_type: None,
        }
    }

    fn error(&mut self, msg: impl Into<String>, span: Span) {
        self.ctx.errors.push(AnalyzeError {
            message: msg.into(),
            span,
        });
    }

    /// ==================================================
    /// 入口
    /// ==================================================
    pub fn analyze_program(&mut self, program: &Program) {
        self.scopes.push(Scope::new(ScopeKind::Module));

        // Pass 1: 声明扫描
        self.scan_declarations(&program.items);

        // Pass 2: 实现扫描
        self.scan_implementations(&program.items);

        // Pass 3: 签名解析
        self.resolve_signatures(&program.items);
        
        // Pass 4: 函数体检查
        self.check_bodies(&program.items);
    }

    /// ==================================================
    /// Phase 1: 声明扫描 (Declaration Scan)
    /// ==================================================
    fn scan_declarations(&mut self, items: &[Item]) {
        for item in items {
            match &item.kind {
                // --- 模块 ---
                ItemKind::ModuleDecl { name, items: sub_items, .. } => {
                    self.define_symbol(name.name.clone(), item.id, name.span);

                    if let Some(subs) = sub_items {
                        self.enter_scope(ScopeKind::Module);
                        self.scan_declarations(subs); 
                        let module_scope = self.exit_scope();
                        self.ctx.namespace_scopes.insert(item.id, module_scope);
                    }
                }
                
                // --- 结构体 ---
                ItemKind::StructDecl(def) => {
                    self.define_symbol(def.name.name.clone(), item.id, def.name.span);
                    self.register_static_methods(item.id, &def.static_methods);
                }

                // --- 【补全】联合体 ---
                ItemKind::UnionDecl(def) => {
                    self.define_symbol(def.name.name.clone(), item.id, def.name.span);
                    self.register_static_methods(item.id, &def.static_methods);
                }

                // --- 枚举 ---
                ItemKind::EnumDecl(def) => {
                    self.define_symbol(def.name.name.clone(), item.id, def.name.span);
                    
                    let mut enum_scope = Scope::new(ScopeKind::Module);
                    for variant in &def.variants {
                        enum_scope.symbols.insert(variant.name.name.clone(), variant.id);
                    }
                    // 静态方法处理
                    for method in &def.static_methods {
                        if enum_scope.symbols.contains_key(&method.name.name) {
                            self.error(format!("Duplicate name '{}' in enum", method.name.name), method.span);
                        } else {
                            enum_scope.symbols.insert(method.name.name.clone(), method.id);
                        }
                    }
                    self.ctx.namespace_scopes.insert(item.id, enum_scope);
                }

                // --- 函数 / 类型别名 ---
                ItemKind::FunctionDecl(def) => {
                    self.define_symbol(def.name.name.clone(), def.id, def.name.span);
                }
                ItemKind::Typedef { name, .. } | ItemKind::TypeAlias { name, .. } => {
                    self.define_symbol(name.name.clone(), item.id, name.span);
                }

                _ => {} 
            }
        }
    }

    // 辅助：注册静态方法 (减少 Struct/Union 代码重复)
    fn register_static_methods(&mut self, item_id: DefId, methods: &[FunctionDefinition]) {
        let mut static_scope = Scope::new(ScopeKind::Module);
        for method in methods {
            if static_scope.symbols.contains_key(&method.name.name) {
                    self.error(format!("Duplicate static method '{}'", method.name.name), method.span);
            } else {
                    static_scope.symbols.insert(method.name.name.clone(), method.id);
            }
        }
        self.ctx.namespace_scopes.insert(item_id, static_scope);
    }

    /// ==================================================
    /// Phase 2: 实现扫描 (Implementation Scan)
    /// ==================================================
    fn scan_implementations(&mut self, items: &[Item]) {
        for item in items {
             match &item.kind {
                ItemKind::ModuleDecl { items: sub_items, .. } => {
                    if let Some(subs) = sub_items {
                        if let Some(scope) = self.ctx.namespace_scopes.get(&item.id) {
                            self.scopes.push(scope.clone());
                            self.scan_implementations(subs);
                            self.scopes.pop();
                        }
                    }
                }

                ItemKind::Implementation { target_type, methods } => {
                    // 1. 计算 Key
                    let key = self.resolve_ast_type(target_type);
                    if let TypeKey::Error = key { continue; }

                    // 2. 【Clone】取出当前的注册表（复印件）
                    // 此时 self 的借用立即结束
                    let mut local_registry = self.ctx.method_registry
                        .get(&key)
                        .cloned()
                        .unwrap_or_default();
                    
                    // 3. 【Modify】修改复印件
                    // 因为 local_registry 是本地变量，不借用 self，
                    // 所以你在循环里可以尽情调用 self.error()！
                    for method in methods {
                        if local_registry.contains_key(&method.name.name) {
                            // 现在可以安全报错了！完美解决！
                            self.error(
                                format!("Duplicate method '{}'", method.name.name), 
                                method.name.span
                            );
                        } else {
                            local_registry.insert(method.name.name.clone(), MethodInfo {
                                name: method.name.name.clone(),
                                def_id: method.id, 
                                is_pub: method.is_pub,
                                span: method.span,
                            });
                        }
                    }

                    // 4. 【Write Back】把修改好的复印件塞回去
                    self.ctx.method_registry.insert(key, local_registry);
                }
                _ => {} 
             }
        }
    }

    /// ==================================================
    /// Phase 3: 签名解析 (Signature Resolution)
    /// ==================================================
   fn resolve_signatures(&mut self, items: &[Item]) {
        for item in items {
            match &item.kind {
                ItemKind::ModuleDecl { items: sub_items, .. } => {
                     if let Some(subs) = sub_items {
                        if let Some(scope) = self.ctx.namespace_scopes.get(&item.id) {
                            self.scopes.push(scope.clone());
                            self.resolve_signatures(subs);
                            self.scopes.pop();
                        }
                    }
                }
                
                ItemKind::StructDecl(def) | ItemKind::UnionDecl(def) => {
                    let mut fields = HashMap::new();
                    for field in &def.fields {
                        let ty = self.resolve_ast_type(&field.ty);
                        fields.insert(field.name.name.clone(), ty);
                    }
                    self.ctx.struct_fields.insert(item.id, fields);
                    
                    for method in &def.static_methods {
                        self.resolve_function_signature(method);
                    }
                }

                ItemKind::EnumDecl(def) => {
                    let enum_type = TypeKey::Named(item.id); 
                    for variant in &def.variants {
                        self.record_type(variant.id, enum_type.clone());
                    }
                    for method in &def.static_methods {
                        self.resolve_function_signature(method);
                    }
                }

                ItemKind::FunctionDecl(def) => {
                    self.resolve_function_signature(def);
                }

                ItemKind::Implementation { methods, .. } => {
                    for method in methods {
                        self.resolve_function_signature(method);
                    }
                }
                
                _ => {}
            }
        }
    }

    fn resolve_function_signature(&mut self, func: &FunctionDefinition) {
        let param_keys = func.params.iter()
            .map(|p| self.resolve_ast_type(&p.ty))
            .collect();
        
        let ret_key = if let Some(ret) = &func.return_type {
            Some(Box::new(self.resolve_ast_type(ret)))
        } else {
            None 
        };

        let ty = TypeKey::Function { params: param_keys, ret: ret_key };
        self.record_type(func.id, ty);
    }

    /// ==================================================
    /// Phase 4: 函数体检查 (Check Bodies)
    /// ==================================================

    fn record_type(&mut self, id: NodeId, ty: TypeKey) {
        self.ctx.types.insert(id, ty);
    }

    fn get_type_of_def(&self, def_id: DefId) -> Option<TypeKey> {
        self.ctx.types.get(&def_id).cloned()
    }

    fn check_type_match(&mut self, expected: &TypeKey, got: &TypeKey, span: Span) {
        if expected == got { return; }

        match (expected, got) {
            (TypeKey::Primitive(p), TypeKey::IntegerLiteral(val)) => {
                if self.is_integer_type(p) {
                    if self.check_int_range(*p, *val) { return; } 
                    else { self.error(format!("Literal {} out of range", val), span); return; }
                }
            }
            (TypeKey::Primitive(p), TypeKey::FloatLiteral(bits)) => {
                let val = f64::from_bits(*bits);
                match p {
                    PrimitiveType::F64 => return,
                    PrimitiveType::F32 => {
                        let val_f32 = val as f32;
                        if val.is_finite() && val_f32.is_infinite() {
                            self.error("Float literal overflows f32", span);
                        }
                        return;
                    }
                    _ => {}
                }
            }
            (TypeKey::IntegerLiteral(_), TypeKey::IntegerLiteral(_)) => return,
            (TypeKey::FloatLiteral(_), TypeKey::FloatLiteral(_)) => return,
            (TypeKey::Error, _) | (_, TypeKey::Error) => return,
            _ => {}
        }

        self.error(format!("Type mismatch: expected {:?}, found {:?}", expected, got), span);
    }

    fn is_integer_type(&self, p: &PrimitiveType) -> bool {
        use PrimitiveType::*;
        matches!(p, I8 | U8 | I16 | U16 | I32 | U32 | I64 | U64 | ISize | USize)
    }

    fn check_int_range(&self, p: PrimitiveType, val: u64) -> bool {
        use PrimitiveType::*;
        match p {
            U8 => val <= u8::MAX as u64,
            I8 => val <= i8::MAX as u64, // 简化版，未处理负数
            U16 => val <= u16::MAX as u64,
            I16 => val <= i16::MAX as u64,
            U32 => val <= u32::MAX as u64,
            I32 => val <= i32::MAX as u64,
            // 64位及size默认放行
            _ => true,
        }
    }

    fn check_bodies(&mut self, items: &[Item]) {
        for item in items {
            match &item.kind {
                ItemKind::ModuleDecl { items: sub_items, .. } => {
                    if let Some(subs) = sub_items {
                        if let Some(scope) = self.ctx.namespace_scopes.get(&item.id) {
                            self.scopes.push(scope.clone());
                            self.check_bodies(subs);
                            self.scopes.pop();
                        }
                    }
                }
                ItemKind::FunctionDecl(func) => self.check_function(func),
                ItemKind::Implementation { methods, .. } => {
                    for method in methods { self.check_function(method); }
                }
                ItemKind::StructDecl(def) | ItemKind::UnionDecl(def) => {
                    for method in &def.static_methods { self.check_function(method); }
                }
                ItemKind::EnumDecl(def) => {
                    for method in &def.static_methods { self.check_function(method); }
                }
                _ => {}
            }
        }
    }

    fn check_function(&mut self, func: &FunctionDefinition) {
        self.enter_scope(ScopeKind::Function);

        for param in &func.params {
            let param_type = self.resolve_ast_type(&param.ty);
            self.record_type(param.id, param_type);
            self.define_symbol(param.name.name.clone(), param.id, param.name.span);
            let mutability = if param.is_mutable { Mutability::Mutable } else { Mutability::Immutable };
            self.ctx.mutabilities.insert(param.id, mutability);
        }

        let expected_ret = if let Some(ret) = &func.return_type {
            self.resolve_ast_type(ret)
        } else {
            TypeKey::Primitive(PrimitiveType::Unit) 
        };

        let prev_ret_type = self.current_return_type.clone();
        self.current_return_type = Some(expected_ret);

        self.check_block(&func.body);

        self.current_return_type = prev_ret_type;
        self.exit_scope();
    }

    fn check_block(&mut self, block: &Block) {
        self.enter_scope(ScopeKind::Block);
        for stmt in &block.stmts {
            self.check_stmt(stmt);
        }
        self.exit_scope();
    }

   fn check_stmt(&mut self, stmt: &Statement) {
        match &stmt.kind {
           StatementKind::VariableDeclaration { modifier, name, type_annotation, initializer, .. } => {
                let declared_type = self.resolve_ast_type(type_annotation);
                
                // 1. 检查初始化逻辑
                if let Some(init_expr) = initializer {
                    // 有初始化：检查类型是否匹配
                    let init_type = self.check_expr(init_expr);
                    self.check_type_match(&declared_type, &init_type, init_expr.span);
                } else {
                    // 没有初始化：检查修饰符是否允许
                    match modifier {
                        Mutability::Constant => {
                            self.error("Constants (const) must be initialized immediately.", stmt.span);
                        }
                        Mutability::Immutable => {
                            // 'set' 变量一旦定义就不能改，所以必须定义时就给值
                            self.error("Immutable variables (set) must be initialized immediately.", stmt.span);
                        }
                        Mutability::Mutable => {
                            // 'mut' 允许不初始化 (保留语义上的“垃圾值”状态)
                            // 这里不做任何操作，允许 pass
                        }
                    }
                }

                // 2. 注册符号和类型 (保持不变)
                self.define_symbol(name.name.clone(), stmt.id, name.span);
                self.record_type(stmt.id, declared_type);
                
                // 3. 记录可变性 (保持不变)
                self.ctx.mutabilities.insert(stmt.id, *modifier);
            }

           StatementKind::Assignment { lhs, rhs } => {
                let lhs_ty = self.check_expr(lhs);
                let rhs_ty = self.check_expr(rhs);
                
                // 1. 类型匹配
                self.check_type_match(&lhs_ty, &rhs_ty, stmt.span);
                
                // 2. 【新增】L-Value 可变性检查
                self.check_lvalue_mutability(lhs);
            }
            
            StatementKind::Block(block) => self.check_block(block),

            StatementKind::ExpressionStatement(expr) => {
                match expr.kind {
                    ExpressionKind::Call { .. } | ExpressionKind::MethodCall { .. } => {
                        self.check_expr(expr); 
                    }
                    _ => {
                        self.error("Only function calls can be used as statements.", expr.span);
                    }
                }
            }

            StatementKind::Return(opt_expr) => {
                let expected_ret_opt = self.current_return_type.clone();
                if let Some(expected) = expected_ret_opt {
                    match opt_expr {
                        Some(expr) => {
                            if let TypeKey::Primitive(PrimitiveType::Unit) = expected {
                                self.error("Procedure (void function) cannot return a value.", expr.span);
                                self.check_expr(expr); 
                            } else {
                                let actual = self.check_expr(expr);
                                self.check_type_match(&expected, &actual, expr.span);
                            }
                        }
                        None => {
                            if let TypeKey::Primitive(PrimitiveType::Unit) = expected { /* OK */ } 
                            else { self.error("Function must return a value.", stmt.span); }
                        }
                    }
                } else {
                    self.error("Return statement outside of function context.", stmt.span);
                }
            }

            // --- 【补全】流程控制检查 ---
            StatementKind::If { condition, then_block, else_branch } => {
                let cond_ty = self.check_expr(condition);
                self.check_type_match(&TypeKey::Primitive(PrimitiveType::Bool), &cond_ty, condition.span);
                self.check_block(then_block);
                if let Some(else_stmt) = else_branch {
                    self.check_stmt(else_stmt);
                }
            }

            StatementKind::While { condition, init_statement, body } => {
                self.enter_scope(ScopeKind::Loop);
                if let Some(init) = init_statement { self.check_stmt(init); }
                
                let cond_ty = self.check_expr(condition);
                self.check_type_match(&TypeKey::Primitive(PrimitiveType::Bool), &cond_ty, condition.span);
                
                self.check_block(body);
                self.exit_scope();
            }

            StatementKind::Switch { target, cases, default_case } => {
                let target_ty = self.check_expr(target);
                for case in cases {
                    for pattern in &case.patterns {
                        let pattern_ty = self.check_expr(pattern);
                        self.check_type_match(&target_ty, &pattern_ty, pattern.span);
                    }
                    self.check_block(&case.body);
                }
                if let Some(default_block) = default_case {
                    self.check_block(default_block);
                }
            }

            StatementKind::Break | StatementKind::Continue => {
                let in_loop = self.scopes.iter().any(|s| s.kind == ScopeKind::Loop);
                if !in_loop {
                    self.error("break/continue used outside of loop", stmt.span);
                }
            }
        }
    }

    fn check_expr(&mut self, expr: &Expression) -> TypeKey {
        let ty = match &expr.kind {
            ExpressionKind::Literal(lit) => match lit {
                Literal::Integer(val) => TypeKey::IntegerLiteral(*val), 
                Literal::Float(val) => TypeKey::FloatLiteral(val.to_bits()),
                Literal::Boolean(_) => TypeKey::Primitive(PrimitiveType::Bool),
                Literal::String(_) => TypeKey::Pointer(
                    Box::new(TypeKey::Primitive(PrimitiveType::U8)), 
                    Mutability::Constant
                ),
                Literal::Char(_) => TypeKey::Primitive(PrimitiveType::I32),
            },

            ExpressionKind::Path(path) => {
                if let Some(def_id) = self.resolve_path(path) {
                    if let Some(def_type) = self.get_type_of_def(def_id) {
                        def_type
                    } else {
                        self.error("Symbol has no type", path.span);
                        TypeKey::Error
                    }
                } else {
                    self.error(format!("Undefined symbol: {:?}", path.segments.first().unwrap().name), path.span);
                    TypeKey::Error
                }
            },

            ExpressionKind::Binary { lhs, op, rhs } => {
                let lhs_ty = self.check_expr(lhs);
                let rhs_ty = self.check_expr(rhs);
                self.check_type_match(&lhs_ty, &rhs_ty, expr.span);
                match op {
                    BinaryOperator::Equal | BinaryOperator::NotEqual | 
                    BinaryOperator::Less | BinaryOperator::Greater |
                    BinaryOperator::LessEqual | BinaryOperator::GreaterEqual => {
                        TypeKey::Primitive(PrimitiveType::Bool)
                    }
                    _ => lhs_ty 
                }
            },

            ExpressionKind::StructLiteral { type_name, fields } => {
                if let Some(def_id) = self.resolve_path(type_name) {
                    let mut initialized_fields = Vec::new();
                    for init in fields {
                        let actual_ty = self.check_expr(&init.value);
                        initialized_fields.push((&init.field_name, actual_ty, init.value.span));
                    }

                    for (field_ident, actual_ty, span) in initialized_fields {
                        let field_name = &field_ident.name;
                        let expected_ty_opt = self.ctx.struct_fields
                            .get(&def_id)
                            .and_then(|fields| fields.get(field_name))
                            .cloned();

                        if let Some(expected_ty) = expected_ty_opt {
                            self.check_type_match(&expected_ty, &actual_ty, span);
                        } else {
                            if self.ctx.struct_fields.contains_key(&def_id) {
                                self.error(format!("Struct has no field named '{}'", field_name), field_ident.span);
                            } else {
                                self.error("Not a struct definition", type_name.span);
                                return TypeKey::Error; 
                            }
                        }
                    }
                    TypeKey::Named(def_id)
                } else {
                    self.error("Unknown struct type", type_name.span);
                    TypeKey::Error
                }
            },

            ExpressionKind::FieldAccess { receiver, field_name } => {
                let receiver_type = self.check_expr(receiver);
                if let TypeKey::Named(def_id) = receiver_type {
                    if let Some(fields) = self.ctx.struct_fields.get(&def_id) {
                        if let Some(field_ty) = fields.get(&field_name.name) {
                            field_ty.clone()
                        } else {
                            self.error(format!("Field '{}' not found", field_name.name), field_name.span);
                            TypeKey::Error
                        }
                    } else {
                        self.error("Type has no fields", field_name.span);
                        TypeKey::Error
                    }
                } else {
                    self.error(format!("Expected struct type, found {:?}", receiver_type), receiver.span);
                    TypeKey::Error
                }
            },

            ExpressionKind::Call { callee, arguments } => {
                let callee_type = self.check_expr(callee);
                if let TypeKey::Function { params, ret } = callee_type {
                    if params.len() != arguments.len() {
                        self.error(format!("Argument count mismatch"), expr.span);
                    }
                    for (arg_expr, param_ty) in arguments.iter().zip(params.iter()) {
                        let arg_actual_ty = self.check_expr(arg_expr);
                        self.check_type_match(param_ty, &arg_actual_ty, arg_expr.span);
                    }
                    if let Some(ret_ty) = ret { *ret_ty } else { TypeKey::Primitive(PrimitiveType::Unit) }
                } else {
                    if !matches!(callee_type, TypeKey::Error) {
                        self.error("Expected function type", callee.span);
                    }
                    TypeKey::Error
                }
            },

            ExpressionKind::MethodCall { receiver, method_name, arguments } => {
                let receiver_type = self.check_expr(receiver);
                if let TypeKey::Error = receiver_type { return TypeKey::Error; }

                let method_info = if let Some(methods) = self.ctx.method_registry.get(&receiver_type) {
                    methods.get(&method_name.name).cloned()
                } else { None };

                if let Some(info) = method_info {
                    if let Some(TypeKey::Function { params, ret }) = self.ctx.types.get(&info.def_id).cloned() {
                        let expected_args = if params.is_empty() { &[] } else { &params[1..] };
                        if expected_args.len() != arguments.len() {
                             self.error(format!("Method argument count mismatch"), expr.span);
                        }
                        for (arg_expr, param_ty) in arguments.iter().zip(expected_args.iter()) {
                            let arg_actual = self.check_expr(arg_expr);
                            self.check_type_match(param_ty, &arg_actual, arg_expr.span);
                        }
                        if let Some(r) = ret { *r } else { TypeKey::Primitive(PrimitiveType::Unit) }
                    } else { TypeKey::Error }
                } else {
                    self.error(format!("Method '{}' not found", method_name.name), method_name.span);
                    TypeKey::Error
                }
            },

            // --- 【补全】一元运算 ---
             ExpressionKind::Unary { op, operand } => {
                let inner = self.check_expr(operand);
                match op {
                    UnaryOperator::AddressOf => TypeKey::Pointer(Box::new(inner), Mutability::Constant),
                    UnaryOperator::Dereference => {
                        if let TypeKey::Pointer(base, _) = inner { *base } 
                        else { self.error("Cannot deref non-pointer", expr.span); TypeKey::Error }
                    }
                    // 补全：取负
                    UnaryOperator::Negate => {
                        let is_valid = match &inner {
                            // 1. 基础类型：必须是数值
                            TypeKey::Primitive(p) => self.is_numeric_type(p),
                            
                            // 2. 字面量：允许（变成负字面量）
                            TypeKey::IntegerLiteral(_) | TypeKey::FloatLiteral(_) => true,
                            
                            // 3. 错误恢复
                            TypeKey::Error => true,
                            
                            _ => false,
                        };

                        if !is_valid {
                            self.error(
                                format!("Cannot apply unary minus operator '-' to type {:?}", inner), 
                                expr.span
                            );
                            TypeKey::Error
                        } else {
                            // 【严格检查】禁止对无符号整数取负
                            // 例如：set u: u8 = 10; set x = -u; // Error!
                            if let TypeKey::Primitive(p) = &inner {
                                if !self.is_signed_numeric(p) {
                                    self.error(
                                        format!("Cannot negate unsigned integer type {:?}", p), 
                                        expr.span
                                    );
                                    // 报错后返回 Error 防止后续产生更多噪音，或者返回 inner 继续检查
                                    // 这里返回 inner 不影响后续逻辑结构
                                }
                            }
                            
                            // 字面量取负是合法的 (IntegerLiteral 只是一个数，还没定类型，取负后变为负数意图)
                            inner
                        }
                    }
                    // 补全：逻辑非
                    UnaryOperator::Not => {
                        self.check_type_match(&TypeKey::Primitive(PrimitiveType::Bool), &inner, expr.span);
                        TypeKey::Primitive(PrimitiveType::Bool)
                    }
                }
            },
            
            ExpressionKind::Cast { expr: _, target_type } => {
                self.resolve_ast_type(target_type)
            },
            
            _ => TypeKey::Error, 
        };

        self.record_type(expr.id, ty.clone());
        ty
    }

    /// ==================================================
    /// 辅助函数
    /// ==================================================
    // 检查表达式是否可以被赋值 (Is it a mutable L-Value?)
    fn check_lvalue_mutability(&mut self, expr: &Expression) {
        match &expr.kind {
            // Case 1: 变量 (Path)
            ExpressionKind::Path(path) => {
                // 既然 check_expr 已经跑过了，path_resolutions 里应该有记录
                // 如果没有，说明之前 resolve 失败了，这里就不报重复错误了
                if let Some(&def_id) = self.ctx.path_resolutions.get(&path.id) {
                    // 查 mutabilities 表
                    if let Some(&mutability) = self.ctx.mutabilities.get(&def_id) {
                        if mutability != Mutability::Mutable {
                            self.error(
                                format!("Cannot assign to immutable variable/parameter '{:?}'", path.segments.last().unwrap().name), 
                                expr.span
                            );
                        }
                    } else {
                        // 可能是全局变量或者 Module？如果没记录 mutability，默认不可变
                        // 或者这是一个 bug
                    }
                }
            }

            // Case 2: 字段访问 (s.x)
            ExpressionKind::FieldAccess { receiver, field_name: _ } => {
                // 如果要修改 s.x，那么 s 必须是可变的
                // 递归检查 receiver
                self.check_lvalue_mutability(receiver);
            }

            // Case 3: 解引用 (ptr^)
            ExpressionKind::Unary { op: UnaryOperator::Dereference, operand } => {
                // 如果要修改 ptr^，那么 ptr 必须是 *T (Mutable Pointer)，不能是 ^T (Const Pointer)
                let ptr_type = self.ctx.types.get(&operand.id).cloned().unwrap_or(TypeKey::Error);
                
                match ptr_type {
                    TypeKey::Pointer(_, Mutability::Constant) => {
                        self.error("Cannot assign to content of a const pointer (^T)", expr.span);
                    }
                    TypeKey::Pointer(_, Mutability::Mutable) => {
                        // OK: *T 允许修改内容
                    }
                    _ => {
                        // check_expr 应该已经报错 "Not a pointer" 了
                    }
                }
            }
            
            // Case 4: 数组索引 (arr[i])
            ExpressionKind::Index { target, .. } => {
                // 修改 arr[i]，意味着 arr 必须可变
                self.check_lvalue_mutability(target);
            }

            // 其他情况：字面量、函数调用结果等，都是右值 (R-Value)，不可赋值
            _ => {
                self.error("Invalid left-hand side of assignment", expr.span);
            }
        }
    }

    // 检查是否为数值类型 (整数 或 浮点数)
    fn is_numeric_type(&self, p: &PrimitiveType) -> bool {
        use PrimitiveType::*;
        matches!(p, 
            // 整数
            I8 | U8 | I16 | U16 | I32 | U32 | I64 | U64 | ISize | USize |
            // 浮点数
            F32 | F64
        )
    }

    // 检查是否为有符号数值 (Signed Int 或 Float)
    fn is_signed_numeric(&self, p: &PrimitiveType) -> bool {
        use PrimitiveType::*;
        matches!(p, 
            // 有符号整数
            I8 | I16 | I32 | I64 | ISize |
            // 浮点数 (天然有符号)
            F32 | F64
        )
    }

    fn resolve_ast_type(&mut self, ty: &Type) -> TypeKey {
        match &ty.kind {
            TypeKind::Primitive(p) => TypeKey::Primitive(*p),
            TypeKind::Named(path) => {
                if let Some(def_id) = self.resolve_path(path) {
                    TypeKey::Named(def_id)
                } else {
                    self.error(format!("Unknown type: {:?}", path.segments.last().unwrap().name), path.span);
                    TypeKey::Error
                }
            }
            TypeKind::Pointer { inner, mutability } => {
                let inner_key = self.resolve_ast_type(inner);
                TypeKey::Pointer(Box::new(inner_key), *mutability)
            }
            TypeKind::Array { inner, size } => {
                let inner_key = self.resolve_ast_type(inner);
                TypeKey::Array(Box::new(inner_key), *size)
            }
            TypeKind::Function { params, ret_type } => {
                let p = params.iter().map(|t| self.resolve_ast_type(t)).collect();
                let r = ret_type.as_ref().map(|t| Box::new(self.resolve_ast_type(t)));
                TypeKey::Function { params: p, ret: r }
            }
        }
    }

    fn resolve_path(&mut self, path: &Path) -> Option<DefId> {
        if path.segments.is_empty() { return None; }
        
        let first_seg = &path.segments[0];
        let mut current_def_id = None;

        for scope in self.scopes.iter().rev() {
            if let Some(&id) = scope.symbols.get(&first_seg.name) {
                current_def_id = Some(id);
                break;
            }
        }

        if current_def_id.is_none() { return None; }

        for segment in path.segments.iter().skip(1) {
            let parent_id = current_def_id.unwrap();
            if let Some(scope) = self.ctx.namespace_scopes.get(&parent_id) {
                if let Some(&child_id) = scope.symbols.get(&segment.name) {
                    current_def_id = Some(child_id);
                } else { return None; }
            } else { return None; }
        }
        
        if let Some(final_id) = current_def_id {
            self.ctx.path_resolutions.insert(path.id, final_id);
        }
        current_def_id
    }

    fn define_symbol(&mut self, name: String, id: DefId, span: Span) {
        let scope = self.scopes.last_mut().unwrap();
        if scope.symbols.contains_key(&name) {
            self.error(format!("Redefinition of '{}'", name), span);
        } else {
            scope.symbols.insert(name, id);
        }
    }
    
    fn enter_scope(&mut self, kind: ScopeKind) {
        self.scopes.push(Scope::new(kind));
    }
    
    fn exit_scope(&mut self) -> Scope {
        self.scopes.pop().expect("Scope unbalanced!")
    }
}