use std::collections::HashMap;
use crate::ast::*;
use crate::source::Span;
use crate::diagnostic::Diagnosable;

/// 定义 ID：指向 AST 中的节点
pub type DefId = NodeId;

/// ======================================================
/// 语义类型键 
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
        is_variadic: bool
    },
    /// 未定类型的整数字面量
    IntegerLiteral(u64),
    /// 浮点字面量: 存 f64 的 to_bits()
    FloatLiteral(u64),
    Error, 
}

/// ======================================================
/// 上下文与核心表结构
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
    pub mangled_names: HashMap<DefId, String>,
    pub constants: HashMap<DefId, u64>,
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
            mangled_names: HashMap::new(),
            constants: HashMap::new(),
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
    pub module_path: Vec<String>,
}

impl Analyzer {
    pub fn new() -> Self {
        Self {
            ctx: AnalysisContext::new(),
            scopes: Vec::new(),
            current_return_type: None,
            module_path: Vec::new(),
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
    /// Phase 1: 声明扫描 
    /// ==================================================
    fn scan_declarations(&mut self, items: &[Item]) {
        for item in items {
            match &item.kind {
                // --- 模块 ---
                ItemKind::ModuleDecl { name, items: sub_items, .. } => {
                    self.define_symbol(name.name.clone(), item.id, name.span);
                    self.record_type(item.id, TypeKey::Named(item.id));

                    if let Some(subs) = sub_items {
                        self.enter_scope(ScopeKind::Module);
                        
                        // 压入路径栈
                        self.module_path.push(name.name.clone());
                        
                        self.scan_declarations(subs); 
                        
                        // 弹出路径栈
                        self.module_path.pop();
                        
                        let module_scope = self.exit_scope();
                        self.ctx.namespace_scopes.insert(item.id, module_scope);
                    }
                }
                
                // --- 结构体 ---
                ItemKind::StructDecl(def) => {
                    self.define_symbol(def.name.name.clone(), item.id, def.name.span);
                    // 【新增】给结构体名注册类型，支持 Struct::new()
                    self.record_type(item.id, TypeKey::Named(item.id));
                    
                    self.register_static_methods(item.id, &def.static_methods);
                }

                // --- 联合体 ---
                ItemKind::UnionDecl(def) => {
                    self.define_symbol(def.name.name.clone(), item.id, def.name.span);
                    // 【新增】
                    self.record_type(item.id, TypeKey::Named(item.id));
                    
                    self.register_static_methods(item.id, &def.static_methods);
                }

                // --- 枚举 ---
                ItemKind::EnumDecl(def) => {
                    self.define_symbol(def.name.name.clone(), item.id, def.name.span);
                    self.record_type(item.id, TypeKey::Named(item.id));
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
                    
                    // 生成并记录修饰名
                    // e.g., math 模块下的 add -> math_add
                    let mangled = self.generate_mangled_name(&def.name.name);
                    self.ctx.mangled_names.insert(def.id, mangled);
                }

                ItemKind::Typedef { name, .. } | ItemKind::TypeAlias { name, .. } => {
                    self.define_symbol(name.name.clone(), item.id, name.span);
                }

                ItemKind::GlobalVariable(def) => {
                    // 1. 注册符号
                    self.define_symbol(def.name.name.clone(), item.id, def.name.span);
                    
                    // 2. 注册修饰名 
                    let mangled = self.generate_mangled_name(&def.name.name);
                    self.ctx.mangled_names.insert(item.id, mangled);
                    
                    // 3. 记录可变性 (供 get_expr_mutability 使用)
                    self.ctx.mutabilities.insert(item.id, def.modifier);
                }

                _ => {} 
            }
        }
    }

    // 辅助函数：注册静态方法 
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
    /// Phase 2: 实现扫描 
    /// ==================================================
    fn scan_implementations(&mut self, items: &[Item]) {
        for item in items {
             match &item.kind {
                ItemKind::ModuleDecl { name, items: sub_items, .. } => {
                    if let Some(subs) = sub_items {
                        if let Some(scope) = self.ctx.namespace_scopes.get(&item.id) {
                            self.scopes.push(scope.clone());
                            
                            // 压入模块路径
                            self.module_path.push(name.name.clone());
                            
                            self.scan_implementations(subs);
                            
                            // 弹出模块路径
                            self.module_path.pop();
                            
                            self.scopes.pop();
                        }
                    }
                }

                ItemKind::StructDecl(def) => {
                    let type_name = &def.name.name;
                    for m in &def.static_methods {
                        let combined_name = format!("{}_{}", type_name, m.name.name);
                        let mangled = self.generate_mangled_name(&combined_name);
                        self.ctx.mangled_names.insert(m.id, mangled);
                    }
                }
                
                ItemKind::EnumDecl(def) => {
                    let type_name = &def.name.name;
                    for m in &def.static_methods {
                        let combined_name = format!("{}_{}", type_name, m.name.name);
                        let mangled = self.generate_mangled_name(&combined_name);
                        self.ctx.mangled_names.insert(m.id, mangled);
                    }
                }

                ItemKind::Implementation { target_type, methods } => {
                    // 1. 计算 Key
                    let key = self.resolve_ast_type(target_type);
                    if let TypeKey::Error = key { continue; }

                    // 2. 生成修饰名并注册
                    let type_name = self.get_mangling_type_name(target_type);
                    
                    for method in methods {
                        // 格式：StructName_MethodName
                        // generate_mangled_name 会自动加上模块前缀：Module_StructName_MethodName
                        let combined_name = format!("{}_{}", type_name, method.name.name);
                        let mangled = self.generate_mangled_name(&combined_name);
                        
                        // 注册到 mangled_names 表，供 Codegen 使用
                        self.ctx.mangled_names.insert(method.id, mangled);
                    }

                    // 2. 取出当前的注册表（clone）
                    // 此时 self 的借用立即结束
                    let mut local_registry = self.ctx.method_registry
                        .get(&key)
                        .cloned()
                        .unwrap_or_default();
                    
                    // 3. 修改复印件
                    // 因为 local_registry 是本地变量，不借用 self，
                    // 所以这里的循环中调用self.error()是可以的
                    for method in methods {
                        if local_registry.contains_key(&method.name.name) {
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

                    // 4. 修改好的复印件写回
                    self.ctx.method_registry.insert(key, local_registry);
                }
                _ => {} 
             }
        }
    }

    /// ==================================================
    /// Phase 3: 签名解析 
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
                
                ItemKind::GlobalVariable(def) => {
                    let ty_key = self.resolve_ast_type(&def.ty);
                    self.record_type(item.id, ty_key);
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

        let ty = TypeKey::Function { params: param_keys, ret: ret_key, is_variadic: func.is_variadic };
        self.record_type(func.id, ty);
    }

    /// ==================================================
    /// Phase 4: 函数体检查
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
            I8 => val <= i8::MAX as u64, //? TODO: 未处理负数??处理了吗？
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

                ItemKind::GlobalVariable(def) => {
                    let declared_ty = self.ctx.types.get(&item.id).unwrap().clone();
                    
                    if let Some(init) = &def.initializer {
                        // 1. 检查类型 
                        let init_ty = self.check_expr(init);
                        self.check_type_match(&declared_ty, &init_ty, init.span);
                        self.coerce_literal_type(init.id, &declared_ty, &init_ty);
                        
                        // 全局初始化求值

                        // 尝试计算表达式的值 (例如 1 << 12)
                        if let Some(val) = self.eval_constant_expr(init) {
                            // 如果是 const 定义，必须注册到常量表，供后续引用
                            if def.modifier == Mutability::Constant {
                                self.ctx.constants.insert(item.id, val);
                            }
                            //? mut 和 set?单独开个map?
                            self.ctx.constants.insert(item.id, val);

                        } else {
                            //? TODO: 待实现实现 __cxa_atexit运行时全局构造机制
                            self.error("Global initializer must be a constant expression (integer arithmetic)", init.span);
                        }

                    } else {
                        if def.modifier != Mutability::Mutable {
                            self.error("Immutable/Const globals must be initialized", def.span);
                        }
                    }
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

        // 修改：只有当 body 存在时才检查
        if let Some(body) = &func.body {
            self.check_block(body);
        }

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
                    self.coerce_literal_type(init_expr.id, &declared_type, &init_type);
                    if *modifier == Mutability::Constant {
                        if let Some(val) = self.eval_constant_expr(init_expr) {
                            // 存入 Context 的常量表
                            self.ctx.constants.insert(stmt.id, val);
                            // println!("DEBUG: Constant '{}' evaluated to {}", name.name, val);
                        } else {
                            self.error("Constant value must be computable at compile time (literals only for now)", init_expr.span);
                        }
                    }
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

                // 2. 注册符号和类型 
                self.define_symbol(name.name.clone(), stmt.id, name.span);
                self.record_type(stmt.id, declared_type);
                
                // 3. 记录可变性 
                self.ctx.mutabilities.insert(stmt.id, *modifier);
            }

           StatementKind::Assignment { lhs, rhs } => {
                let lhs_ty = self.check_expr(lhs);
                let rhs_ty = self.check_expr(rhs);
                
                // 1. 类型匹配
                self.check_type_match(&lhs_ty, &rhs_ty, stmt.span);
                
                // 如果 rhs 是字面量，告诉它具体的类型
                self.coerce_literal_type(rhs.id, &lhs_ty, &rhs_ty);

                // 2. L-Value 可变性检查
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
                                self.coerce_literal_type(expr.id, &expected, &actual);
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
                        self.coerce_literal_type(pattern.id, &target_ty, &pattern_ty);
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

                // 尝试双向固化字面量
                // 如果 lhs 是 i32 变量，rhs 是字面量 50，这里会把 50 固化为 i32
                self.coerce_literal_type(rhs.id, &lhs_ty, &rhs_ty);
                // 反之亦然（例如 50 + x）
                self.coerce_literal_type(lhs.id, &rhs_ty, &lhs_ty);

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
                    
                    for init in fields {
                        // 1. 先计算实际给的值的类型 (e.g., IntegerLiteral(10))
                        let actual_ty = self.check_expr(&init.value);
                        
                        // 2. 查找结构体定义中这个字段期望的类型 (e.g., u8)
                        let field_name = &init.field_name.name;
                        let expected_ty_opt = self.ctx.struct_fields
                            .get(&def_id)
                            .and_then(|fs| fs.get(field_name))
                            .cloned();

                        // 3. 检查匹配 + 回写类型
                        if let Some(expected_ty) = expected_ty_opt {
                            // 检查类型是否兼容
                            self.check_type_match(&expected_ty, &actual_ty, init.value.span);
                            self.coerce_literal_type(init.value.id, &expected_ty, &actual_ty);

                        } else {
                            // 错误处理逻辑保持不变
                            if self.ctx.struct_fields.contains_key(&def_id) {
                                self.error(format!("Struct has no field named '{}'", field_name), init.field_name.span);
                            } else {
                                self.error("Not a struct definition", type_name.span);
                                //? 更多的sync和报错?
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
                // 解构时加上 is_variadic
                if let TypeKey::Function { params, ret, is_variadic } = callee_type {
                    
                    // 1. 检查参数数量
                    if is_variadic {
                        if arguments.len() < params.len() {
                            self.error(format!("Variadic function requires at least {} arguments", params.len()), expr.span);
                        }
                    } else {
                        if params.len() != arguments.len() {
                            self.error(format!("Argument count mismatch: expected {}, got {}", params.len(), arguments.len()), expr.span);
                        }
                    }

                    // 2. 检查固定参数的类型
                    // zip 只会迭代到最短的那一边，所以多出来的变长参数不会进入这个循环
                    for (arg_expr, param_ty) in arguments.iter().zip(params.iter()) {
                        let arg_actual_ty = self.check_expr(arg_expr);
                        self.check_type_match(param_ty, &arg_actual_ty, arg_expr.span);
                        self.coerce_literal_type(arg_expr.id, param_ty, &arg_actual_ty);
                    }

                    // 3. 处理变长参数 
                    // 对于多出来的参数，需要 check_expr 以便计算它们的类型（否则 Codegen 查表会 panic）
                    if arguments.len() > params.len() {
                        for arg_expr in &arguments[params.len()..] {
                            let arg_ty = self.check_expr(arg_expr);
                            
                            // C 语言的变长参数的Default Argument Promotions
                            // float -> double
                            // char/short -> int
                            // 如果是整数字面量，且没被约束，默认回写为 i32 (int 提升)
                            if let TypeKey::IntegerLiteral(val) = arg_ty {
                                // 如果超过 u32::MAX，说明必须用 i64
                                // 否则默认提升为 i32 
                                //? 确认检查+test?
                                let target_type = if val > u32::MAX as u64 {
                                    TypeKey::Primitive(PrimitiveType::I64)
                                } else {
                                    TypeKey::Primitive(PrimitiveType::I32)
                                };
                                self.coerce_literal_type(arg_expr.id, &target_type, &arg_ty);
                            }
                            // 如果是浮点字面量，默认回写为 f64 (double)
                            if let TypeKey::FloatLiteral(_) = arg_ty {
                                self.coerce_literal_type(arg_expr.id, &TypeKey::Primitive(PrimitiveType::F64), &arg_ty);
                            }
                        }
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
                    if let Some(TypeKey::Function { params, ret, .. }) = self.ctx.types.get(&info.def_id).cloned() {
                        let expected_args = if params.is_empty() { &[] } else { &params[1..] };
                        if expected_args.len() != arguments.len() {
                             self.error(format!("Method argument count mismatch"), expr.span);
                        }
                        for (arg_expr, param_ty) in arguments.iter().zip(expected_args.iter()) {
                            let arg_actual = self.check_expr(arg_expr);
                            self.check_type_match(param_ty, &arg_actual, arg_expr.span);
                            // 固化方法参数的字面量类型
                            self.coerce_literal_type(arg_expr.id, param_ty, &arg_actual);
                        }
                        if let Some(r) = ret { *r } else { TypeKey::Primitive(PrimitiveType::Unit) }
                    } else { TypeKey::Error }
                } else {
                    self.error(format!("Method '{}' not found", method_name.name), method_name.span);
                    TypeKey::Error
                }
            },

            ExpressionKind::Index { target, index } => {
                // 1. 递归检查目标（数组）
                let target_ty = self.check_expr(target);
                
                // 2. 递归检查索引（必须是整数）
                let index_ty = self.check_expr(index);
                
                // 检查索引是否为整数类型
                let is_index_valid = match &index_ty {
                    TypeKey::Primitive(p) => self.is_integer_type(p),
                    TypeKey::IntegerLiteral(_) => true,
                    _ => false,
                };

                if !is_index_valid {
                    self.error("Array index must be an integer", index.span);
                } else {
                    // 固化索引字面量类型 (统一固化i64)
                    self.coerce_literal_type(index.id, &TypeKey::Primitive(PrimitiveType::I64), &index_ty);
                }

                // 3. 检查 Target 类型并解包
                match target_ty {
                    TypeKey::Array(inner, _size) => {
                        // 如果是数组 [T; N]，索引操作的结果类型就是 T
                        *inner
                    }
                    
                    TypeKey::Pointer(_, _) => {
                        // 禁止指针使用 []
                        self.error("Cannot index a pointer with '[]'. Pointers and Arrays are distinct in 9-lang.", target.span);
                        TypeKey::Error
                    }
                    
                    TypeKey::Error => TypeKey::Error, // 级联错误，忽略
                    
                    _ => {
                        self.error(format!("Type {:?} cannot be indexed", target_ty), target.span);
                        TypeKey::Error
                    }
                }
            },

            ExpressionKind::StaticAccess { target, member } => {
                // 检查 Target 类型 (Struct Name 或 Enum Name)
                let target_ty = self.check_expr(target); 
                if let TypeKey::Named(container_id) = target_ty {
                    // 在 container (struct/enum) 的命名空间里查找 member
                    if let Some(scope) = self.ctx.namespace_scopes.get(&container_id) {
                        
                        if let Some(&def_id) = scope.symbols.get(&member.name) {
                            // 找到了(可能是静态方法，也可能是 Enum Variant)
                            let found_type = self.get_type_of_def(def_id);
                            //? found_types 检查?
                            
                            // 记录解析结果，供 Codegen 使用
                            self.ctx.path_resolutions.insert(expr.id, def_id);
                            
                            // 返回类型
                            if let Some(ty) = self.get_type_of_def(def_id) {
                                // 如果是 Enum Variant，已经处理为 IntegerLiteral
                                //? 这里只检查Function
                                self.record_type(expr.id, ty.clone());
                                return ty;
                            }
                        }
                    }
                }
                //? TODO: Fallback 到Enum IntegerLiteral?
                TypeKey::Error
            },

            // --- 一元运算 ---
            ExpressionKind::Unary { op, operand } => {
                let inner = self.check_expr(operand);
                match op {
                    UnaryOperator::AddressOf => {
                        // 1. 获取操作数的可变性
                        let mutability = self.get_expr_mutability(operand);
                        
                        // 2. 生成对应的指针类型
                        // 如果 operand 是 mut，生成 *T (Mutable)
                        // 否则生成 ^T (Constant)
                        TypeKey::Pointer(Box::new(inner), mutability)
                    }
                    UnaryOperator::Dereference => {
                        if let TypeKey::Pointer(base, _) = inner { *base } 
                        else { self.error("Cannot deref non-pointer", expr.span); TypeKey::Error }
                    }
                    // 取负
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
                            // 禁止对无符号整数取负
                            // e.g., set u: u8 = 10; set x = -u; // Error!
                            if let TypeKey::Primitive(p) = &inner {
                                if !self.is_signed_numeric(p) {
                                    self.error(
                                        format!("Cannot negate unsigned integer type {:?}", p), 
                                        expr.span
                                    );
                                    // 报错后返回 inner 继续检查
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
            
            ExpressionKind::Cast { expr: src_expr, target_type } => {
                // 1. 解析目标类型 (T)
                let target_ty = self.resolve_ast_type(target_type);

                // 2. 递归检查源表达式 (val)
                // 这会触发 Path 解析，填充 path_resolutions 和 types 表
                let src_ty = self.check_expr(src_expr);

                //? TODO: 3. 检查 Cast 是否合法
                //? e.g.,禁止 Struct 转 Int，或者做一些基础检查
                
                //? TODO: 4. 固化字面量(优化)
                //? 如果是 10 as u8，我们可以顺手把 10 固化为 u8，减少 Codegen 的 cast 指令
                //? 但 LLVM 优化器也会做这件事?
                //? self.coerce_literal_type(src_expr.id, &target_ty, &src_ty);

                // Cast 表达式的类型就是目标类型
                target_ty
            },
        };

        self.record_type(expr.id, ty.clone());
        ty
    }

    /// ==================================================
    /// 辅助函数
    /// ==================================================
    // 检查表达式是否可以被赋值 
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
                        //? 可能是全局变量或者 Module？如果没记录 mutability，默认不可变
                        //? 或者这是一个 bug?
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
                        //? check_expr 应该已经报错 "Not a pointer" 了
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

    // 辅助：判断一个表达式是否是可变的 (Mutable L-Value)
    // 用于决定取地址 (&) 操作生成的是 *T 还是 ^T
    fn get_expr_mutability(&self, expr: &Expression) -> Mutability {
        match &expr.kind {
            // 1. 变量：查表看定义时是不是 mut
            ExpressionKind::Path(path) => {
                if let Some(&def_id) = self.ctx.path_resolutions.get(&path.id) {
                    // 如果表里没记录（比如全局变量暂未处理），默认不可变
                    *self.ctx.mutabilities.get(&def_id).unwrap_or(&Mutability::Immutable)
                } else {
                    Mutability::Immutable
                }
            }

            // 2. 字段访问 (obj.field)：取决于 obj 是否可变
            ExpressionKind::FieldAccess { receiver, .. } => {
                self.get_expr_mutability(receiver)
            }

            // 3. 数组索引 (arr[i])：取决于 arr 是否可变
            ExpressionKind::Index { target, .. } => {
                self.get_expr_mutability(target)
            }

            // 4. 解引用 (ptr^)：取决于 ptr 本身是指向可变还是不可变
            // 如果 ptr 是 *T (Mutable Pointer)，那么 ptr^ 是可变的。
            // 如果 ptr 是 ^T (Const Pointer)，那么 ptr^ 是不可变的。
            ExpressionKind::Unary { op: UnaryOperator::Dereference, operand } => {
                if let Some(TypeKey::Pointer(_, mutability)) = self.ctx.types.get(&operand.id) {
                    *mutability
                } else {
                    Mutability::Immutable
                }
            }

            // 其他情况（字面量、函数调用返回值等）都是右值，不可变
            _ => Mutability::Immutable,
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

    // 辅助：类型强转回写
    // 当发现一个“未定类型的字面量”成功匹配了一个“具体类型”时，
    // 把这个具体类型更新到 AST 节点的类型表中，供 Codegen 使用。
    fn coerce_literal_type(&mut self, node_id: NodeId, expected: &TypeKey, actual: &TypeKey) {
        match actual {
            TypeKey::IntegerLiteral(_) | TypeKey::FloatLiteral(_) => {
                // 只有当期望的是具体的基础类型时（例如 u8, i32），我们才进行固化
                if let TypeKey::Primitive(_) = expected {
                    // 【覆盖旧记录】把 IntegerLiteral(...) 覆盖为 Primitive(...)
                    self.ctx.types.insert(node_id, expected.clone());
                }
            }
            _ => {}
        }
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
                TypeKey::Function { params: p, ret: r, is_variadic: false}
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

    fn generate_mangled_name(&self, name: &str) -> String {
        // 入口函数 main 不修饰
        if name == "main" {
            return name.to_string();
        }
        
        // 如果在根模块，直接返回名字
        if self.module_path.is_empty() {
            return name.to_string();
        }

        // 拼接：mod_submod_name
        let prefix = self.module_path.join("_");
        format!("{}_{}", prefix, name)
    }

    // 辅助：从 AST 类型中提取用于修饰名字的基础名称
    // e.g., Type(Path(Vector)) -> "Vector"
    //      Type(Pointer(Vector)) -> "Vector"
    fn get_mangling_type_name(&self, ty: &Type) -> String {
        match &ty.kind {
            TypeKind::Named(path) => {
                // 取路径的最后一段，例如 std::io::File -> File
                path.segments.last().unwrap().name.clone()
            },
            TypeKind::Pointer { inner, .. } => self.get_mangling_type_name(inner),
            TypeKind::Array { inner, .. } => format!("Arr_{}", self.get_mangling_type_name(inner)), //? 更“安全”的处理方式？
            TypeKind::Primitive(p) => format!("{:?}", p), // e.g. "I32"
            _ => "Unknown".to_string(),
        }
    }

    /// 尝试计算编译时常量表达式
    /// 如果计算成功，返回 Some(u64)
    /// 如果包含无法在编译期确定的内容（如函数调用、变量），返回 None
    fn eval_constant_expr(&self, expr: &Expression) -> Option<u64> {
        match &expr.kind {
            // 1. 字面量
            ExpressionKind::Literal(Literal::Integer(val)) => Some(*val),
            
            // 2. 引用其他常量
            ExpressionKind::Path(path) => {
                let def_id = self.ctx.path_resolutions.get(&path.id)?;
                // 只有当目标是 const 定义时，才能取值
                self.ctx.constants.get(def_id).cloned()
            },

            // 3. 二元运算 (递归计算)
            ExpressionKind::Binary { lhs, op, rhs } => {
                let l = self.eval_constant_expr(lhs)?;
                let r = self.eval_constant_expr(rhs)?;
                
                match op {
                    BinaryOperator::Add => Some(l.wrapping_add(r)),
                    BinaryOperator::Subtract => Some(l.wrapping_sub(r)),
                    BinaryOperator::Multiply => Some(l.wrapping_mul(r)),
                    BinaryOperator::Divide => if r == 0 { None } else { Some(l / r) },
                    BinaryOperator::Modulo => if r == 0 { None } else { Some(l % r) },
                    
                    // 位运算
                    BinaryOperator::ShiftLeft => Some(l << r),
                    BinaryOperator::ShiftRight => Some(l >> r),
                    BinaryOperator::BitwiseAnd => Some(l & r),
                    BinaryOperator::BitwiseOr  => Some(l | r),
                    BinaryOperator::BitwiseXor => Some(l ^ r),
                    
                    _ => None, // 比较运算暂不支持作为数值常量
                }
            },
            
            // 4. 一元运算
            ExpressionKind::Unary { op: UnaryOperator::Negate, operand } => {
                let val = self.eval_constant_expr(operand)?;
                // 这里按位取负 (u64 view)
                Some((-(val as i64)) as u64) 
            },

            ExpressionKind::Cast { expr: src_expr, .. } => {
                let val = self.eval_constant_expr(src_expr)?;
                
                // 获取 Cast 表达式的目标类型 (check_expr 已经计算并记录了)
                if let Some(target_key) = self.ctx.types.get(&expr.id) {
                    match target_key {
                        // 1. 如果目标是整数，模拟位宽截断truncate
                        // 用于计算掩码(e.g. val & 0xFF)
                        TypeKey::Primitive(p) => {
                            match p {
                                PrimitiveType::U8 | PrimitiveType::I8 => Some(val & 0xFF),
                                PrimitiveType::U16 | PrimitiveType::I16 => Some(val & 0xFFFF),
                                PrimitiveType::U32 | PrimitiveType::I32 => Some(val & 0xFFFFFFFF),
                                // 64位及其他保留原值
                                _ => Some(val)
                            }
                        }
                        // 2. 如果目标是指针 (如 0xB8000 as *u16)，保留地址值
                        TypeKey::Pointer(..) => Some(val),
                        
                        // 其他情况 (如转 Float)，暂不支持常量求值
                        //? 更完善的consexpr机制？
                        _ => Some(val) 
                    }
                } else {
                    // 理论上 check_expr 应该已经填好类型了，直接返回
                    Some(val)
                }
            },

            // 其他情况（函数调用等）不是常量
            //? 更完善的constexpr?
            _ => None,
        }
    }
}