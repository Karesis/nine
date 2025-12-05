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
use crate::source::Span;
use crate::target::TargetMetrics;
use std::collections::HashMap;
use std::collections::HashSet;

/// 定义 ID：指向 AST 中的节点
pub type DefId = NodeId;

/// ======================================================
/// 语义类型键
/// ======================================================
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeKey {
    Primitive(PrimitiveType),
    
    // 修改：Named 改为 Instantiated
    // 既代表普通结构体 (args 为空)，也代表泛型实例 List<i32>
    // DefId 指向 StructDecl / EnumDecl / Typedef
    Instantiated {
        def_id: DefId,
        args: Vec<TypeKey>, 
    },

    // 新增：泛型占位符 (例如 T)
    // DefId 指向 AST 中的 GenericParam 节点
    GenericParam(DefId), 

    Pointer(Box<TypeKey>, Mutability),
    Array(Box<TypeKey>, u64),
    Function {
        params: Vec<TypeKey>,
        ret: Option<Box<TypeKey>>,
        is_variadic: bool,
    },
    
    IntegerLiteral(u64),
    FloatLiteral(u64),
    Error,
}

impl TypeKey {
    // 辅助：快速创建一个不带泛型参数的类型实例
    pub fn non_generic(def_id: DefId) -> Self {
        TypeKey::Instantiated {
            def_id,
            args: Vec::new(),
        }
    }

    // 检查类型是否包含泛型参数 (T)
    pub fn is_concrete(&self) -> bool {
        match self {
            TypeKey::GenericParam(_) => false,
            TypeKey::Instantiated { args, .. } => args.iter().all(|a| a.is_concrete()),
            TypeKey::Pointer(inner, _) => inner.is_concrete(),
            TypeKey::Array(inner, _) => inner.is_concrete(),
            TypeKey::Function { params, ret, .. } => {
                params.iter().all(|p| p.is_concrete()) && 
                ret.as_ref().map_or(true, |r| r.is_concrete())
            }
            _ => true,
        }
    }
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
    Generic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Private,
}

#[derive(Debug, Clone)]
pub struct Scope {
    pub kind: ScopeKind,
    pub symbols: HashMap<String, (DefId, Visibility)>,
}

impl Scope {
    pub fn new(kind: ScopeKind) -> Self {
        Self {
            kind,
            symbols: HashMap::new(),
        }
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
    // 专门用于布局计算：记录结构体字段的有序类型
    pub struct_definitions: HashMap<DefId, Vec<TypeKey>>,
    pub errors: Vec<AnalyzeError>,
    pub target: TargetMetrics,
    // Key: Struct DefId, Value: [T_id, U_id, ...]
    pub def_generic_params: HashMap<DefId, Vec<DefId>>,
    pub generic_param_defs: HashSet<DefId>, // 记录哪些 ID 是泛型参数
    // 记录泛型参数的约束
    // Key: 泛型参数 T 的 ID
    // Value: 约束列表 (例如 [Printable的TypeKey, Clone的TypeKey])
    pub generic_constraints: HashMap<DefId, Vec<TypeKey>>,

    // 收集需要生成的具体结构体实例
    // e.g. Instantiated(Box, [i32])
    pub concrete_structs: HashSet<TypeKey>,

    // 收集需要生成的具体函数实例
    // Key: (函数定义ID, 实参类型列表)
    // e.g. (foo_id, [i32])
    pub concrete_functions: HashSet<(DefId, Vec<TypeKey>)>,
}

impl AnalysisContext {
    pub fn new(target: TargetMetrics) -> Self {
        Self {
            namespace_scopes: HashMap::new(),
            method_registry: HashMap::new(),
            path_resolutions: HashMap::new(),
            types: HashMap::new(),
            struct_fields: HashMap::new(),
            mutabilities: HashMap::new(),
            mangled_names: HashMap::new(),
            constants: HashMap::new(),
            struct_definitions: HashMap::new(),
            errors: Vec::new(),
            def_generic_params: HashMap::new(),
            generic_param_defs: HashSet::new(),
            generic_constraints: HashMap::new(),
            concrete_structs: HashSet::new(),
            concrete_functions: HashSet::new(),
            target,
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
    pub fn new(target: TargetMetrics) -> Self {
        Self {
            ctx: AnalysisContext::new(target),
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
        // --- Loop 1: 注册所有的定义 (Mod, Struct, Fn, Const...) ---
        for item in items {
            match &item.kind {
                // 跳过 Import，留到 Loop 2 处理
                ItemKind::Import { .. } => continue,

                // --- 模块 ---
                ItemKind::ModuleDecl {
                    name,
                    items: sub_items,
                    is_pub,
                    ..
                } => {
                    let vis = if *is_pub {
                        Visibility::Public
                    } else {
                        Visibility::Private
                    };
                    self.define_symbol(name.name.clone(), item.id, vis, name.span);
                    self.record_type(item.id, TypeKey::non_generic(item.id));

                    if let Some(subs) = sub_items {
                        self.enter_scope(ScopeKind::Module);
                        self.module_path.push(name.name.clone());

                        self.scan_declarations(subs);

                        self.module_path.pop();
                        let module_scope = self.exit_scope();
                        self.ctx.namespace_scopes.insert(item.id, module_scope);
                    }
                }

                ItemKind::CapDecl(def) => {
                    let vis = if def.is_pub {
                        Visibility::Public
                    } else {
                        Visibility::Private
                    };
                    // 1. 注册符号 "Iterable"
                    self.define_symbol(def.name.name.clone(), item.id, vis, def.name.span);
                    
                    // 2. 注册类型 (Cap 本身也是一种类型，可以作为变量类型)
                    self.record_type(item.id, TypeKey::non_generic(item.id));
                }

                // --- 结构体 ---
                ItemKind::StructDecl(def) => {
                    let vis = if def.is_pub { Visibility::Public } else { Visibility::Private };
                    self.define_symbol(def.name.name.clone(), item.id, vis, def.name.span);

                    // 【核心补充 1】记录泛型参数顺序
                    // 将 <T, U> 的 ID 提取出来，按顺序存入表
                    let param_ids: Vec<DefId> = def.generics.iter().map(|g| g.id).collect();
                    self.ctx.def_generic_params.insert(item.id, param_ids.clone());

                    // 【核心补充 2】标记这些 ID 是“泛型参数” (用于 resolve_ast_type 识别)
                    for param_id in param_ids {
                        self.ctx.generic_param_defs.insert(param_id);
                    }

                    // 记录类型 (使用 helper)
                    self.record_type(item.id, TypeKey::non_generic(item.id));
                    
                    self.register_static_methods(item.id, &def.static_methods);
                }

                // --- 联合体 (逻辑同上) ---
                ItemKind::UnionDecl(def) => {
                    let vis = if def.is_pub { Visibility::Public } else { Visibility::Private };
                    self.define_symbol(def.name.name.clone(), item.id, vis, def.name.span);

                    let param_ids: Vec<DefId> = def.generics.iter().map(|g| g.id).collect();
                    self.ctx.def_generic_params.insert(item.id, param_ids.clone());
                    for param_id in param_ids { self.ctx.generic_param_defs.insert(param_id); }

                    self.record_type(item.id, TypeKey::non_generic(item.id));
                    self.register_static_methods(item.id, &def.static_methods);
                }

                // --- 枚举 ---
                ItemKind::EnumDecl(def) => {
                    let vis = if def.is_pub { Visibility::Public } else { Visibility::Private };
                    self.define_symbol(def.name.name.clone(), item.id, vis, def.name.span);

                    // 【核心补充】枚举也有泛型
                    let param_ids: Vec<DefId> = def.generics.iter().map(|g| g.id).collect();
                    self.ctx.def_generic_params.insert(item.id, param_ids.clone());
                    for param_id in param_ids { self.ctx.generic_param_defs.insert(param_id); }

                    self.record_type(item.id, TypeKey::non_generic(item.id));

                    // ... (Scope 和 Variant 的注册逻辑保持不变，不需要改动) ...
                    let mut enum_scope = Scope::new(ScopeKind::Module);
                    for variant in &def.variants {
                        enum_scope.symbols.insert(variant.name.name.clone(), (variant.id, Visibility::Public));
                    }
                    for method in &def.static_methods {
                        let m_vis = if method.is_pub { Visibility::Public } else { Visibility::Private };
                        if enum_scope.symbols.contains_key(&method.name.name) {
                            self.error(format!("Duplicate..."), method.span);
                        } else {
                            enum_scope.symbols.insert(method.name.name.clone(), (method.id, m_vis));
                        }
                    }
                    self.ctx.namespace_scopes.insert(item.id, enum_scope);
                }

                // --- 函数 ---
                ItemKind::FunctionDecl(def) => {
                    let vis = if def.is_pub { Visibility::Public } else { Visibility::Private };
                    self.define_symbol(def.name.name.clone(), item.id, vis, def.name.span);

                    // 【新增：填表逻辑】
                    let param_ids: Vec<DefId> = def.generics.iter().map(|g| g.id).collect();
                    
                    // 1. 存入通用泛型参数表
                    self.ctx.def_generic_params.insert(item.id, param_ids.clone());
                    
                    // 2. 标记 T 为泛型参数 (供 resolve_ast_type 识别)
                    for param_id in param_ids {
                        self.ctx.generic_param_defs.insert(param_id);
                    }

                    let mangled = if def.is_extern {
                        def.name.name.clone()
                    } else {
                        self.generate_mangled_name(&def.name.name)
                    };
                    self.ctx.mangled_names.insert(def.id, mangled);
                }

                // --- 类型别名 ---
                ItemKind::Typedef { name, is_pub, .. }
                | ItemKind::TypeAlias { name, is_pub, .. } => {
                    let vis = if *is_pub {
                        Visibility::Public
                    } else {
                        Visibility::Private
                    };
                    self.define_symbol(name.name.clone(), item.id, vis, name.span);
                }

                // --- 全局变量 ---
                ItemKind::GlobalVariable(def) => {
                    let vis = if def.is_pub {
                        Visibility::Public
                    } else {
                        Visibility::Private
                    };
                    self.define_symbol(def.name.name.clone(), item.id, vis, def.name.span);

                    let mangled = if def.is_extern {
                        def.name.name.clone()
                    } else {
                        self.generate_mangled_name(&def.name.name)
                    };
                    self.ctx.mangled_names.insert(item.id, mangled);
                    self.ctx.mutabilities.insert(item.id, def.modifier);
                }

                _ => {}
            }
        }

        // --- Loop 2: 处理导入 (Use) ---
        for item in items {
            // 注意这里解构出了 is_pub
            if let ItemKind::Import {
                path,
                alias,
                is_pub,
            } = &item.kind
            {
                // 1. 尝试解析路径
                if let Some(target_id) = self.resolve_path(path) {
                    // 2. 决定引入的名字
                    let name = if let Some(alias_ident) = alias {
                        alias_ident.name.clone()
                    } else {
                        path.segments.last().unwrap().name.clone()
                    };

                    // 3. 决定可见性 (Pub Use 支持)
                    let vis = if *is_pub {
                        Visibility::Public
                    } else {
                        Visibility::Private
                    };

                    // 4. 注册符号 (传入 vis)
                    self.define_symbol(name, target_id, vis, path.span);

                    self.ctx.path_resolutions.insert(path.id, target_id);

                    if let Some(ty) = self.ctx.types.get(&target_id).cloned() {
                        self.record_type(item.id, ty);
                    }
                } else {
                    self.error(
                        format!(
                            "Cannot resolve import '{:?}'",
                            path.segments.last().unwrap().name
                        ),
                        path.span,
                    );
                }
            }
        }
    }

    // 辅助函数更新：注册静态方法
    // 必须更新以匹配 Scope 的 (DefId, Visibility) 结构
    fn register_static_methods(&mut self, item_id: DefId, methods: &[FunctionDefinition]) {
        let mut static_scope = Scope::new(ScopeKind::Module);

        for method in methods {
            // 获取方法的可见性
            let m_vis = if method.is_pub {
                Visibility::Public
            } else {
                Visibility::Private
            };

            if static_scope.symbols.contains_key(&method.name.name) {
                self.error(
                    format!("Duplicate static method '{}'", method.name.name),
                    method.span,
                );
            } else {
                static_scope
                    .symbols
                    .insert(method.name.name.clone(), (method.id, m_vis));
            }
        }
        self.ctx.namespace_scopes.insert(item_id, static_scope);
    }

    /// 进入泛型作用域，并注册参数和解析约束
    /// 例如处理: <T: Printable + Clone>
    fn enter_generic_scope(&mut self, generics: &[GenericParam]) {
        // 1. 开启新作用域
        self.enter_scope(ScopeKind::Generic);

        for param in generics {
            // 2. 标记这个 ID 是泛型参数
            self.ctx.generic_param_defs.insert(param.id);

            // 3. 在当前作用域注册 T (这样结构体内部才能引用 T)
            self.define_symbol(
                param.name.name.clone(),
                param.id,
                Visibility::Private, // T 只在内部可见
                param.name.span,
            );

            // 4. 【关键】解析约束 (Constraints)
            // T: Printable + Clone
            let mut constraint_keys = Vec::new();
            
            // 遍历 Path 列表
            for constraint_path in &param.constraints {
                // 直接使用 resolve_path！
                if let Some(def_id) = self.resolve_path(constraint_path) {
                    
                    // 找到了 Cap 定义，把它转为 TypeKey::non_generic 存起来
                    // 如果 Cap 本身带泛型 (Iterable#<i32>)，resolve_ast_type 其实更合适，
                    // 但这里我们先简单处理为 Named。
                    // 为了支持带泛型的约束 (e.g. T: Iterable#<i32>)，
                    // 我们其实应该调用 self.resolve_ast_type_from_path(constraint_path)
                    // 但为了不引入太多变动，目前先只取 def_id。
                    
                    // 稍微 Hack 一下：如果约束是带泛型的 Path，我们最好把它当做 Type 解析
                    // 既然我们只是存 ID 给后续检查用，这里先只存 DefId 对应的 TypeKey
                    constraint_keys.push(TypeKey::non_generic(def_id));
                    
                } else {
                    self.error(
                        format!("Unknown capability constraint: {:?}", constraint_path.segments.last().unwrap().name),
                        constraint_path.span
                    );
                }
            }

            // 5. 存入 Context
            if !constraint_keys.is_empty() {
                self.ctx.generic_constraints.insert(param.id, constraint_keys);
            }
        }
    }

    /// ==================================================
    /// Phase 2: 实现扫描
    /// ==================================================
    fn scan_implementations(&mut self, items: &[Item]) {
        for item in items {
            match &item.kind {
                ItemKind::ModuleDecl {
                    name,
                    items: sub_items,
                    ..
                } => {
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

                ItemKind::Implementation {
                    generics, // <--- 1. 解构出 generics
                    target_type,
                    methods,
                    ..        // 暂时忽略 implements
                } => {
                    // 2. 【关键】开启泛型作用域
                    // 如果不加这一行，imp#<T> for List#<T> 里的 T 就无法识别
                    self.enter_generic_scope(generics);

                    // 3. 计算 Key (此时 T 已经在 scope 里了，能正常解析)
                    let key = self.resolve_ast_type(target_type);

                    // 错误处理：注意必须在 continue 前退出作用域，保持栈平衡
                    if let TypeKey::Error = key {
                        self.exit_scope(); 
                        continue;
                    }

                    // 4. 生成修饰名 (使用新的接受 &TypeKey 的版本)
                    // 此时 key 可能是 Instantiated { def_id: List, args: [GenericParam(T)] }
                    // 生成的名字可能是 "std_collections_List_GenParam_0" 之类的
                    let type_name = self.get_mangling_type_name(&key);

                    for method in methods {
                        // 格式：StructName_MethodName
                        // generate_mangled_name 会自动加上模块前缀
                        let combined_name = format!("{}_{}", type_name, method.name.name);
                        let mangled = self.generate_mangled_name(&combined_name);

                        // 注册到 mangled_names 表，供 Codegen 使用
                        self.ctx.mangled_names.insert(method.id, mangled);
                    }

                    // 5. 更新方法注册表 (逻辑基本不变)
                    let mut local_registry = self
                        .ctx
                        .method_registry
                        .get(&key)
                        .cloned()
                        .unwrap_or_default();

                    for method in methods {
                        if local_registry.contains_key(&method.name.name) {
                            self.error(
                                format!("Duplicate method '{}'", method.name.name),
                                method.name.span,
                            );
                        } else {
                            local_registry.insert(
                                method.name.name.clone(),
                                MethodInfo {
                                    name: method.name.name.clone(),
                                    def_id: method.id,
                                    is_pub: method.is_pub,
                                    span: method.span,
                                },
                            );
                        }
                    }

                    // 写回
                    self.ctx.method_registry.insert(key, local_registry);

                    // 6. 【关键】退出作用域
                    self.exit_scope();
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
                ItemKind::ModuleDecl {
                    items: sub_items, ..
                } => {
                    if let Some(subs) = sub_items {
                        if let Some(scope) = self.ctx.namespace_scopes.get(&item.id) {
                            self.scopes.push(scope.clone());
                            self.resolve_signatures(subs);
                            self.scopes.pop();
                        }
                    }
                }

                ItemKind::StructDecl(def) | ItemKind::UnionDecl(def) => {
                    // 1. 进入泛型作用域 (处理 <T: Cap>)
                    self.enter_generic_scope(&def.generics);

                    // 2. 解析字段 (此时字段类型可以使用 T)
                    let mut fields = HashMap::new();
                    let mut field_order_list = Vec::new();
                    for field in &def.fields {
                        let ty = self.resolve_ast_type(&field.ty);
                        fields.insert(field.name.name.clone(), ty.clone());
                        field_order_list.push(ty);
                    }
                    self.ctx.struct_fields.insert(item.id, fields);
                    self.ctx.struct_definitions.insert(item.id, field_order_list);

                    // 3. 解析静态方法
                    for method in &def.static_methods {
                        self.resolve_function_signature(method);
                    }

                    // 4. 退出作用域
                    self.exit_scope();
                }

                ItemKind::EnumDecl(def) => {
                    self.enter_generic_scope(&def.generics); // <--- 加入这行

                    let enum_type = TypeKey::non_generic(item.id); 
                    for variant in &def.variants {
                        self.record_type(variant.id, enum_type.clone());
                    }
                    for method in &def.static_methods {
                        self.resolve_function_signature(method);
                    }

                    self.exit_scope(); // <--- 退出
                }

                ItemKind::FunctionDecl(def) => {
                    // 函数也有泛型参数 fn foo<T>(...)
                    self.enter_scope(ScopeKind::Generic);
                    for param in &def.generics {
                        self.ctx.generic_param_defs.insert(param.id); // 确保标记为泛型
                        self.define_symbol(
                            param.name.name.clone(),
                            param.id,
                            Visibility::Private,
                            param.span
                        );
                    }

                    self.resolve_function_signature(def);
                    self.exit_scope();
                }

                ItemKind::CapDecl(def) => {
                    // Cap 也可能有泛型: cap Iterable<T>
                    self.enter_generic_scope(&def.generics); // <--- 加入这行

                    for method in &def.methods {
                        self.resolve_function_signature(method);
                    }

                    self.exit_scope(); // <--- 退出
                }

                ItemKind::Implementation { generics, target_type, methods, .. } => {
                     self.enter_scope(ScopeKind::Generic);
                     // 1. 注册 imp<T> 的 T
                     for param in generics {
                        self.ctx.generic_param_defs.insert(param.id);
                        self.define_symbol(
                            param.name.name.clone(),
                            param.id,
                            Visibility::Private,
                            param.span
                        );
                     }
                     
                     // 2. 解析 Target Type (e.g. Box<T>)
                     // 因为 T 已经在 scope 里了，resolve_ast_type 能正确处理 Box<T>
                     let _ = self.resolve_ast_type(target_type);
                     
                     for method in methods {
                        self.resolve_function_signature(method);
                     }
                     
                     self.exit_scope();
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
        let param_keys = func
            .params
            .iter()
            .map(|p| self.resolve_ast_type(&p.ty))
            .collect();

        let ret_key = if let Some(ret) = &func.return_type {
            Some(Box::new(self.resolve_ast_type(ret)))
        } else {
            None
        };

        let ty = TypeKey::Function {
            params: param_keys,
            ret: ret_key,
            is_variadic: func.is_variadic,
        };
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
        if expected == got {
            return;
        }

        match (expected, got) {
            (TypeKey::Primitive(p), TypeKey::IntegerLiteral(val)) => {
                if self.is_integer_type(p) {
                    if self.check_int_range(*p, *val) {
                        return;
                    } else {
                        self.error(format!("Literal {} out of range", val), span);
                        return;
                    }
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

        self.error(
            format!("Type mismatch: expected {:?}, found {:?}", expected, got),
            span,
        );
    }

    fn is_integer_type(&self, p: &PrimitiveType) -> bool {
        use PrimitiveType::*;
        matches!(
            p,
            I8 | U8 | I16 | U16 | I32 | U32 | I64 | U64 | ISize | USize
        )
    }

    fn check_int_range(&self, p: PrimitiveType, val: u64) -> bool {
        use PrimitiveType::*;
        match p {
            // === 固定宽度类型 ===
            U8 => val <= u8::MAX as u64,
            I8 => val <= (i8::MAX as u64) + 1, // 允许 abs(MIN) 即 128

            U16 => val <= u16::MAX as u64,
            I16 => val <= (i16::MAX as u64) + 1, // 允许 32768

            U32 => val <= u32::MAX as u64,
            I32 => val <= (i32::MAX as u64) + 1, // 允许 2147483648

            U64 => true,                         // val 是 u64，永远不会超过 u64
            I64 => val <= (i64::MAX as u64) + 1, // 允许 abs(i64::MIN)

            // === 平台相关类型 (Target Dependent) ===
            USize => {
                if self.ctx.target.ptr_byte_width == 4 {
                    // 32位平台: 不能超过 u32::MAX
                    val <= u32::MAX as u64
                } else {
                    // 64位平台: u64 范围内都行
                    true
                }
            }
            ISize => {
                if self.ctx.target.ptr_byte_width == 4 {
                    // 32位平台: 不能超过 abs(i32::MIN)
                    val <= (i32::MAX as u64) + 1
                } else {
                    // 64位平台: 不能超过 abs(i64::MIN)
                    val <= (i64::MAX as u64) + 1
                }
            }

            // 其他非整数类型
            _ => true,
        }
    }

    fn check_bodies(&mut self, items: &[Item]) {
        for item in items {
            match &item.kind {
                ItemKind::ModuleDecl {
                    items: sub_items, ..
                } => {
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
                    for method in methods {
                        self.check_function(method);
                    }
                }
                ItemKind::StructDecl(def) | ItemKind::UnionDecl(def) => {
                    for method in &def.static_methods {
                        self.check_function(method);
                    }
                }
                ItemKind::EnumDecl(def) => {
                    for method in &def.static_methods {
                        self.check_function(method);
                    }
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
            self.define_symbol(
                param.name.name.clone(),
                param.id,
                Visibility::Private,
                param.name.span,
            );
            let mutability = if param.is_mutable {
                Mutability::Mutable
            } else {
                Mutability::Immutable
            };
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
            StatementKind::VariableDeclaration {
                modifier,
                name,
                type_annotation,
                initializer,
                ..
            } => {
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
                            self.error(
                                "Constants (const) must be initialized immediately.",
                                stmt.span,
                            );
                        }
                        Mutability::Immutable => {
                            // 'set' 变量一旦定义就不能改，所以必须定义时就给值
                            self.error(
                                "Immutable variables (set) must be initialized immediately.",
                                stmt.span,
                            );
                        }
                        Mutability::Mutable => {
                            // 'mut' 允许不初始化 (保留语义上的“垃圾值”状态)
                            // 这里不做任何操作，允许 pass
                        }
                    }
                }

                // 2. 注册符号和类型
                self.define_symbol(name.name.clone(), stmt.id, Visibility::Private, name.span);
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

            StatementKind::ExpressionStatement(expr) => match expr.kind {
                ExpressionKind::Call { .. } | ExpressionKind::MethodCall { .. } => {
                    self.check_expr(expr);
                }
                _ => {
                    self.error("Only function calls can be used as statements.", expr.span);
                }
            },

            StatementKind::Return(opt_expr) => {
                let expected_ret_opt = self.current_return_type.clone();
                if let Some(expected) = expected_ret_opt {
                    match opt_expr {
                        Some(expr) => {
                            if let TypeKey::Primitive(PrimitiveType::Unit) = expected {
                                self.error(
                                    "Procedure (void function) cannot return a value.",
                                    expr.span,
                                );
                                self.check_expr(expr);
                            } else {
                                let actual = self.check_expr(expr);
                                self.check_type_match(&expected, &actual, expr.span);
                                self.coerce_literal_type(expr.id, &expected, &actual);
                            }
                        }
                        None => {
                            if let TypeKey::Primitive(PrimitiveType::Unit) = expected { /* OK */
                            } else {
                                self.error("Function must return a value.", stmt.span);
                            }
                        }
                    }
                } else {
                    self.error("Return statement outside of function context.", stmt.span);
                }
            }
            StatementKind::If {
                condition,
                then_block,
                else_branch,
            } => {
                let cond_ty = self.check_expr(condition);
                self.check_type_match(
                    &TypeKey::Primitive(PrimitiveType::Bool),
                    &cond_ty,
                    condition.span,
                );
                self.check_block(then_block);
                if let Some(else_stmt) = else_branch {
                    self.check_stmt(else_stmt);
                }
            }

            StatementKind::While {
                condition,
                init_statement,
                body,
            } => {
                self.enter_scope(ScopeKind::Loop);
                if let Some(init) = init_statement {
                    self.check_stmt(init);
                }

                let cond_ty = self.check_expr(condition);
                self.check_type_match(
                    &TypeKey::Primitive(PrimitiveType::Bool),
                    &cond_ty,
                    condition.span,
                );

                self.check_block(body);
                self.exit_scope();
            }

            StatementKind::Switch {
                target,
                cases,
                default_case,
            } => {
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
                    Mutability::Constant,
                ),
                Literal::Char(_) => TypeKey::Primitive(PrimitiveType::I32),
            },

            ExpressionKind::Path(path) => {
                // 1. 解析路径
                if let Some(def_id) = self.resolve_path(path) {
                    
                    // 2. 获取该定义的类型
                    if let Some(def_type) = self.get_type_of_def(def_id) {
                        
                        // 3. 【副作用】尝试收集泛型实例化信息
                        // 如果它是个函数且带了泛型参数，这里会记录下来
                        self.try_record_function_instantiation(def_id, path);

                        // 4. 返回类型
                        def_type
                    } else {
                        self.error("Symbol has no type", path.span);
                        TypeKey::Error
                    }
                } else {
                    self.error(
                        format!(
                            "Undefined symbol: {:?}",
                            path.segments.first().unwrap().name
                        ),
                        path.span,
                    );
                    TypeKey::Error
                }
            }

            ExpressionKind::Binary { lhs, op, rhs } => {
                let lhs_ty = self.check_expr(lhs);
                let rhs_ty = self.check_expr(rhs);

                // --- 类型统一逻辑 ---
                let common_type = match (&lhs_ty, &rhs_ty) {
                    // 1. 类型完全相同：直接通过
                    (t1, t2) if t1 == t2 => t1.clone(),

                    // 2. 左边是字面量，右边是具体整数：右边说了算
                    // e.g. 2 * a (i32) -> 结果为 i32
                    (TypeKey::IntegerLiteral(val), TypeKey::Primitive(p))
                        if self.is_integer_type(p) =>
                    {
                        if !self.check_int_range(*p, *val) {
                            self.error(
                                format!("Literal {} out of range for {:?}", val, p),
                                expr.span,
                            );
                        }
                        rhs_ty.clone()
                    }

                    // 3. 左边是具体整数，右边是字面量：左边说了算
                    // e.g. a (i32) * 2 -> 结果为 i32
                    (TypeKey::Primitive(p), TypeKey::IntegerLiteral(val))
                        if self.is_integer_type(p) =>
                    {
                        if !self.check_int_range(*p, *val) {
                            self.error(
                                format!("Literal {} out of range for {:?}", val, p),
                                expr.span,
                            );
                        }
                        lhs_ty.clone()
                    }

                    // 4. 浮点数同理 (FloatLiteral vs Primitive)
                    (TypeKey::FloatLiteral(_), TypeKey::Primitive(p))
                        if matches!(p, PrimitiveType::F32 | PrimitiveType::F64) =>
                    {
                        rhs_ty.clone()
                    }
                    (TypeKey::Primitive(p), TypeKey::FloatLiteral(_))
                        if matches!(p, PrimitiveType::F32 | PrimitiveType::F64) =>
                    {
                        lhs_ty.clone()
                    }

                    // 5. 无法统一 (比如 i32 + i64，或者 i32 + bool)
                    _ => {
                        // 复用 check_type_match 产生报错信息
                        self.check_type_match(&lhs_ty, &rhs_ty, expr.span);
                        TypeKey::Error
                    }
                };

                // 2. 双向固化：将类型信息回写到 AST
                self.coerce_literal_type(lhs.id, &common_type, &lhs_ty);
                self.coerce_literal_type(rhs.id, &common_type, &rhs_ty);

                // 3. 返回结果类型
                match op {
                    // 比较运算永远返回 Bool
                    BinaryOperator::Equal
                    | BinaryOperator::NotEqual
                    | BinaryOperator::Less
                    | BinaryOperator::Greater
                    | BinaryOperator::LessEqual
                    | BinaryOperator::GreaterEqual => TypeKey::Primitive(PrimitiveType::Bool),

                    // 算术运算返回统一后的类型
                    _ => common_type,
                }
            }

            ExpressionKind::StructLiteral { type_name, fields } => {
                // 1. 解析结构体名称拿到 DefId
                if let Some(def_id) = self.resolve_path(type_name) {
                    
                    // 2. 解析泛型实参 (e.g., Box#<i32>)
                    let mut type_args = Vec::new();
                    if let Some(last_seg) = type_name.segments.last() {
                        if let Some(ast_args) = &last_seg.generic_args {
                            for arg in ast_args {
                                type_args.push(self.resolve_ast_type(arg));
                            }
                        }
                    }

                    // 3. 构造完整的类型键 (Instantiated)
                    let struct_type = TypeKey::Instantiated { 
                        def_id, 
                        args: type_args.clone() // 后面替换时要用到 args
                    };

                    // 4. 检查这是否真的是一个结构体定义
                    if !self.ctx.struct_fields.contains_key(&def_id) {
                        self.error("Not a struct definition", type_name.span);
                        return TypeKey::Error;
                    }

                    // 5. 遍历用户填写的字段进行检查
                    for init in fields {
                        // A. 检查用户赋的值是什么类型
                        let actual_ty = self.check_expr(&init.value);
                        let field_name = &init.field_name.name;

                        // B. 查找定义中的字段类型 (可能是泛型 T)
                        let raw_field_ty_opt = self.ctx.struct_fields.get(&def_id)
                            .and_then(|fs| fs.get(field_name));

                        if let Some(raw_ty) = raw_field_ty_opt {
                            // C. 【关键步骤】执行泛型替换！
                            // 如果 raw_ty 是 T，且 args 是 [i32]，则 expected_ty 变为 i32
                            let expected_ty = self.substitute_generics(raw_ty, def_id, &type_args);

                            // D. 检查类型匹配
                            self.check_type_match(&expected_ty, &actual_ty, init.value.span);
                            
                            // E. 固化字面量类型 (如果赋值的是 10，这里确定它是 i32)
                            self.coerce_literal_type(init.value.id, &expected_ty, &actual_ty);
                        } else {
                            self.error(
                                format!("Struct has no field named '{}'", field_name),
                                init.field_name.span,
                            );
                        }
                    }

                    // 6. 记录整个表达式的类型
                    self.record_type(expr.id, struct_type.clone());
                    
                    // 7. 返回类型
                    struct_type
                } else {
                    self.error("Unknown struct type", type_name.span);
                    TypeKey::Error
                }
            }

            ExpressionKind::FieldAccess { receiver, field_name } => {
                let receiver_type = self.check_expr(receiver);
                
                // 解构 Instantiated，拿到 def_id 和 args
                if let TypeKey::Instantiated { def_id, args } = receiver_type {
                    if let Some(fields) = self.ctx.struct_fields.get(&def_id) {
                        if let Some(raw_field_ty) = fields.get(&field_name.name) {
                            
                            // obj 是 Box<i32> (args=[i32])，字段是 val: T
                            // 这里算出 val 的类型是 i32
                            let actual_field_ty = self.substitute_generics(raw_field_ty, def_id, &args);
                            
                            self.record_type(expr.id, actual_field_ty.clone());
                            actual_field_ty
                        } else {
                            self.error("Field not found", field_name.span);
                            TypeKey::Error
                        }
                    } else {
                        TypeKey::Error
                    }
                } else {
                    self.error("Expected struct type", receiver.span);
                    TypeKey::Error
                }
            }

            ExpressionKind::Call { callee, arguments } => {
                let callee_type = self.check_expr(callee);
                // 解构时加上 is_variadic
                if let TypeKey::Function {
                    params,
                    ret,
                    is_variadic,
                } = callee_type
                {
                    // 1. 检查参数数量
                    if is_variadic {
                        if arguments.len() < params.len() {
                            self.error(
                                format!(
                                    "Variadic function requires at least {} arguments",
                                    params.len()
                                ),
                                expr.span,
                            );
                        }
                    } else {
                        if params.len() != arguments.len() {
                            self.error(
                                format!(
                                    "Argument count mismatch: expected {}, got {}",
                                    params.len(),
                                    arguments.len()
                                ),
                                expr.span,
                            );
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
                                self.coerce_literal_type(
                                    arg_expr.id,
                                    &TypeKey::Primitive(PrimitiveType::F64),
                                    &arg_ty,
                                );
                            }
                        }
                    }

                    if let Some(ret_ty) = ret {
                        *ret_ty
                    } else {
                        TypeKey::Primitive(PrimitiveType::Unit)
                    }
                } else {
                    if !matches!(callee_type, TypeKey::Error) {
                        self.error("Expected function type", callee.span);
                    }
                    TypeKey::Error
                }
            }

            ExpressionKind::MethodCall {
                receiver,
                method_name,
                arguments,
            } => {
                let receiver_type = self.check_expr(receiver);
                if let TypeKey::Error = receiver_type {
                    return TypeKey::Error;
                }

                let method_info =
                    if let Some(methods) = self.ctx.method_registry.get(&receiver_type) {
                        methods.get(&method_name.name).cloned()
                    } else {
                        None
                    };

                if let Some(info) = method_info {
                    if let Some(TypeKey::Function { params, ret, .. }) =
                        self.ctx.types.get(&info.def_id).cloned()
                    {
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
                        if let Some(r) = ret {
                            *r
                        } else {
                            TypeKey::Primitive(PrimitiveType::Unit)
                        }
                    } else {
                        TypeKey::Error
                    }
                } else {
                    self.error(
                        format!("Method '{}' not found", method_name.name),
                        method_name.span,
                    );
                    TypeKey::Error
                }
            }

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
                    self.coerce_literal_type(
                        index.id,
                        &TypeKey::Primitive(PrimitiveType::I64),
                        &index_ty,
                    );
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
                        self.error(
                            format!("Type {:?} cannot be indexed", target_ty),
                            target.span,
                        );
                        TypeKey::Error
                    }
                }
            }

            ExpressionKind::StaticAccess { target, member } => {
                // 1. 检查 Target 类型
                let target_ty = self.check_expr(target);

                let container_id = if let TypeKey::Instantiated { def_id, .. } = target_ty {
                    def_id
                } else {
                    self.error("Expected Struct/Enum...", target.span);
                    return TypeKey::Error;
                };

                let symbol_info = if let Some(scope) = self.ctx.namespace_scopes.get(&container_id)
                {
                    scope.symbols.get(&member.name).cloned()
                } else {
                    None
                };
                if let Some((def_id, visibility)) = symbol_info {
                    // --- 可见性检查 ---
                    if visibility == Visibility::Private {
                        self.error(
                            format!("Static member '{}' is private", member.name),
                            member.span,
                        );
                        // 即使报错，我们继续往下走，记录类型以防级联报错
                    }

                    // 记录解析结果
                    self.ctx.path_resolutions.insert(expr.id, def_id);

                    // 返回类型
                    if let Some(ty) = self.get_type_of_def(def_id) {
                        self.record_type(expr.id, ty.clone());
                        return ty;
                    } else {
                        // 防御性：符号存在但没有类型记录
                        return TypeKey::Error;
                    }
                } else {
                    // 没找到符号
                    if self.ctx.namespace_scopes.contains_key(&container_id) {
                        self.error(
                            format!("Member '{}' not found in type", member.name),
                            member.span,
                        );
                    } else {
                        self.error("Type has no static members", target.span);
                    }
                    TypeKey::Error
                }
            }

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
                        if let TypeKey::Pointer(base, _) = inner {
                            *base
                        } else {
                            self.error("Cannot deref non-pointer", expr.span);
                            TypeKey::Error
                        }
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
                                format!(
                                    "Cannot apply unary minus operator '-' to type {:?}",
                                    inner
                                ),
                                expr.span,
                            );
                            TypeKey::Error
                        } else {
                            // 禁止对无符号整数取负
                            // e.g., set u: u8 = 10; set x = -u; // Error!
                            if let TypeKey::Primitive(p) = &inner {
                                if !self.is_signed_numeric(p) {
                                    self.error(
                                        format!("Cannot negate unsigned integer type {:?}", p),
                                        expr.span,
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
                        self.check_type_match(
                            &TypeKey::Primitive(PrimitiveType::Bool),
                            &inner,
                            expr.span,
                        );
                        TypeKey::Primitive(PrimitiveType::Bool)
                    }
                }
            }

            ExpressionKind::Cast {
                expr: src_expr,
                target_type,
            } => {
                // 1. 解析目标类型 (T)
                let target_ty = self.resolve_ast_type(target_type);

                // 2. 递归检查源表达式 (val)
                // 这会触发 Path 解析，填充 path_resolutions 和 types 表
                let src_ty = self.check_expr(src_expr);

                // 验证 Cast 是否合法
                if !self.validate_cast(&src_ty, &target_ty) {
                    self.error(
                        format!("Invalid cast from {:?} to {:?}", src_ty, target_ty),
                        expr.span,
                    );
                }
                // Cast 表达式的类型就是目标类型
                target_ty
            }

            ExpressionKind::SizeOf(target_type) => {
                // 1. 解析内部类型
                let key = self.resolve_ast_type(target_type);

                // 2. 将解析出的 TypeKey 记录到 Context 中
                self.record_type(target_type.id, key);

                // 3. 返回类型为 u64 (usize)
                TypeKey::Primitive(PrimitiveType::U64)
            }

            ExpressionKind::AlignOf(target_type) => {
                // 1. 解析内部类型
                let key = self.resolve_ast_type(target_type);
                // 2. 记录类型 (供 Codegen 使用)
                self.record_type(target_type.id, key);
                // 3. 返回结果类型 (usize/u64)
                TypeKey::Primitive(PrimitiveType::U64)
            }
        };

        self.record_type(expr.id, ty.clone());
        ty
    }

    /// ==================================================
    /// 辅助函数
    /// ==================================================
    
    // 辅助函数：尝试收集泛型函数的实例化信息
    fn try_record_function_instantiation(&mut self, def_id: DefId, path: &Path) {
        // Step 1: 先快速检查这是否是一个泛型定义，并获取参数数量
        // 注意：这里只拿 usize 长度，拿到后立即释放借用
        let expected_param_count = if let Some(ids) = self.ctx.def_generic_params.get(&def_id) {
            if ids.is_empty() {
                return; // 不是泛型函数，直接返回
            }
            ids.len()
        } else {
            return; // 根本没记录，不是泛型函数
        };

        // --- 此时对 self.ctx 的不可变借用已经结束 ---

        // Step 2: 提取并解析路径中的泛型实参
        // 这一步需要 &mut self (调用 resolve_ast_type)，现在是安全的
        let mut call_args = Vec::new();
        if let Some(seg) = path.segments.last() {
            if let Some(ast_args) = &seg.generic_args {
                for arg in ast_args {
                    call_args.push(self.resolve_ast_type(arg));
                }
            }
        }

        // Step 3: 校验参数数量
        if call_args.len() != expected_param_count {
            // 参数数量不对，忽略
            return;
        }

        // Step 4: 记录实例化
        // 这一步虽然又借用了 self.ctx (插入)，但之前的 &mut self 借用已经结束
        if call_args.iter().all(|a| a.is_concrete()) {
            self.ctx.concrete_functions.insert((def_id, call_args));
        }
    }

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
                                format!(
                                    "Cannot assign to immutable variable/parameter '{:?}'",
                                    path.segments.last().unwrap().name
                                ),
                                expr.span,
                            );
                        }
                    } else {
                        // 没记录 mutability 的（如函数名、模块名、或解析失败的符号）一律视为不可变
                    }
                }
            }

            // Case 2: 字段访问 (s.x)
            ExpressionKind::FieldAccess {
                receiver,
                field_name: _,
            } => {
                // 如果要修改 s.x，那么 s 必须是可变的
                // 递归检查 receiver
                self.check_lvalue_mutability(receiver);
            }

            // Case 3: 解引用 (ptr^)
            ExpressionKind::Unary {
                op: UnaryOperator::Dereference,
                operand,
            } => {
                // 如果要修改 ptr^，那么 ptr 必须是 *T (Mutable Pointer)，不能是 ^T (Const Pointer)
                let ptr_type = self
                    .ctx
                    .types
                    .get(&operand.id)
                    .cloned()
                    .unwrap_or(TypeKey::Error);

                match ptr_type {
                    TypeKey::Pointer(_, Mutability::Constant) => {
                        self.error(
                            "Cannot assign to content of a const pointer (^T)",
                            expr.span,
                        );
                    }
                    TypeKey::Pointer(_, Mutability::Mutable) => {
                        // OK: *T 允许修改内容
                    }
                    _ => {
                        // Pass.
                        // 如果走到这里，说明 operand 不是指针。
                        // check_expr 阶段肯定已经报过 "Cannot deref non-pointer" 错误了。
                        // 为了避免重复报错（级联错误），这里静默跳过。
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
                    *self
                        .ctx
                        .mutabilities
                        .get(&def_id)
                        .unwrap_or(&Mutability::Immutable)
                } else {
                    Mutability::Immutable
                }
            }

            // 2. 字段访问 (obj.field)：取决于 obj 是否可变
            ExpressionKind::FieldAccess { receiver, .. } => self.get_expr_mutability(receiver),

            // 3. 数组索引 (arr[i])：取决于 arr 是否可变
            ExpressionKind::Index { target, .. } => self.get_expr_mutability(target),

            // 4. 解引用 (ptr^)：取决于 ptr 本身是指向可变还是不可变
            // 如果 ptr 是 *T (Mutable Pointer)，那么 ptr^ 是可变的。
            // 如果 ptr 是 ^T (Const Pointer)，那么 ptr^ 是不可变的。
            ExpressionKind::Unary {
                op: UnaryOperator::Dereference,
                operand,
            } => {
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
        matches!(
            p,
            // 整数
            I8 | U8 | I16 | U16 | I32 | U32 | I64 | U64 | ISize | USize |
            // 浮点数
            F32 | F64
        )
    }

    // 检查是否为有符号数值 (Signed Int 或 Float)
    fn is_signed_numeric(&self, p: &PrimitiveType) -> bool {
        use PrimitiveType::*;
        matches!(
            p,
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
                    // 1. 检查是否是泛型参数 T
                    if self.ctx.generic_param_defs.contains(&def_id) {
                        return TypeKey::GenericParam(def_id);
                    }

                    // 2. 收集泛型实参 List<i32>
                    let mut args = Vec::new();
                    // 取 path 的最后一段看有没有 generic_args
                    if let Some(last_seg) = path.segments.last() {
                        if let Some(ast_args) = &last_seg.generic_args {
                            for arg in ast_args {
                                args.push(self.resolve_ast_type(arg));
                            }
                        }
                    }

                    // 3. 返回 Instantiated
                    let key = TypeKey::Instantiated { def_id, args };
                    // 如果是具体类型，且是结构体/枚举，加入生成队列
                    if key.is_concrete() {
                        // 这里简单判断一下 def_id 是 Struct/Enum 才加
                        if self.ctx.struct_fields.contains_key(&def_id) { 
                            self.ctx.concrete_structs.insert(key.clone());
                        }
                    }
                    key // 返回
                } else {
                    self.error(format!("Unknown type: ..."), path.span);
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
                let r = ret_type
                    .as_ref()
                    .map(|t| Box::new(self.resolve_ast_type(t)));
                TypeKey::Function {
                    params: p,
                    ret: r,
                    is_variadic: false,
                }
            }
        }
    }

    fn resolve_path(&mut self, path: &Path) -> Option<DefId> {
        if path.segments.is_empty() {
            return None;
        }

        let first_seg = &path.segments[0];
        let mut current_def_id = None;

        // 1. 查找第一段 (Local Scope)
        // 在本地作用域查找时，不需要检查可见性（自家东西随便用）
        for scope in self.scopes.iter().rev() {
            if let Some((id, _)) = scope.symbols.get(&first_seg.name) {
                // 解构元组
                current_def_id = Some(*id);
                break;
            }
        }

        if current_def_id.is_none() {
            return None;
        }

        // 2. 查找后续段 (Drill down)
        for (i, segment) in path.segments.iter().enumerate().skip(1) {
            let parent_id = current_def_id.unwrap();

            if let Some(scope) = self.ctx.namespace_scopes.get(&parent_id) {
                if let Some((child_id, visibility)) = scope.symbols.get(&segment.name) {
                    // 【关键检查】
                    // 如果我们在钻入另一个模块/结构体，必须检查目标是否 Public
                    // 这里做一个简单的判断：只有 Public 的符号允许通过 resolve_path 访问
                    // (除非我们是在 parent_id 对应的模块内部，但 resolve_path 很难判断这一点)
                    // 对于 v0.1，严格规则：跨层级访问必须是 Public。
                    if *visibility == Visibility::Private {
                        self.error(
                            format!("Symbol '{}' is private", segment.name),
                            segment.span,
                        );
                        // 为了避免级联报错，这里返回None
                        return None;
                    }

                    current_def_id = Some(*child_id);
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }

        if let Some(final_id) = current_def_id {
            self.ctx.path_resolutions.insert(path.id, final_id);
        }
        current_def_id
    }

    fn define_symbol(&mut self, name: String, id: DefId, vis: Visibility, span: Span) {
        let scope = self.scopes.last_mut().unwrap();
        if scope.symbols.contains_key(&name) {
            self.error(format!("Redefinition of '{}'", name), span);
        } else {
            scope.symbols.insert(name, (id, vis));
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

    // 根据语义类型生成修饰名 (用于 Codegen)
    // e.g. List<i32> -> "std_collections_List_i32"
    fn get_mangling_type_name(&self, ty: &TypeKey) -> String {
        match ty {
            // 基础类型直接转字符串
            TypeKey::Primitive(p) => format!("{:?}", p), 

            // 实例化的类型 (List<i32>)
            TypeKey::Instantiated { def_id, args } => {
                // 1. 查表获取基础名字 (e.g. "std_collections_List")
                // 我们在 scan_declarations 阶段已经填好了 mangled_names
                let base_name = self.ctx.mangled_names.get(def_id)
                    .cloned()
                    .unwrap_or_else(|| {
                        // 理论上只要 scan_decl 跑过，这里一定有值
                        // 除非是匿名类型或其他边缘情况
                        format!("Struct_{}", def_id.0) 
                    });

                if args.is_empty() {
                    base_name
                } else {
                    // 2. 递归拼接泛型参数
                    // List<i32, f64> -> List_I32_F64
                    let args_str: Vec<String> = args.iter()
                        .map(|arg| self.get_mangling_type_name(arg))
                        .collect();
                    format!("{}_{}", base_name, args_str.join("_"))
                }
            }

            // 指针类型
            TypeKey::Pointer(inner, _) => format!("Ptr_{}", self.get_mangling_type_name(inner)),
            
            // 数组类型
            TypeKey::Array(inner, size) => format!("Arr_{}_{}", size, self.get_mangling_type_name(inner)),

            // 函数指针 (稍微复杂点，把参数全拼起来)
            TypeKey::Function { params, ret, .. } => {
                let params_str: Vec<String> = params.iter()
                    .map(|p| self.get_mangling_type_name(p))
                    .collect();
                let ret_str = if let Some(r) = ret {
                    self.get_mangling_type_name(r)
                } else {
                    "Unit".to_string()
                };
                format!("Fn_{}_Ret_{}", params_str.join("_"), ret_str)
            }

            // 泛型参数本身 (T)
            // 在实例化后应该都被替换了，如果这里还能遇到 T，说明是在生成泛型模板本身的签名？
            // 或者用于调试。
            TypeKey::GenericParam(id) => format!("GenParam_{}", id.0),

            _ => "Unknown".to_string(),
        }
    }

    /// 尝试计算编译时常量表达式
    /// 如果计算成功，返回 Some(u64)
    /// 如果包含无法在编译期确定的内容（如函数调用、变量），返回 None
    fn eval_constant_expr(&mut self, expr: &Expression) -> Option<u64> {
        match &expr.kind {
            // 1. 字面量
            ExpressionKind::Literal(Literal::Integer(val)) => Some(*val),

            // 2. 引用其他常量
            ExpressionKind::Path(path) => {
                let def_id = self.ctx.path_resolutions.get(&path.id)?;
                // 只有当目标是 const 定义时，才能取值
                self.ctx.constants.get(def_id).cloned()
            }

            // 3. 二元运算 (递归计算)
            ExpressionKind::Binary { lhs, op, rhs } => {
                let l = self.eval_constant_expr(lhs)?;
                let r = self.eval_constant_expr(rhs)?;

                match op {
                    BinaryOperator::Add => Some(l.wrapping_add(r)),
                    BinaryOperator::Subtract => Some(l.wrapping_sub(r)),
                    BinaryOperator::Multiply => Some(l.wrapping_mul(r)),
                    BinaryOperator::Divide => {
                        if r == 0 {
                            None
                        } else {
                            Some(l / r)
                        }
                    }
                    BinaryOperator::Modulo => {
                        if r == 0 {
                            None
                        } else {
                            Some(l % r)
                        }
                    }

                    // 位运算
                    BinaryOperator::ShiftLeft => Some(l << r),
                    BinaryOperator::ShiftRight => Some(l >> r),
                    BinaryOperator::BitwiseAnd => Some(l & r),
                    BinaryOperator::BitwiseOr => Some(l | r),
                    BinaryOperator::BitwiseXor => Some(l ^ r),

                    _ => None, // 比较运算暂不支持作为数值常量
                }
            }

            // 4. 一元运算
            ExpressionKind::Unary {
                op: UnaryOperator::Negate,
                operand,
            } => {
                let val = self.eval_constant_expr(operand)?;
                // 这里按位取负 (u64 view)
                Some((-(val as i64)) as u64)
            }

            ExpressionKind::SizeOf(target_type) => {
                // 尝试获取类型的静态大小
                let key = if let Some(k) = self.ctx.types.get(&target_type.id) {
                    k.clone()
                } else {
                    self.resolve_ast_type(target_type)
                };

                self.get_type_static_size(&key)
            }

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
                                _ => Some(val),
                            }
                        }
                        // 2. 如果目标是指针 (如 0xB8000 as *u16)，保留地址值
                        TypeKey::Pointer(..) => Some(val),

                        // 其他情况 (如转 Float)，暂不支持常量求值
                        //? 更完善的consexpr机制？
                        _ => Some(val),
                    }
                } else {
                    // 理论上 check_expr 应该已经填好类型了，直接返回
                    Some(val)
                }
            }

            ExpressionKind::Literal(Literal::Boolean(b)) => Some(if *b { 1 } else { 0 }),

            ExpressionKind::AlignOf(target_type) => {
                // 获取类型键 (check_expr 已经存储，但为了健壮性，若没存则重新解析)
                let key = if let Some(k) = self.ctx.types.get(&target_type.id) {
                    k.clone()
                } else {
                    self.resolve_ast_type(target_type)
                };

                self.get_type_static_align(&key)
            }

            _ => None,
        }
    }

    /// 尝试在语义分析阶段计算类型大小
    /// 返回 None 表示该类型的大小依赖后端 Layout (如 Struct)，无法在 Analyzer 阶段确定
    fn get_type_static_size(&self, key: &TypeKey) -> Option<u64> {
        self.get_type_layout(key).map(|l| l.size)
    }

    /// 检查类型转换是否合法
    /// 规则参考 Rust/C:
    /// 1. 整数 <-> 整数 (包含 Bool, Char)
    /// 2. 整数 <-> 浮点
    /// 3. 浮点 <-> 浮点
    /// 4. 指针 <-> 整数 (size_t)
    /// 5. 指针 <-> 指针
    /// 6. 数组 -> 指针 (Array Decay)
    fn validate_cast(&self, src: &TypeKey, target: &TypeKey) -> bool {
        match (src, target) {
            // 1. 同类型转换：总是允许 (虽然多余)
            (t1, t2) if t1 == t2 => true,

            // 2. 基础数值类型互转 (Int, Float, Bool, Char)
            (TypeKey::Primitive(p1), TypeKey::Primitive(p2)) => {
                self.is_numeric_or_bool_char(p1) && self.is_numeric_or_bool_char(p2)
            }

            // 3. 字面量 -> 基础类型 (只要是数值)
            (TypeKey::IntegerLiteral(_), TypeKey::Primitive(p)) => self.is_numeric_or_bool_char(p),
            (TypeKey::FloatLiteral(_), TypeKey::Primitive(p)) => self.is_numeric_or_bool_char(p),

            // 4. 指针 <-> 整数
            (TypeKey::Pointer(..), TypeKey::Primitive(p)) => self.is_integer_type(p),
            (TypeKey::Primitive(p), TypeKey::Pointer(..)) => self.is_integer_type(p),
            // 字面量 -> 指针 (e.g. 0 as *void)
            (TypeKey::IntegerLiteral(_), TypeKey::Pointer(..)) => true,

            // 5. 指针 <-> 指针
            (TypeKey::Pointer(..), TypeKey::Pointer(..)) => true,

            // 6. 数组 -> 指针 (Decay)
            // [i32; 10] as *i32
            // 实际上允许数组转任意指针，不仅限于元素类型，因为这是 unsafe cast
            (TypeKey::Array(..), TypeKey::Pointer(..)) => true,

            // 其他非法
            // e.g. Struct -> Int, Struct -> Struct
            _ => false,
        }
    }

    fn get_type_static_align(&self, key: &TypeKey) -> Option<u64> {
        self.get_type_layout(key).map(|l| l.align)
    }

    fn is_numeric_or_bool_char(&self, p: &PrimitiveType) -> bool {
        self.is_numeric_type(p) || matches!(p, PrimitiveType::Bool)
        //? TODO: 引入unicode char
    }

    // 计算任意类型的布局
    fn get_type_layout(&self, key: &TypeKey) -> Option<Layout> {
        match key {
            TypeKey::Primitive(p) => {
                let s = self.get_primitive_size(p);
                Some(Layout::new(s, s)) // 基础类型：对齐 = 大小
            }
            // 指针固定
            TypeKey::Pointer(..) | TypeKey::Function { .. } => {
                let ptr_size = self.ctx.target.ptr_byte_width;
                let ptr_align = self.ctx.target.ptr_align;
                Some(Layout::new(ptr_size, ptr_align))
            }

            // 数组：Align 等于元素 Align，Size 等于 元素Size * N (数组整体大小必须包含 stride)
            TypeKey::Array(inner, count) => {
                let inner_layout = self.get_type_layout(inner)?;
                // 数组大小 = 元素步长(包含padding) * 数量
                // 元素的 size 已经是经过对齐修正的了
                Some(Layout::new(inner_layout.size * count, inner_layout.align))
            }

            // 结构体：核心逻辑
            TypeKey::Instantiated { def_id, args } => {
                // 这里传入 args 是为了计算布局时把 T 换成 i32
                self.compute_struct_layout(*def_id, args) 
            }

            TypeKey::GenericParam(_) => None,

            _ => None,
        }
    }

    // 复刻 C ABI 布局算法
    fn compute_struct_layout(&self, struct_id: DefId, args: &[TypeKey]) -> Option<Layout> {
        let field_types = self.ctx.struct_definitions.get(&struct_id)?;

        let mut current_offset = 0u64;
        let mut max_align = 1u64;

        for raw_field_type in field_types {
            // 【关键修复】先替换 T -> i32
            let actual_type = self.substitute_generics(raw_field_type, struct_id, args);
            
            // 然后计算 i32 的大小
            let field_layout = self.get_type_layout(&actual_type)?;

            let mask = field_layout.align - 1;
            if (current_offset & mask) != 0 {
                current_offset = (current_offset + mask) & !mask;
            }
            current_offset += field_layout.size;
            if field_layout.align > max_align {
                max_align = field_layout.align;
            }
        }

        let mask = max_align - 1;
        if (current_offset & mask) != 0 {
            current_offset = (current_offset + mask) & !mask;
        }

        Some(Layout::new(current_offset, max_align))
    }

    fn get_primitive_size(&self, p: &PrimitiveType) -> u64 {
        use PrimitiveType::*;
        match p {
            I8 | U8 | Bool => 1,
            I16 | U16 => 2,
            I32 | U32 | F32 => 4,
            I64 | U64 | F64 => 8,
            // 【关键】根据 target 决定
            ISize | USize => self.ctx.target.usize_width(),
            Unit => 0,
        }
    }

    // 辅助：判断 DefId 是否是泛型参数
    fn is_generic_param(&self, id: DefId) -> bool {
        self.ctx.generic_param_defs.contains(&id)
    }

    /// 泛型替换：将类型中的泛型参数 (T) 替换为具体的实参 (args)
    /// generic_def_id: 定义泛型的结构体/函数的 ID (用于查找 T 是第几个参数)
    /// args: 具体的实参列表 (如 [i32])
    fn substitute_generics(&self, ty: &TypeKey, generic_def_id: DefId, args: &[TypeKey]) -> TypeKey {
        match ty {
            // 1. 遇到泛型参数 T
            TypeKey::GenericParam(param_id) => {
                // 查表：这个 param_id 是 generic_def_id 的第几个参数？
                if let Some(param_list) = self.ctx.def_generic_params.get(&generic_def_id) {
                    if let Some(index) = param_list.iter().position(|id| id == param_id) {
                        // 找到了！T 是第 index 个参数
                        if let Some(arg_ty) = args.get(index) {
                            return arg_ty.clone(); // 替换为 i32
                        }
                    }
                }
                // 没找到或者越界？返回原样 (理论上不应发生，除非编译器 Bug)
                ty.clone() 
            }

            // 2. 递归替换内部类型
            TypeKey::Pointer(inner, mutability) => {
                let new_inner = self.substitute_generics(inner, generic_def_id, args);
                TypeKey::Pointer(Box::new(new_inner), *mutability)
            }
            
            TypeKey::Array(inner, size) => {
                let new_inner = self.substitute_generics(inner, generic_def_id, args);
                TypeKey::Array(Box::new(new_inner), *size)
            }

            // 3. 处理嵌套泛型实例: List<T> -> List<i32>
            TypeKey::Instantiated { def_id, args: inner_args } => {
                let new_args = inner_args.iter()
                    .map(|a| self.substitute_generics(a, generic_def_id, args))
                    .collect();
                TypeKey::Instantiated { def_id: *def_id, args: new_args }
            }

            // 4. 函数指针
            TypeKey::Function { params, ret, is_variadic } => {
                let new_params = params.iter()
                    .map(|p| self.substitute_generics(p, generic_def_id, args))
                    .collect();
                let new_ret = ret.as_ref()
                    .map(|r| Box::new(self.substitute_generics(r, generic_def_id, args)));
                
                TypeKey::Function { params: new_params, ret: new_ret, is_variadic: *is_variadic }
            }

            // 基础类型不变
            _ => ty.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub size: u64,
    pub align: u64,
}

impl Layout {
    pub fn new(size: u64, align: u64) -> Self {
        Self { size, align }
    }
}
