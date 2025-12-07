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

    Instantiated {
        def_id: DefId,
        args: Vec<TypeKey>,
    },

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
    pub fn non_generic(def_id: DefId) -> Self {
        TypeKey::Instantiated {
            def_id,
            args: Vec::new(),
        }
    }

    pub fn is_concrete(&self) -> bool {
        match self {
            TypeKey::GenericParam(_) => false,
            TypeKey::Instantiated { args, .. } => args.iter().all(|a| a.is_concrete()),
            TypeKey::Pointer(inner, _) => inner.is_concrete(),
            TypeKey::Array(inner, _) => inner.is_concrete(),
            TypeKey::Function { params, ret, .. } => {
                params.iter().all(|p| p.is_concrete())
                    && ret.as_ref().map_or(true, |r| r.is_concrete())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefKind {
    Struct,
    Union,
    Enum,
    Cap,
    EnumVariant,
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

    pub struct_definitions: HashMap<DefId, Vec<(String, TypeKey)>>,
    pub errors: Vec<AnalyzeError>,
    pub target: TargetMetrics,

    pub def_generic_params: HashMap<DefId, Vec<DefId>>,
    pub generic_param_defs: HashSet<DefId>,

    pub generic_constraints: HashMap<DefId, Vec<TypeKey>>,

    pub concrete_structs: HashSet<TypeKey>,

    pub concrete_functions: HashSet<(DefId, Vec<TypeKey>)>,

    pub instantiated_structs: HashMap<String, Vec<(String, TypeKey)>>,

    pub non_generic_functions: HashSet<DefId>,

    pub node_generic_args: HashMap<NodeId, Vec<TypeKey>>,

    // 记录定义的种类 (Struct/Union/Enum)
    pub def_kind: HashMap<DefId, DefKind>,

    // 记录 Enum 的底层整数类型 (C-style Enum)
    // 如 enum Color : u8 { ... } -> u8
    // 如果没有显式指定，默认为 i32
    pub enum_underlying_types: HashMap<DefId, PrimitiveType>,

    // 实现注册表
    // 记录：某个具体类型 (TypeKey) 实现了 某个 Cap (DefId)
    // Value: 该实现块的 DefId
    // Key: (TypeKey, Cap_DefId) -> Impl_DefId
    pub impl_registry: HashMap<(TypeKey, DefId), DefId>,

    // Key: Cap DefId -> (Method Name -> MethodInfo)
    // 用于在泛型函数里查询 T: Cap 的方法
    pub cap_methods: HashMap<DefId, HashMap<String, MethodInfo>>,

    // 记录枚举成员对应的整数值
    // Key: Variant DefId -> Value: i64 (为了通用，存 i64)
    pub enum_variant_values: HashMap<DefId, i64>,
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
            instantiated_structs: HashMap::new(),
            non_generic_functions: HashSet::new(),
            node_generic_args: HashMap::new(),
            def_kind: HashMap::new(),
            enum_underlying_types: HashMap::new(),
            impl_registry: HashMap::new(),
            cap_methods: HashMap::new(),
            enum_variant_values: HashMap::new(),
            target,
        }
    }
}

#[derive(Debug)]
pub struct AnalyzeError {
    pub message: String,
    pub span: Span,
}

impl AnalysisContext {
    /// 泛型替换：将类型中的泛型参数 (T) 替换为具体的实参 (args)
    /// generic_def_id: 定义泛型的结构体/函数的 ID (用于查找 T 是第几个参数)
    /// args: 具体的实参列表 (如 [i32])
    pub fn substitute_generics(
        &self,
        ty: &TypeKey,
        generic_def_id: DefId,
        args: &[TypeKey],
    ) -> TypeKey {
        match ty {
            TypeKey::GenericParam(param_id) => {
                if let Some(param_list) = self.def_generic_params.get(&generic_def_id) {
                    if let Some(index) = param_list.iter().position(|id| id == param_id) {
                        if let Some(arg_ty) = args.get(index) {
                            return arg_ty.clone();
                        }
                    }
                }

                ty.clone()
            }

            TypeKey::Pointer(inner, mutability) => {
                let new_inner = self.substitute_generics(inner, generic_def_id, args);
                TypeKey::Pointer(Box::new(new_inner), *mutability)
            }

            TypeKey::Array(inner, size) => {
                let new_inner = self.substitute_generics(inner, generic_def_id, args);
                TypeKey::Array(Box::new(new_inner), *size)
            }

            TypeKey::Instantiated {
                def_id,
                args: inner_args,
            } => {
                let new_args = inner_args
                    .iter()
                    .map(|a| self.substitute_generics(a, generic_def_id, args))
                    .collect();
                TypeKey::Instantiated {
                    def_id: *def_id,
                    args: new_args,
                }
            }

            TypeKey::Function {
                params,
                ret,
                is_variadic,
            } => {
                let new_params = params
                    .iter()
                    .map(|p| self.substitute_generics(p, generic_def_id, args))
                    .collect();
                let new_ret = ret
                    .as_ref()
                    .map(|r| Box::new(self.substitute_generics(r, generic_def_id, args)));

                TypeKey::Function {
                    params: new_params,
                    ret: new_ret,
                    is_variadic: *is_variadic,
                }
            }

            _ => ty.clone(),
        }
    }

    pub fn get_mangling_type_name(&self, ty: &TypeKey) -> String {
        match ty {
            TypeKey::Primitive(p) => format!("{:?}", p),

            TypeKey::Instantiated { def_id, args } => {
                let base_name = self
                    .mangled_names
                    .get(def_id)
                    .cloned()
                    .unwrap_or_else(|| format!("Struct_{}", def_id.0));

                if args.is_empty() {
                    base_name
                } else {
                    let args_str: Vec<String> = args
                        .iter()
                        .map(|arg| self.get_mangling_type_name(arg))
                        .collect();
                    format!("{}_{}", base_name, args_str.join("_"))
                }
            }

            TypeKey::Pointer(inner, _) => format!("Ptr_{}", self.get_mangling_type_name(inner)),

            TypeKey::Array(inner, size) => {
                format!("Arr_{}_{}", size, self.get_mangling_type_name(inner))
            }

            TypeKey::Function { params, ret, .. } => {
                let params_str: Vec<String> = params
                    .iter()
                    .map(|p| self.get_mangling_type_name(p))
                    .collect();
                let ret_str = if let Some(r) = ret {
                    self.get_mangling_type_name(r)
                } else {
                    "Unit".to_string()
                };
                format!("Fn_{}_Ret_{}", params_str.join("_"), ret_str)
            }

            TypeKey::GenericParam(id) => format!("GenParam_{}", id.0),

            _ => "Unknown".to_string(),
        }
    }

    pub fn get_mangled_function_name(&self, def_id: DefId, args: &[TypeKey]) -> String {
        let base_name = self
            .mangled_names
            .get(&def_id)
            .cloned()
            .unwrap_or_else(|| format!("Fn{}", def_id.0));

        if args.is_empty() {
            base_name
        } else {
            let args_str: Vec<String> = args
                .iter()
                .map(|a| self.get_mangling_type_name(a))
                .collect();
            format!("{}_{}", base_name, args_str.join("_"))
        }
    }

    pub fn are_types_compatible(&self, concrete: &TypeKey, template: &TypeKey) -> bool {
        match (concrete, template) {
            (a, b) if a == b => true,

            (TypeKey::Instantiated { def_id: a, .. }, TypeKey::Instantiated { def_id: b, .. }) => {
                a == b
            }

            (TypeKey::Pointer(inner_a, mut_a), TypeKey::Pointer(inner_b, mut_b)) => {
                mut_a == mut_b && self.are_types_compatible(inner_a, inner_b)
            }

            (TypeKey::Array(inner_a, size_a), TypeKey::Array(inner_b, size_b)) => {
                size_a == size_b && self.are_types_compatible(inner_a, inner_b)
            }

            _ => false,
        }
    }

    fn extract_generic_args(&self, ty: &TypeKey) -> Vec<TypeKey> {
        match ty {
            TypeKey::Instantiated { args, .. } => args.clone(),
            TypeKey::Pointer(inner, _) => self.extract_generic_args(inner),
            TypeKey::Array(inner, _) => self.extract_generic_args(inner),
            _ => Vec::new(),
        }
    }

    pub fn extract_def_id(&self, ty: &TypeKey) -> Option<DefId> {
        match ty {
            TypeKey::Instantiated { def_id, .. } => Some(*def_id),

            TypeKey::Pointer(inner, _) => self.extract_def_id(inner),
            TypeKey::Array(inner, _) => self.extract_def_id(inner),
            _ => None,
        }
    }

    pub fn get_type_layout(&self, key: &TypeKey) -> Option<Layout> {
        match key {
            TypeKey::Primitive(p) => {
                let s = self.get_primitive_size(p);
                Some(Layout::new(s, s))
            }

            TypeKey::Pointer(..) | TypeKey::Function { .. } => {
                let ptr_size = self.target.ptr_byte_width;
                let ptr_align = self.target.ptr_align;
                Some(Layout::new(ptr_size, ptr_align))
            }

            TypeKey::Array(inner, count) => {
                let inner_layout = self.get_type_layout(inner)?;

                Some(Layout::new(inner_layout.size * count, inner_layout.align))
            }

            TypeKey::Instantiated { def_id, args } => self.compute_composite_layout(*def_id, args),

            TypeKey::GenericParam(_) => None,

            _ => None,
        }
    }

    fn get_primitive_size(&self, p: &PrimitiveType) -> u64 {
        use PrimitiveType::*;
        match p {
            I8 | U8 | Bool => 1,
            I16 | U16 => 2,
            I32 | U32 | F32 => 4,
            I64 | U64 | F64 => 8,

            ISize | USize => self.target.usize_width(),
            Unit => 0,
        }
    }

    fn compute_composite_layout(&self, def_id: DefId, args: &[TypeKey]) -> Option<Layout> {
        let kind = self
            .def_kind
            .get(&def_id)
            .cloned()
            .unwrap_or(DefKind::Struct); // 默认为 Struct 防止 panic
        match kind {
            // === Case A: 结构体 (字段顺序排列) ===
            DefKind::Struct => {
                let field_list = self.struct_definitions.get(&def_id)?;

                let mut current_offset = 0u64;
                let mut max_align = 1u64;

                for (_, raw_field_type) in field_list {
                    let actual_type = self.substitute_generics(raw_field_type, def_id, args);
                    let field_layout = self.get_type_layout(&actual_type)?;

                    // 对齐
                    let mask = field_layout.align - 1;
                    if (current_offset & mask) != 0 {
                        current_offset = (current_offset + mask) & !mask;
                    }

                    // 累加 Size
                    current_offset += field_layout.size;

                    // 更新 Max Align
                    if field_layout.align > max_align {
                        max_align = field_layout.align;
                    }
                }

                // 末尾填充
                let mask = max_align - 1;
                if (current_offset & mask) != 0 {
                    current_offset = (current_offset + mask) & !mask;
                }

                Some(Layout::new(current_offset, max_align))
            }

            // === Case B: 联合体 (字段重叠) ===
            DefKind::Union => {
                let field_list = self.struct_definitions.get(&def_id)?;

                let mut max_size = 0u64;
                let mut max_align = 1u64;

                for (_, raw_field_type) in field_list {
                    let actual_type = self.substitute_generics(raw_field_type, def_id, args);
                    let field_layout = self.get_type_layout(&actual_type)?;
                    if field_layout.size > max_size {
                        max_size = field_layout.size;
                    }
                    if field_layout.align > max_align {
                        max_align = field_layout.align;
                    }
                }
                let mask = max_align - 1;
                if (max_size & mask) != 0 {
                    max_size = (max_size + mask) & !mask;
                }

                Some(Layout::new(max_size, max_align))
            }

            // === Case C: 枚举 (C-Style) ===
            DefKind::Enum => {
                // 枚举本身的大小 = 底层整数类型的大小
                let underlying = *self.enum_underlying_types.get(&def_id)?;

                // 复用 get_primitive_size
                let size = self.get_primitive_size(&underlying);
                let align = size; // 整数通常 align = size

                Some(Layout::new(size, align))
            }

            // === Case D: Capability ===
            DefKind::Cap => {
                return None;
            }

            // === Case E: Enum Variant ===
            DefKind::EnumVariant => {
                return None;
            }
        }
    }
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

        self.scan_declarations(&program.items);

        self.scan_implementations(&program.items);

        self.resolve_signatures(&program.items);

        self.check_bodies(&program.items);

        self.finalize_monomorphization();
    }

    /// ==================================================
    /// Phase 1: 声明扫描
    /// ==================================================
    fn scan_declarations(&mut self, items: &[Item]) {
        for item in items {
            match &item.kind {
                ItemKind::Import { .. } => continue,

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

                // --- 接口 (Cap) ---
                ItemKind::CapDecl(def) => {
                    let vis = if def.is_pub {
                        Visibility::Public
                    } else {
                        Visibility::Private
                    };
                    self.define_symbol(def.name.name.clone(), item.id, vis, def.name.span);

                    // 1. 注册类型
                    self.record_type(item.id, TypeKey::non_generic(item.id));

                    // 2. 【这就是丢失的那一行！】必须标记它是个 Cap！
                    self.ctx.def_kind.insert(item.id, DefKind::Cap);

                    // 3. 注册泛型参数
                    let param_ids: Vec<DefId> = def.generics.iter().map(|g| g.id).collect();
                    self.ctx
                        .def_generic_params
                        .insert(item.id, param_ids.clone());
                    for param_id in param_ids.clone() {
                        self.ctx.generic_param_defs.insert(param_id);
                    }

                    // 4. 注册方法 (但不加入 non_generic_functions!)
                    let mut methods_map = HashMap::new();
                    for method in &def.methods {
                        methods_map.insert(
                            method.name.name.clone(),
                            MethodInfo {
                                name: method.name.name.clone(),
                                def_id: method.id,
                                is_pub: true,
                                span: method.span,
                            },
                        );
                        // 【核心修复】把方法 ID 也标记为 Cap！
                        // 这样 check_standard_method_call 里的保护逻辑才能生效
                        self.ctx.def_kind.insert(method.id, DefKind::Cap);
                        let combined = format!("{}_{}", def.name.name, method.name.name);
                        let mangled = self.generate_mangled_name(&combined);
                        self.ctx.mangled_names.insert(method.id, mangled);
                        self.ctx
                            .def_generic_params
                            .insert(method.id, param_ids.clone());

                        // 注意：确实不要在这里加 non_generic_functions.insert
                    }
                    self.ctx.cap_methods.insert(item.id, methods_map);
                }

                ItemKind::StructDecl(def) => {
                    let vis = if def.is_pub {
                        Visibility::Public
                    } else {
                        Visibility::Private
                    };
                    self.define_symbol(def.name.name.clone(), item.id, vis, def.name.span);

                    let param_ids: Vec<DefId> = def.generics.iter().map(|g| g.id).collect();

                    self.ctx
                        .def_generic_params
                        .insert(item.id, param_ids.clone());

                    for param_id in &param_ids {
                        self.ctx.generic_param_defs.insert(*param_id);
                    }

                    self.record_type(item.id, TypeKey::non_generic(item.id));

                    for method in &def.static_methods {
                        if method.generics.is_empty() {
                            self.ctx
                                .def_generic_params
                                .insert(method.id, param_ids.clone());
                        } else {
                        }
                    }

                    self.register_static_methods(item.id, &def.static_methods);
                    self.ctx.def_kind.insert(item.id, DefKind::Struct);
                }

                ItemKind::UnionDecl(def) => {
                    let vis = if def.is_pub {
                        Visibility::Public
                    } else {
                        Visibility::Private
                    };
                    self.define_symbol(def.name.name.clone(), item.id, vis, def.name.span);

                    let param_ids: Vec<DefId> = def.generics.iter().map(|g| g.id).collect();
                    self.ctx
                        .def_generic_params
                        .insert(item.id, param_ids.clone());
                    for param_id in param_ids {
                        self.ctx.generic_param_defs.insert(param_id);
                    }

                    self.record_type(item.id, TypeKey::non_generic(item.id));

                    let is_enum_concrete = def.generics.is_empty();

                    for method in &def.static_methods {
                        if is_enum_concrete && method.generics.is_empty() {
                            self.ctx.non_generic_functions.insert(method.id);
                        }
                    }

                    self.register_static_methods(item.id, &def.static_methods);
                    self.ctx.def_kind.insert(item.id, DefKind::Union);
                }

                ItemKind::EnumDecl(def) => {
                    let vis = if def.is_pub {
                        Visibility::Public
                    } else {
                        Visibility::Private
                    };
                    self.define_symbol(def.name.name.clone(), item.id, vis, def.name.span);

                    let param_ids: Vec<DefId> = def.generics.iter().map(|g| g.id).collect();
                    self.ctx
                        .def_generic_params
                        .insert(item.id, param_ids.clone());
                    for param_id in param_ids {
                        self.ctx.generic_param_defs.insert(param_id);
                    }
                    let enum_type = TypeKey::non_generic(item.id);
                    self.record_type(item.id, enum_type.clone());

                    let mut enum_scope = Scope::new(ScopeKind::Module);
                    let mut current_val = 0i64;

                    for variant in &def.variants {
                        // 1. 【核心修复】必须注册符号到 Scope！否则 Status::Ok 无法解析
                        enum_scope
                            .symbols
                            .insert(variant.name.name.clone(), (variant.id, Visibility::Public));

                        // 2. 标记 Kind (告诉 Codegen 这是一个 Variant)
                        self.ctx.def_kind.insert(variant.id, DefKind::EnumVariant);

                        // 3. 计算值 (支持 1+2 等常量表达式)
                        let val = if let Some(explicit_expr) = &variant.value {
                            if let Some(u64_val) = self.eval_constant_expr(explicit_expr) {
                                u64_val as i64
                            } else {
                                self.error(
                                    "Enum value must be a constant expression",
                                    explicit_expr.span,
                                );
                                0 // Fallback
                            }
                        } else {
                            current_val
                        };

                        self.ctx.enum_variant_values.insert(variant.id, val);
                        self.record_type(variant.id, enum_type.clone());
                        current_val = val + 1; // 自动递增
                    }

                    for method in &def.static_methods {
                        let m_vis = if method.is_pub {
                            Visibility::Public
                        } else {
                            Visibility::Private
                        };
                        if enum_scope.symbols.contains_key(&method.name.name) {
                            self.error(format!("Duplicate method..."), method.span);
                        } else {
                            enum_scope
                                .symbols
                                .insert(method.name.name.clone(), (method.id, m_vis));
                        }
                    }

                    self.ctx.namespace_scopes.insert(item.id, enum_scope);
                    self.ctx.def_kind.insert(item.id, DefKind::Enum);
                    let underlying = def.underlying_type.unwrap_or(PrimitiveType::I32);
                    self.ctx.enum_underlying_types.insert(item.id, underlying);
                }

                ItemKind::FunctionDecl(def) => {
                    let vis = if def.is_pub {
                        Visibility::Public
                    } else {
                        Visibility::Private
                    };

                    self.define_symbol(def.name.name.clone(), def.id, vis, def.name.span);

                    let param_ids: Vec<DefId> = def.generics.iter().map(|g| g.id).collect();
                    self.ctx
                        .def_generic_params
                        .insert(def.id, param_ids.clone());

                    for param_id in param_ids {
                        self.ctx.generic_param_defs.insert(param_id);
                    }

                    if def.generics.is_empty() {
                        self.ctx.non_generic_functions.insert(def.id);
                    }

                    let mangled = if def.is_extern {
                        def.name.name.clone()
                    } else {
                        self.generate_mangled_name(&def.name.name)
                    };
                    self.ctx.mangled_names.insert(def.id, mangled);
                }

                ItemKind::Typedef { name, is_pub, .. }
                | ItemKind::TypeAlias { name, is_pub, .. } => {
                    let vis = if *is_pub {
                        Visibility::Public
                    } else {
                        Visibility::Private
                    };
                    self.define_symbol(name.name.clone(), item.id, vis, name.span);
                }

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

        for item in items {
            if let ItemKind::Import {
                path,
                alias,
                is_pub,
            } = &item.kind
            {
                if let Some(target_id) = self.resolve_path(path) {
                    let name = if let Some(alias_ident) = alias {
                        alias_ident.name.clone()
                    } else {
                        path.segments.last().unwrap().name.clone()
                    };

                    let vis = if *is_pub {
                        Visibility::Public
                    } else {
                        Visibility::Private
                    };

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

    fn register_static_methods(&mut self, item_id: DefId, methods: &[FunctionDefinition]) {
        let mut static_scope = Scope::new(ScopeKind::Module);

        for method in methods {
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
        self.enter_scope(ScopeKind::Generic);

        for param in generics {
            self.ctx.generic_param_defs.insert(param.id);

            self.define_symbol(
                param.name.name.clone(),
                param.id,
                Visibility::Private,
                param.name.span,
            );

            // === 解析约束 (T: Converter + ...) ===
            let mut constraint_keys = Vec::new();

            for constraint_path in &param.constraints {
                // 1. 解析约束的 DefId
                if let Some(def_id) = self.resolve_path(constraint_path) {
                    // 2. 【核心修复】从 AST Path 中提取泛型参数 (e.g. <f64>)
                    let mut constraint_args = Vec::new();

                    // 检查 Path 的最后一段是否有 generic_args
                    if let Some(last_seg) = constraint_path.segments.last() {
                        if let Some(ast_args) = &last_seg.generic_args {
                            for arg_ty in ast_args {
                                // 递归解析参数类型 (f64 -> Primitive(F64))
                                constraint_args.push(self.resolve_ast_type(arg_ty));
                            }
                        }
                    }

                    // 3. 构造完整的 Instantiated TypeKey
                    let key = TypeKey::Instantiated {
                        def_id,
                        args: constraint_args, // 把解析出来的 [f64] 存进去！
                    };

                    constraint_keys.push(key);
                } else {
                    self.error(
                        format!(
                            "Unknown capability constraint: {:?}",
                            constraint_path.segments.last().unwrap().name
                        ),
                        constraint_path.span,
                    );
                }
            }

            if !constraint_keys.is_empty() {
                self.ctx
                    .generic_constraints
                    .insert(param.id, constraint_keys);
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

                            self.module_path.push(name.name.clone());

                            self.scan_implementations(subs);

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
                    generics,
                    implements,
                    target_type,
                    methods,
                } => {
                    // 1. 判断 imp 块本身是否带泛型
                    // (放在 enter_scope 之前也没问题，因为 generics 是 AST 里的字段)
                    let is_impl_concrete = generics.is_empty();

                    self.enter_generic_scope(generics);

                    let key = self.resolve_ast_type(target_type);
                    if let TypeKey::Error = key {
                        self.exit_scope();
                        continue;
                    }

                    // 检查是否是 Trait 实现
                    if let Some(cap_path) = implements {
                        if let Some(cap_def_id) = self.resolve_path(cap_path) {
                            if self.ctx.def_kind.get(&cap_def_id) == Some(&DefKind::Cap) {
                                self.ctx
                                    .impl_registry
                                    .insert((key.clone(), cap_def_id), item.id);
                            } else {
                                self.error("Expected a capability (cap) here", cap_path.span);
                            }
                        } else {
                            self.error("Unknown capability", cap_path.span);
                        }
                    }

                    let type_name = self.ctx.get_mangling_type_name(&key);
                    let param_ids: Vec<DefId> = generics.iter().map(|g| g.id).collect();

                    // --- 第一个循环：生成名字、处理泛型继承、注册待编译函数 ---
                    for method in methods {
                        let combined_name = format!("{}_{}", type_name, method.name.name);
                        let mangled = self.generate_mangled_name(&combined_name);
                        self.ctx.mangled_names.insert(method.id, mangled);

                        // 继承 Impl 的泛型参数
                        if method.generics.is_empty() {
                            self.ctx
                                .def_generic_params
                                .insert(method.id, param_ids.clone());
                        }

                        // 【核心修复】加入待编译列表
                        // 只有当 (Impl具体) 且 (方法无泛型) 时，才需要 Codegen 直接编译
                        if is_impl_concrete && method.generics.is_empty() {
                            self.ctx.non_generic_functions.insert(method.id);
                        }
                    }

                    // --- 第二个循环：构建方法注册表 ---
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
                    self.ctx.method_registry.insert(key, local_registry);

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
                    self.enter_generic_scope(&def.generics);

                    let mut fields = HashMap::new();
                    let mut field_order_list: Vec<(String, TypeKey)> = Vec::new();
                    for field in &def.fields {
                        let ty = self.resolve_ast_type(&field.ty);
                        fields.insert(field.name.name.clone(), ty.clone());

                        field_order_list.push((field.name.name.clone(), ty));
                    }
                    self.ctx.struct_fields.insert(item.id, fields);
                    self.ctx
                        .struct_definitions
                        .insert(item.id, field_order_list);

                    for method in &def.static_methods {
                        self.resolve_function_signature(method);
                    }

                    self.exit_scope();
                }

                ItemKind::EnumDecl(def) => {
                    self.enter_generic_scope(&def.generics);

                    let enum_type = TypeKey::non_generic(item.id);
                    for variant in &def.variants {
                        self.record_type(variant.id, enum_type.clone());
                    }
                    for method in &def.static_methods {
                        self.resolve_function_signature(method);
                    }

                    self.exit_scope();
                }

                ItemKind::FunctionDecl(def) => {
                    self.enter_scope(ScopeKind::Generic);
                    for param in &def.generics {
                        self.ctx.generic_param_defs.insert(param.id);
                        self.define_symbol(
                            param.name.name.clone(),
                            param.id,
                            Visibility::Private,
                            param.span,
                        );
                    }

                    self.resolve_function_signature(def);
                    self.exit_scope();
                }

                ItemKind::CapDecl(def) => {
                    self.enter_generic_scope(&def.generics);

                    self.define_symbol(
                        "Self".to_string(),
                        item.id,
                        Visibility::Private,
                        def.name.span,
                    );

                    for method in &def.methods {
                        self.resolve_function_signature(method);
                    }

                    self.exit_scope();
                }

                ItemKind::Implementation {
                    generics,
                    target_type,
                    methods,
                    ..
                } => {
                    self.enter_scope(ScopeKind::Generic);

                    for param in generics {
                        self.ctx.generic_param_defs.insert(param.id);
                        self.define_symbol(
                            param.name.name.clone(),
                            param.id,
                            Visibility::Private,
                            param.span,
                        );
                    }

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
            U8 => val <= u8::MAX as u64,
            I8 => val <= (i8::MAX as u64) + 1,

            U16 => val <= u16::MAX as u64,
            I16 => val <= (i16::MAX as u64) + 1,

            U32 => val <= u32::MAX as u64,
            I32 => val <= (i32::MAX as u64) + 1,

            U64 => true,
            I64 => val <= (i64::MAX as u64) + 1,

            USize => {
                if self.ctx.target.ptr_byte_width == 4 {
                    val <= u32::MAX as u64
                } else {
                    true
                }
            }
            ISize => {
                if self.ctx.target.ptr_byte_width == 4 {
                    val <= (i32::MAX as u64) + 1
                } else {
                    val <= (i64::MAX as u64) + 1
                }
            }

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
                ItemKind::Implementation {
                    generics, methods, ..
                } => {
                    self.enter_generic_scope(generics);
                    for method in methods {
                        self.check_function(method);
                    }
                    self.exit_scope();
                }
                ItemKind::StructDecl(def) => {
                    self.enter_generic_scope(&def.generics);

                    for method in &def.static_methods {
                        self.check_function(method);
                    }

                    self.exit_scope();
                }
                ItemKind::EnumDecl(def) => {
                    self.enter_generic_scope(&def.generics);
                    for method in &def.static_methods {
                        self.check_function(method);
                    }
                    self.exit_scope();
                }

                ItemKind::GlobalVariable(def) => {
                    let declared_ty = self.ctx.types.get(&item.id).unwrap().clone();

                    if let Some(init) = &def.initializer {
                        let init_ty = self.check_expr(init);
                        self.check_type_match(&declared_ty, &init_ty, init.span);
                        self.coerce_literal_type(init.id, &declared_ty, &init_ty);

                        if let Some(val) = self.eval_constant_expr(init) {
                            if def.modifier == Mutability::Constant {
                                self.ctx.constants.insert(item.id, val);
                            }

                            self.ctx.constants.insert(item.id, val);
                        } else {
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
        self.enter_generic_scope(&func.generics);
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

        if let Some(body) = &func.body {
            self.check_block(body);
        }

        self.current_return_type = prev_ret_type;
        self.exit_scope();
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

                if let Some(init_expr) = initializer {
                    let init_type = self.check_expr(init_expr);
                    self.check_type_match(&declared_type, &init_type, init_expr.span);
                    self.coerce_literal_type(init_expr.id, &declared_type, &init_type);
                    if *modifier == Mutability::Constant {
                        if let Some(val) = self.eval_constant_expr(init_expr) {
                            self.ctx.constants.insert(stmt.id, val);
                        } else {
                            self.error("Constant value must be computable at compile time (literals only for now)", init_expr.span);
                        }
                    }
                } else {
                    match modifier {
                        Mutability::Constant => {
                            self.error(
                                "Constants (const) must be initialized immediately.",
                                stmt.span,
                            );
                        }
                        Mutability::Immutable => {
                            self.error(
                                "Immutable variables (set) must be initialized immediately.",
                                stmt.span,
                            );
                        }
                        Mutability::Mutable => {}
                    }
                }

                self.define_symbol(name.name.clone(), stmt.id, Visibility::Private, name.span);
                self.record_type(stmt.id, declared_type);

                self.ctx.mutabilities.insert(stmt.id, *modifier);
            }

            StatementKind::Assignment { lhs, rhs } => {
                let lhs_ty = self.check_expr(lhs);
                let rhs_ty = self.check_expr(rhs);

                self.check_type_match(&lhs_ty, &rhs_ty, stmt.span);

                self.coerce_literal_type(rhs.id, &lhs_ty, &rhs_ty);

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

            ExpressionKind::Path(path) => self.check_path_expr(path, expr.id),

            ExpressionKind::Binary { lhs, op, rhs } => {
                let lhs_ty = self.check_expr(lhs);
                let rhs_ty = self.check_expr(rhs);

                let common_type = match (&lhs_ty, &rhs_ty) {
                    (t1, t2) if t1 == t2 => t1.clone(),

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

                    _ => {
                        self.check_type_match(&lhs_ty, &rhs_ty, expr.span);
                        TypeKey::Error
                    }
                };

                self.coerce_literal_type(lhs.id, &common_type, &lhs_ty);
                self.coerce_literal_type(rhs.id, &common_type, &rhs_ty);

                match op {
                    BinaryOperator::Equal
                    | BinaryOperator::NotEqual
                    | BinaryOperator::Less
                    | BinaryOperator::Greater
                    | BinaryOperator::LessEqual
                    | BinaryOperator::GreaterEqual => TypeKey::Primitive(PrimitiveType::Bool),

                    _ => common_type,
                }
            }

            ExpressionKind::StructLiteral { type_name, fields } => {
                if let Some(def_id) = self.resolve_path(type_name) {
                    let mut type_args = Vec::new();
                    if let Some(last_seg) = type_name.segments.last() {
                        if let Some(ast_args) = &last_seg.generic_args {
                            for arg in ast_args {
                                type_args.push(self.resolve_ast_type(arg));
                            }
                        }
                    }

                    let struct_type = TypeKey::Instantiated {
                        def_id,
                        args: type_args.clone(),
                    };

                    if !self.ctx.struct_fields.contains_key(&def_id) {
                        self.error("Not a struct definition", type_name.span);
                        return TypeKey::Error;
                    }

                    for init in fields {
                        let actual_ty = self.check_expr(&init.value);
                        let field_name = &init.field_name.name;

                        let raw_field_ty_opt = self
                            .ctx
                            .struct_fields
                            .get(&def_id)
                            .and_then(|fs| fs.get(field_name));

                        if let Some(raw_ty) = raw_field_ty_opt {
                            let expected_ty =
                                self.ctx.substitute_generics(raw_ty, def_id, &type_args);

                            self.check_type_match(&expected_ty, &actual_ty, init.value.span);

                            self.coerce_literal_type(init.value.id, &expected_ty, &actual_ty);
                        } else {
                            self.error(
                                format!("Struct has no field named '{}'", field_name),
                                init.field_name.span,
                            );
                        }
                    }

                    self.record_type(expr.id, struct_type.clone());

                    struct_type
                } else {
                    self.error("Unknown struct type", type_name.span);
                    TypeKey::Error
                }
            }

            ExpressionKind::FieldAccess {
                receiver,
                field_name,
            } => {
                let receiver_type = self.check_expr(receiver);

                if let TypeKey::Instantiated { def_id, args } = receiver_type {
                    if let Some(fields) = self.ctx.struct_fields.get(&def_id) {
                        if let Some(raw_field_ty) = fields.get(&field_name.name) {
                            let actual_field_ty =
                                self.ctx.substitute_generics(raw_field_ty, def_id, &args);

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

                if let TypeKey::Function {
                    params,
                    ret,
                    is_variadic,
                } = callee_type
                {
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

                    for (arg_expr, param_ty) in arguments.iter().zip(params.iter()) {
                        let arg_actual_ty = self.check_expr(arg_expr);
                        self.check_type_match(param_ty, &arg_actual_ty, arg_expr.span);
                        self.coerce_literal_type(arg_expr.id, param_ty, &arg_actual_ty);
                    }

                    if arguments.len() > params.len() {
                        for arg_expr in &arguments[params.len()..] {
                            let arg_ty = self.check_expr(arg_expr);

                            if let TypeKey::IntegerLiteral(val) = arg_ty {
                                let target_type = if val > u32::MAX as u64 {
                                    TypeKey::Primitive(PrimitiveType::I64)
                                } else {
                                    TypeKey::Primitive(PrimitiveType::I32)
                                };
                                self.coerce_literal_type(arg_expr.id, &target_type, &arg_ty);
                            }

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
            } => self.check_method_call_dispatch(expr.id, receiver, method_name, arguments),

            ExpressionKind::Index { target, index } => {
                let target_ty = self.check_expr(target);

                let index_ty = self.check_expr(index);

                let is_index_valid = match &index_ty {
                    TypeKey::Primitive(p) => self.is_integer_type(p),
                    TypeKey::IntegerLiteral(_) => true,
                    _ => false,
                };

                if !is_index_valid {
                    self.error("Array index must be an integer", index.span);
                } else {
                    self.coerce_literal_type(
                        index.id,
                        &TypeKey::Primitive(PrimitiveType::I64),
                        &index_ty,
                    );
                }

                match target_ty {
                    TypeKey::Array(inner, _size) => *inner,

                    TypeKey::Pointer(_, _) => {
                        self.error("Cannot index a pointer with '[]'. Pointers and Arrays are distinct in 9-lang.", target.span);
                        TypeKey::Error
                    }

                    TypeKey::Error => TypeKey::Error,

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
                let target_ty = self.check_expr(target);

                let container_id = match &target_ty {
                    TypeKey::Instantiated { def_id, args } => {
                        if !args.is_empty() {
                            self.ctx.node_generic_args.insert(expr.id, args.clone());
                        }
                        *def_id
                    }
                    _ => {
                        self.error(
                            format!(
                                "Expected Struct/Enum type for '::' access, found {:?}",
                                target_ty
                            ),
                            target.span,
                        );
                        return TypeKey::Error;
                    }
                };

                let symbol_info = if let Some(scope) = self.ctx.namespace_scopes.get(&container_id)
                {
                    scope.symbols.get(&member.name).cloned()
                } else {
                    None
                };

                if let Some((def_id, visibility)) = symbol_info {
                    if visibility == Visibility::Private {
                        self.error(
                            format!("Static member '{}' is private", member.name),
                            member.span,
                        );
                    }

                    self.ctx.path_resolutions.insert(expr.id, def_id);

                    if let Some(raw_ty) = self.get_type_of_def(def_id) {
                        eprintln!(
                            "check_expr: ExpressionKind::StaticAccess [DEBUG] StaticAccess Raw Type: {:?}",
                            raw_ty
                        );

                        if let TypeKey::Instantiated { args, .. } = &target_ty {
                            if !args.is_empty() {
                                let concrete_ty =
                                    self.ctx.substitute_generics(&raw_ty, container_id, args);

                                eprintln!(
                                    "check_expr: ExpressionKind::StaticAccess [DEBUG]   -> Substituted Type: {:?}",
                                    concrete_ty
                                );

                                self.record_type(expr.id, concrete_ty.clone());
                                return concrete_ty;
                            }
                        }

                        self.record_type(expr.id, raw_ty.clone());
                        raw_ty
                    } else {
                        TypeKey::Error
                    }
                } else {
                    if self.ctx.namespace_scopes.contains_key(&container_id) {
                        self.error(format!("Member '{}' not found", member.name), member.span);
                    } else {
                        self.error("Type has no static members", target.span);
                    }
                    TypeKey::Error
                }
            }

            ExpressionKind::Unary { op, operand } => {
                let inner = self.check_expr(operand);
                match op {
                    UnaryOperator::AddressOf => {
                        let mutability = self.get_expr_mutability(operand);

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

                    UnaryOperator::Negate => {
                        let is_valid = match &inner {
                            TypeKey::Primitive(p) => self.is_numeric_type(p),

                            TypeKey::IntegerLiteral(_) | TypeKey::FloatLiteral(_) => true,

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
                            if let TypeKey::Primitive(p) = &inner {
                                if !self.is_signed_numeric(p) {
                                    self.error(
                                        format!("Cannot negate unsigned integer type {:?}", p),
                                        expr.span,
                                    );
                                }
                            }

                            inner
                        }
                    }

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
                let target_ty = self.resolve_ast_type(target_type);

                // 如果解析目标类型出错，直接返回 Error，避免后续误报
                if let TypeKey::Error = target_ty {
                    return TypeKey::Error;
                }

                let src_ty = self.check_expr(src_expr);

                if !self.validate_cast(&src_ty, &target_ty) {
                    self.error(
                        format!("Invalid cast from {:?} to {:?}", src_ty, target_ty),
                        src_expr.span, // 注意：报错位置最好指向源表达式或整个 cast 表达式
                    );
                    // 即使报错，通常也返回 Error，或者返回 target_ty 防止级联报错
                    return TypeKey::Error;
                }

                // 【关键】告诉编译器：这个 Cast 表达式的结果类型是 target_ty
                self.record_type(expr.id, target_ty.clone());

                target_ty
            }

            ExpressionKind::SizeOf(target_type) => {
                let key = self.resolve_ast_type(target_type);

                self.record_type(target_type.id, key);

                TypeKey::Primitive(PrimitiveType::U64)
            }

            ExpressionKind::AlignOf(target_type) => {
                let key = self.resolve_ast_type(target_type);

                self.record_type(target_type.id, key);

                TypeKey::Primitive(PrimitiveType::U64)
            }
        };

        self.record_type(expr.id, ty.clone());
        ty
    }

    /// 1. MethodCall 总入口
    fn check_method_call_dispatch(
        &mut self,
        expr_id: NodeId,
        receiver: &Expression,
        method_name: &Identifier,
        arguments: &[Expression],
    ) -> TypeKey {
        let receiver_type = self.check_expr(receiver);
        if let TypeKey::Error = receiver_type {
            return TypeKey::Error;
        }

        if let Some(ty) =
            self.check_standard_method_call(expr_id, &receiver_type, method_name, arguments)
        {
            return ty;
        }

        if let Some(ty) =
            self.check_field_fn_ptr_call(expr_id, &receiver_type, method_name, arguments)
        {
            return ty;
        }

        self.error(
            format!(
                "Method or Field '{}' not found on type {:?}",
                method_name.name, receiver_type
            ),
            method_name.span,
        );
        TypeKey::Error
    }

    fn check_standard_method_call(
        &mut self,
        expr_id: NodeId,
        receiver_type: &TypeKey,
        method_name: &Identifier,
        arguments: &[Expression],
    ) -> Option<TypeKey> {
        // 1. 查找方法定义
        let (method_info, context_args) =
            self.find_method_in_registry(receiver_type, &method_name.name)?;

        // === [DEBUG 1] 检查查找结果和泛型注册情况 ===
        eprintln!("[DEBUG] Method Check: '{}'", method_name.name);
        eprintln!("[DEBUG]   Method DefId: {:?}", method_info.def_id);
        eprintln!("[DEBUG]   Found Context Args: {:?}", context_args);
        // 关键检查：Analyzer 认为这个方法 ID 拥有哪些泛型参数？
        eprintln!(
            "[DEBUG]   Registered Generic Params for Method: {:?}",
            self.ctx.def_generic_params.get(&method_info.def_id)
        );

        // 2. 记录泛型参数 (给 Codegen 用)
        if !context_args.is_empty() {
            let accepts_generics = self
                .ctx
                .def_generic_params
                .get(&method_info.def_id)
                .map(|ids| !ids.is_empty())
                .unwrap_or(false);

            // === [DEBUG] 泛型决策 ===
            eprintln!(
                "[DEBUG] Method: {}, DefId: {:?}",
                method_name.name, method_info.def_id
            );
            eprintln!(
                "[DEBUG]   Context Args (from trait/struct): {:?}",
                context_args
            );
            eprintln!(
                "[DEBUG]   Registered Params: {:?}",
                self.ctx.def_generic_params.get(&method_info.def_id)
            );
            eprintln!("[DEBUG]   Accepts Generics? {}", accepts_generics);

            if accepts_generics {
                eprintln!("[DEBUG] INSERTING generics for {}", method_name.name);
                eprintln!("[DEBUG]   -> ACTION: Recording generic args for Codegen.");
                self.ctx
                    .node_generic_args
                    .insert(expr_id, context_args.clone());

                if self.ctx.def_kind.get(&method_info.def_id) != Some(&DefKind::Cap) {
                    self.try_record_instantiation_with_args(method_info.def_id, &context_args);
                }
            } else {
                eprintln!("[DEBUG]   -> ACTION: Skipping generic args (Non-generic method).");
                eprintln!(
                    "[DEBUG] SKIPPING generics for {} (not generic)",
                    method_name.name
                );
            }
        }

        // 3. 检查参数与返回值
        if let Some(TypeKey::Function { params, ret, .. }) =
            self.ctx.types.get(&method_info.def_id).cloned()
        {
            let expected_args = if params.is_empty() { &[] } else { &params[1..] };

            if expected_args.len() != arguments.len() {
                self.error("Method arg count mismatch", method_name.span);
            }

            for (arg_expr, param_ty) in arguments.iter().zip(expected_args.iter()) {
                let arg_actual = self.check_expr(arg_expr);

                let mut expected_concrete_ty =
                    self.ctx
                        .substitute_generics(param_ty, method_info.def_id, &context_args);

                expected_concrete_ty = self.replace_self_type(
                    &expected_concrete_ty,
                    method_info.def_id,
                    receiver_type,
                );

                self.check_type_match(&expected_concrete_ty, &arg_actual, arg_expr.span);
                self.coerce_literal_type(arg_expr.id, &expected_concrete_ty, &arg_actual);
            }

            let ret_ty = if let Some(r) = ret {
                // === [DEBUG 2] 检查返回值替换 ===
                eprintln!("[DEBUG]   Original Ret Type: {:?}", r);

                let mut ty = self
                    .ctx
                    .substitute_generics(&r, method_info.def_id, &context_args);
                eprintln!("[DEBUG]   After Substitute: {:?}", ty);

                ty = self.replace_self_type(&ty, method_info.def_id, receiver_type);
                eprintln!("[DEBUG]   After Self Replace: {:?}", ty);

                ty
            } else {
                TypeKey::Primitive(PrimitiveType::Unit)
            };

            self.record_type(expr_id, ret_ty.clone());
            Some(ret_ty)
        } else {
            self.error("ICE: Method definition type missing", method_name.span);
            Some(TypeKey::Error)
        }
    }

    /// 路径 2：函数指针字段检查
    fn check_field_fn_ptr_call(
        &mut self,
        expr_id: NodeId,
        receiver_type: &TypeKey,
        method_name: &Identifier,
        arguments: &[Expression],
    ) -> Option<TypeKey> {
        let struct_id = self.ctx.extract_def_id(receiver_type)?;
        let struct_args = self.ctx.extract_generic_args(receiver_type);

        let fields = self.ctx.struct_fields.get(&struct_id)?;
        let raw_field_ty = fields.get(&method_name.name)?;

        let field_ty = self
            .ctx
            .substitute_generics(raw_field_ty, struct_id, &struct_args);

        if let TypeKey::Function { params, ret, .. } = field_ty {
            if params.len() != arguments.len() {
                self.error(
                    format!(
                        "Field function expects {} args, got {}",
                        params.len(),
                        arguments.len()
                    ),
                    method_name.span,
                );
            }

            for (arg_expr, param_ty) in arguments.iter().zip(params.iter()) {
                let arg_actual = self.check_expr(arg_expr);
                self.check_type_match(param_ty, &arg_actual, arg_expr.span);
                self.coerce_literal_type(arg_expr.id, param_ty, &arg_actual);
            }

            let ret_ty = if let Some(r) = ret {
                *r
            } else {
                TypeKey::Primitive(PrimitiveType::Unit)
            };
            self.record_type(expr_id, ret_ty.clone());
            Some(ret_ty)
        } else {
            self.error(
                format!("Field '{}' is not a function", method_name.name),
                method_name.span,
            );
            Some(TypeKey::Error)
        }
    }

    /// 在方法注册表中查找方法
    /// 返回: (方法信息, 该方法所属 Struct/Cap 的具体泛型实参)
    fn find_method_in_registry(
        &self,
        receiver_type: &TypeKey,
        method_name: &str,
    ) -> Option<(MethodInfo, Vec<TypeKey>)> {
        // [DEBUG] 入口日志
        eprintln!(
            "[DEBUG] find_method_in_registry: Looking for '{}' on type {:?}",
            method_name, receiver_type
        );

        // 1. 尝试精确匹配 (针对非泛型类型)
        if let Some(methods) = self.ctx.method_registry.get(receiver_type) {
            if let Some(info) = methods.get(method_name) {
                eprintln!("[DEBUG]   -> Match Found in Exact Registry!");
                let args = self.ctx.extract_generic_args(receiver_type);
                return Some((info.clone(), args));
            }
        }

        // 2. 尝试泛型定义模糊匹配 (Struct<T>)
        if let TypeKey::Instantiated { def_id, .. } = receiver_type {
            let args = self.ctx.extract_generic_args(receiver_type);

            for (key, methods) in &self.ctx.method_registry {
                if let TypeKey::Instantiated { def_id: k_id, .. } = key {
                    if k_id == def_id {
                        if let Some(info) = methods.get(method_name) {
                            eprintln!("[DEBUG]   -> Match Found in Fuzzy Registry (Struct)!");
                            return Some((info.clone(), args));
                        }
                    }
                }
            }
        }

        // 3. 【核心】尝试从泛型约束中查找 (Abstract Dispatch)
        if let TypeKey::GenericParam(param_id) = receiver_type {
            eprintln!(
                "[DEBUG]   -> Checking constraints for GenericParam({:?})...",
                param_id
            );

            if let Some(constraints) = self.ctx.generic_constraints.get(param_id) {
                eprintln!("[DEBUG]   -> Constraints found: {:?}", constraints);

                for constraint in constraints {
                    // 约束通常是 Instantiated { def_id: CapId, args: cap_args }
                    if let TypeKey::Instantiated {
                        def_id: cap_id,
                        args: cap_args,
                    } = constraint
                    {
                        eprintln!(
                            "[DEBUG]     -> Checking Cap {:?} with args {:?}",
                            cap_id, cap_args
                        );

                        // 去 cap_methods 表里查
                        if let Some(methods) = self.ctx.cap_methods.get(cap_id) {
                            if let Some(info) = methods.get(method_name) {
                                eprintln!(
                                    "[DEBUG]     -> Match Found in Cap Methods! Returning args: {:?}",
                                    cap_args
                                );
                                // 【必须！】直接使用约束里的参数 (cap_args)
                                return Some((info.clone(), cap_args.clone()));
                            } else {
                                eprintln!(
                                    "[DEBUG]     -> Cap found, but method '{}' not in it.",
                                    method_name
                                );
                            }
                        } else {
                            eprintln!(
                                "[DEBUG]     -> Cap ID {:?} not found in cap_methods table!",
                                cap_id
                            );
                        }
                    } else {
                        eprintln!(
                            "[DEBUG]     -> Constraint is not Instantiated?? {:?}",
                            constraint
                        );
                    }
                }
            } else {
                eprintln!("[DEBUG]   -> No constraints found for this param.");
            }
        }

        eprintln!("[DEBUG]   -> No match found.");
        None
    }

    /// 5. 终结阶段：预计算单态化数据
    /// 将收集到的泛型实例（生数据）转换为具体的字段列表（熟数据）
    fn finalize_monomorphization(&mut self) {
        let structs_to_process: Vec<TypeKey> = self.ctx.concrete_structs.iter().cloned().collect();

        for struct_key in structs_to_process {
            if let TypeKey::Instantiated { def_id, args } = &struct_key {
                let mangled_name = self.ctx.get_mangling_type_name(&struct_key);

                if let Some(raw_fields) = self.ctx.struct_definitions.get(def_id).cloned() {
                    let mut concrete_fields = Vec::new();

                    for (name, raw_ty) in raw_fields {
                        let concrete_ty = self.ctx.substitute_generics(&raw_ty, *def_id, args);
                        concrete_fields.push((name, concrete_ty));
                    }

                    self.ctx
                        .instantiated_structs
                        .insert(mangled_name, concrete_fields);
                }
            }
        }
    }

    /// ==================================================
    /// 辅助函数
    /// ==================================================

    fn try_record_function_instantiation(&mut self, def_id: DefId, path: &Path) {
        let expected_param_count = if let Some(ids) = self.ctx.def_generic_params.get(&def_id) {
            if ids.is_empty() {
                return;
            }
            ids.len()
        } else {
            return;
        };

        let mut call_args = Vec::new();
        for seg in &path.segments {
            if let Some(ast_args) = &seg.generic_args {
                for arg in ast_args {
                    call_args.push(self.resolve_ast_type(arg));
                }
            }
        }

        if call_args.len() != expected_param_count {
            return;
        }

        if call_args.iter().all(|a| a.is_concrete()) {
            self.ctx.concrete_functions.insert((def_id, call_args));
        }
    }

    /// 辅助函数：处理 Cap 方法签名中的 Self 替换
    /// 将类型中的 "Self" (表现为 Cap 定义本身) 替换为 实际接收者类型
    ///
    /// 参数:
    /// - ty: 要检查/替换的原始类型 (e.g. fn(Self) -> Self)
    /// - _method_def_id: 方法 ID (暂时用不到，但为了接口扩展性保留)
    /// - receiver_type: 实际的调用者类型 (e.g. T, i32)
    fn replace_self_type(
        &self,
        ty: &TypeKey,
        _method_def_id: DefId,
        receiver_type: &TypeKey,
    ) -> TypeKey {
        match ty {
            // === 核心逻辑 ===
            // 检查是否遇到了 Cap 定义，如果是，说明这就是 "Self"
            TypeKey::Instantiated { def_id, args } => {
                // 查 def_kind 表，看这个 ID 是不是 Cap
                if let Some(DefKind::Cap) = self.ctx.def_kind.get(def_id) {
                    // 命中！这就是 Self 占位符。
                    // 直接替换为实际的 receiver_type (例如 T 或 i32)
                    return receiver_type.clone();
                }

                // 如果不是 Cap (比如是 List<Self>)，则递归处理参数
                let new_args = args
                    .iter()
                    .map(|a| self.replace_self_type(a, _method_def_id, receiver_type))
                    .collect();

                TypeKey::Instantiated {
                    def_id: *def_id,
                    args: new_args,
                }
            }

            // === 递归处理其他复合类型 ===

            // 指针: ^Self -> ^T
            TypeKey::Pointer(inner, mutability) => {
                let new_inner = self.replace_self_type(inner, _method_def_id, receiver_type);
                TypeKey::Pointer(Box::new(new_inner), *mutability)
            }

            // 数组: Self[10] -> T[10]
            TypeKey::Array(inner, size) => {
                let new_inner = self.replace_self_type(inner, _method_def_id, receiver_type);
                TypeKey::Array(Box::new(new_inner), *size)
            }

            // 函数: fn(Self) -> Self -> fn(T) -> T
            TypeKey::Function {
                params,
                ret,
                is_variadic,
            } => {
                let new_params = params
                    .iter()
                    .map(|p| self.replace_self_type(p, _method_def_id, receiver_type))
                    .collect();

                let new_ret = ret
                    .as_ref()
                    .map(|r| Box::new(self.replace_self_type(r, _method_def_id, receiver_type)));

                TypeKey::Function {
                    params: new_params,
                    ret: new_ret,
                    is_variadic: *is_variadic,
                }
            }

            // === 基础类型保持不变 ===
            // Primitive(i32), GenericParam(T) 等
            _ => ty.clone(),
        }
    }
    // 检查具体类型是否满足约束
    fn check_constraint(&mut self, concrete_ty: &TypeKey, constraint_def_id: DefId, span: Span) {
        // 1. 简单查表 (优化：如果具体的 impl 已经存在，直接通过)
        // e.g. impl Printable for i32
        if self
            .ctx
            .impl_registry
            .contains_key(&(concrete_ty.clone(), constraint_def_id))
        {
            return;
        }

        // 2. 候选者收集 (解决借用冲突 + 准备做歧义检查)
        let mut candidates = Vec::new();

        for ((impl_pattern_ty, impl_cap_id), impl_def_id) in &self.ctx.impl_registry {
            // A. Cap ID 必须匹配
            if *impl_cap_id != constraint_def_id {
                continue;
            }

            // B. 尝试 Unify (模式匹配)
            let mut mapping = HashMap::new();
            if self.unify_types(concrete_ty, impl_pattern_ty, &mut mapping) {
                // 匹配成功！记录候选者
                candidates.push((*impl_def_id, mapping));
            }
        }

        // --- 至此，对 self.ctx 的借用结束 ---

        // 3. 严格一致性检查 (Strict Coherence)
        if candidates.is_empty() {
            // 没找到
            self.error(
                format!(
                    "Type {:?} does not satisfy capability constraint (DefId {:?})",
                    concrete_ty, constraint_def_id
                ),
                span,
            );
        } else if candidates.len() > 1 {
            // 找到多个！歧义！
            // 这就是解决 UB 的关键：拒绝猜测
            self.error(
                format!(
                    "Ambiguous implementations for {:?}: multiple impls apply.",
                    concrete_ty
                ),
                span,
            );
        } else {
            // 4. 递归验证约束 (Verification)
            // 找到了唯一的候选者，现在检查它的 Where Clauses 是否满足
            let (impl_id, mapping) = &candidates[0];
            self.verify_impl_obligations(*impl_id, mapping, span);
        }
    }

    fn check_lvalue_mutability(&mut self, expr: &Expression) {
        match &expr.kind {
            ExpressionKind::Path(path) => {
                if let Some(&def_id) = self.ctx.path_resolutions.get(&path.id) {
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
                    }
                }
            }

            ExpressionKind::FieldAccess {
                receiver,
                field_name: _,
            } => {
                self.check_lvalue_mutability(receiver);
            }

            ExpressionKind::Unary {
                op: UnaryOperator::Dereference,
                operand,
            } => {
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
                    TypeKey::Pointer(_, Mutability::Mutable) => {}
                    _ => {}
                }
            }

            ExpressionKind::Index { target, .. } => {
                self.check_lvalue_mutability(target);
            }

            _ => {
                self.error("Invalid left-hand side of assignment", expr.span);
            }
        }
    }

    fn get_expr_mutability(&self, expr: &Expression) -> Mutability {
        match &expr.kind {
            ExpressionKind::Path(path) => {
                if let Some(&def_id) = self.ctx.path_resolutions.get(&path.id) {
                    *self
                        .ctx
                        .mutabilities
                        .get(&def_id)
                        .unwrap_or(&Mutability::Immutable)
                } else {
                    Mutability::Immutable
                }
            }

            ExpressionKind::FieldAccess { receiver, .. } => self.get_expr_mutability(receiver),

            ExpressionKind::Index { target, .. } => self.get_expr_mutability(target),

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

            _ => Mutability::Immutable,
        }
    }

    fn is_numeric_type(&self, p: &PrimitiveType) -> bool {
        use PrimitiveType::*;
        matches!(
            p,
            I8 | U8 | I16 | U16 | I32 | U32 | I64 | U64 | ISize | USize | F32 | F64
        )
    }

    fn is_signed_numeric(&self, p: &PrimitiveType) -> bool {
        use PrimitiveType::*;
        matches!(p, I8 | I16 | I32 | I64 | ISize | F32 | F64)
    }

    fn coerce_literal_type(&mut self, node_id: NodeId, expected: &TypeKey, actual: &TypeKey) {
        match actual {
            TypeKey::IntegerLiteral(_) | TypeKey::FloatLiteral(_) => {
                if let TypeKey::Primitive(_) = expected {
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
                let name_str = &path.segments.last().unwrap().name;

                eprintln!("[DEBUG] Resolving TypePath: '{}' ...", name_str);
                if let Some(def_id) = self.resolve_path(path) {
                    let is_generic = self.ctx.generic_param_defs.contains(&def_id);
                    eprintln!(
                        "[DEBUG]   -> Found DefId: {:?}. Is GenericParam? {}",
                        def_id, is_generic
                    );

                    if self.ctx.generic_param_defs.contains(&def_id) {
                        return TypeKey::GenericParam(def_id);
                    }

                    let mut args = Vec::new();

                    if let Some(last_seg) = path.segments.last() {
                        if let Some(ast_args) = &last_seg.generic_args {
                            eprintln!(
                                "[DEBUG]   -> Parsing generic arguments for '{}'...",
                                name_str
                            );
                            for arg in ast_args {
                                args.push(self.resolve_ast_type(arg));
                            }
                        }
                    }

                    let key = TypeKey::Instantiated { def_id, args };

                    if key.is_concrete() {
                        if self.ctx.struct_fields.contains_key(&def_id) {
                            self.ctx.concrete_structs.insert(key.clone());
                        }
                    }
                    key
                } else {
                    eprintln!(
                        "[DEBUG]   -> FAILED to resolve path '{}' in current scopes!",
                        name_str
                    );
                    self.dump_current_scopes();
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

        for scope in self.scopes.iter().rev() {
            if let Some((id, _)) = scope.symbols.get(&first_seg.name) {
                current_def_id = Some(*id);
                break;
            }
        }

        if current_def_id.is_none() {
            return None;
        }

        for (i, segment) in path.segments.iter().enumerate().skip(1) {
            let parent_id = current_def_id.unwrap();

            if let Some(scope) = self.ctx.namespace_scopes.get(&parent_id) {
                if let Some((child_id, visibility)) = scope.symbols.get(&segment.name) {
                    if *visibility == Visibility::Private {
                        self.error(
                            format!("Symbol '{}' is private", segment.name),
                            segment.span,
                        );

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
        if name == "main" {
            return name.to_string();
        }

        if self.module_path.is_empty() {
            return name.to_string();
        }

        let prefix = self.module_path.join("_");
        format!("{}_{}", prefix, name)
    }

    /// 层级路径解析：查找符号并收集沿途的泛型参数
    fn resolve_path_hierarchically(&mut self, path: &Path) -> Option<PathResolutionResult> {
        let mut current_def_id = None;
        let mut all_generic_args = Vec::new();
        let mut struct_context = None;

        for (i, segment) in path.segments.iter().enumerate() {
            let found_id = if i == 0 {
                self.find_symbol_in_scopes(&segment.name)?
            } else {
                let parent = current_def_id.unwrap();
                self.find_symbol_in_namespace(parent, &segment.name)?
            };

            current_def_id = Some(found_id);

            let mut segment_args = Vec::new();
            if let Some(ast_args) = &segment.generic_args {
                for arg in ast_args {
                    segment_args.push(self.resolve_ast_type(arg));
                }
            }

            all_generic_args.extend(segment_args.clone());

            if self.ctx.def_generic_params.contains_key(&found_id) {
                if !segment_args.is_empty() {
                    struct_context = Some((found_id, segment_args));
                }
            }
        }

        Some(PathResolutionResult {
            def_id: current_def_id?,
            all_generic_args,
            struct_context,
        })
    }

    fn find_symbol_in_scopes(&mut self, name: &str) -> Option<DefId> {
        for scope in self.scopes.iter().rev() {
            if let Some((id, _)) = scope.symbols.get(name) {
                return Some(*id);
            }
        }
        None
    }

    fn find_symbol_in_namespace(&mut self, parent_id: DefId, name: &str) -> Option<DefId> {
        if let Some(scope) = self.ctx.namespace_scopes.get(&parent_id) {
            if let Some((id, vis)) = scope.symbols.get(name) {
                return Some(*id);
            }
        }
        None
    }

    /// 处理 Path 表达式的核心逻辑
    fn check_path_expr(&mut self, path: &Path, expr_id: NodeId) -> TypeKey {
        if let Some(res) = self.resolve_path_hierarchically(path) {
            let def_id = res.def_id;

            self.ctx.path_resolutions.insert(expr_id, def_id);

            if !res.all_generic_args.is_empty() {
                self.ctx
                    .node_generic_args
                    .insert(expr_id, res.all_generic_args.clone());
            }

            self.try_record_instantiation_with_args(def_id, &res.all_generic_args);
            self.verify_generic_args(def_id, &res.all_generic_args, path.span);

            if let Some(mut raw_type) = self.get_type_of_def(def_id) {
                if let Some((struct_id, struct_args)) = res.struct_context {
                    raw_type = self
                        .ctx
                        .substitute_generics(&raw_type, struct_id, &struct_args);
                }

                if let TypeKey::Instantiated { args, .. } = &mut raw_type {
                    if args.is_empty() && !res.all_generic_args.is_empty() {
                        *args = res.all_generic_args;
                    }
                }

                self.record_type(expr_id, raw_type.clone());
                return raw_type;
            } else {
                self.error("Symbol has no type", path.span);
            }
        } else {
            self.error(format!("Failed to resolve path '{:?}'", path), path.span);
        }
        TypeKey::Error
    }

    fn try_record_instantiation_with_args(&mut self, def_id: DefId, args: &[TypeKey]) {
        if let Some(param_ids) = self.ctx.def_generic_params.get(&def_id) {
            if param_ids.is_empty() {
                return;
            }

            if args.len() == param_ids.len() && args.iter().all(|a| a.is_concrete()) {
                self.ctx.concrete_functions.insert((def_id, args.to_vec()));
            }
        }
    }

    fn verify_generic_args(&mut self, def_id: DefId, args: &[TypeKey], span: Span) {
        let param_ids = match self.ctx.def_generic_params.get(&def_id) {
            Some(ids) => ids.clone(),
            None => return,
        };

        if param_ids.len() != args.len() {
            return;
        }

        for (param_id, arg_ty) in param_ids.iter().zip(args.iter()) {
            let constraints = self.ctx.generic_constraints.get(param_id).cloned();

            if let Some(cons) = constraints {
                for constraint_ty_key in cons {
                    // 适配新的 TypeKey::Instantiated 结构
                    if let TypeKey::Instantiated { def_id: cap_id, .. } = constraint_ty_key {
                        // 4. 现在 self 是自由身，可以调用 &mut 方法了
                        self.check_constraint(arg_ty, cap_id, span);
                    }
                }
            }
        }
    }

    /// 递归验证 Impl 的约束条件 (Where Clauses)
    fn verify_impl_obligations(
        &mut self,
        impl_id: DefId,
        mapping: &HashMap<DefId, TypeKey>,
        span: Span,
    ) -> bool {
        // 1. 获取该 Impl 块定义的泛型参数列表
        // 先 clone 出来，避免持有 self.ctx 的引用
        let param_ids = match self.ctx.def_generic_params.get(&impl_id) {
            Some(ids) => ids.clone(),
            None => return true, // 没有泛型参数，也就没有约束
        };

        // 2. 遍历每一个泛型参数，检查其约束
        for param_id in param_ids {
            // 获取约束列表 (e.g. Clone, Debug)
            let constraints = self
                .ctx
                .generic_constraints
                .get(&param_id)
                .cloned()
                .unwrap_or_default();

            // 获取该参数对应的具体类型 (e.g. i32)
            let concrete_ty = match mapping.get(&param_id) {
                Some(ty) => ty,
                None => continue,
            };

            // 3. 对每一个约束进行递归检查
            for constraint_key in constraints {
                if let TypeKey::Instantiated {
                    def_id: cap_id,
                    args: cap_args,
                } = constraint_key
                {
                    // 如果约束本身带有泛型 (e.g. T: Converter<U>)，需要替换 U
                    // 这里我们用 map_to_vec 将 mapping 转成 substitute 需要的 args
                    let args_vec = self.map_to_vec(impl_id, mapping);

                    // 实际上 cap_args 里的泛型参数也是属于 impl_id 的，所以可以直接替换
                    let mut concrete_cap_args = Vec::new();
                    for arg in cap_args {
                        concrete_cap_args
                            .push(self.ctx.substitute_generics(&arg, impl_id, &args_vec));
                    }

                    // 此时我们还不能 check_constraint 带参数的 Cap (目前的 check_constraint 签名只接受 cap_id)
                    // 但对于简单的 Cap (无泛型)，concrete_cap_args 为空
                    // 这里为了严谨，我们只检查 Cap ID。
                    // TODO: 升级 check_constraint 以支持带泛型的 Cap (如 check_trait_ref)

                    // 递归检查：e.g. 检查 i32 是否实现了 Clone
                    self.check_constraint(concrete_ty, cap_id, span);
                }
            }
        }
        true
    }

    /// 辅助：将泛型参数映射表 (Map) 转换为有序的实参列表 (Vec)
    /// 用于为 substitute_generics 准备 args 参数
    fn map_to_vec(&self, def_id: DefId, mapping: &HashMap<DefId, TypeKey>) -> Vec<TypeKey> {
        // 1. 获取该定义 (Struct/Impl/Fn) 的泛型参数 ID 列表 (有序)
        let param_ids = self
            .ctx
            .def_generic_params
            .get(&def_id)
            .cloned()
            .unwrap_or_default(); // 如果没有泛型，返回空列表

        // 2. 按顺序从 mapping 中提取类型
        let mut args = Vec::new();
        for id in param_ids {
            if let Some(ty) = mapping.get(&id) {
                args.push(ty.clone());
            } else {
                // 如果 mapping 里缺了某个参数，说明 Unify 不完整
                // 在这里可以报错，或者暂时填入 Error 类型防止 Panic
                // 正常情况下只要 Unify 成功，这里一定有值
                args.push(TypeKey::Error);
            }
        }
        args
    }

    /// 尝试将具体类型与模式类型进行匹配，并提取泛型参数的映射
    /// concrete: 具体类型 (e.g. List<i32>)
    /// pattern: 模式类型 (e.g. List<T>)
    /// mapping: 输出参数，存储 T -> i32 的映射
    /// 返回值: 是否匹配成功
    fn unify_types(
        &self,
        concrete: &TypeKey,
        pattern: &TypeKey,
        mapping: &mut HashMap<DefId, TypeKey>,
    ) -> bool {
        match (concrete, pattern) {
            // 1. 如果 pattern 是泛型参数 T
            (_, TypeKey::GenericParam(param_id)) => {
                if let Some(existing) = mapping.get(param_id) {
                    // 如果 T 之前已经映射过，必须与现在的类型一致
                    // e.g. pattern: Pair<T, T>, concrete: Pair<i32, f64> -> 失败
                    return existing == concrete;
                } else {
                    // 记录映射 T -> concrete
                    mapping.insert(*param_id, concrete.clone());
                    return true;
                }
            }

            // 2. 基础类型：必须完全相等
            (TypeKey::Primitive(p1), TypeKey::Primitive(p2)) => p1 == p2,

            // 3. 指针/数组：递归匹配
            (TypeKey::Pointer(inner1, mut1), TypeKey::Pointer(inner2, mut2)) => {
                mut1 == mut2 && self.unify_types(inner1, inner2, mapping)
            }
            (TypeKey::Array(inner1, size1), TypeKey::Array(inner2, size2)) => {
                size1 == size2 && self.unify_types(inner1, inner2, mapping)
            }

            // 4. 实例化类型 (Struct/Enum): List<i32> vs List<T>
            (
                TypeKey::Instantiated {
                    def_id: id1,
                    args: args1,
                },
                TypeKey::Instantiated {
                    def_id: id2,
                    args: args2,
                },
            ) => {
                if id1 != id2 || args1.len() != args2.len() {
                    return false;
                }
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    if !self.unify_types(a1, a2, mapping) {
                        return false;
                    }
                }
                true
            }

            // 函数类型匹配 (递归)
            (
                TypeKey::Function {
                    params: p1,
                    ret: r1,
                    ..
                },
                TypeKey::Function {
                    params: p2,
                    ret: r2,
                    ..
                },
            ) => {
                if p1.len() != p2.len() {
                    return false;
                }
                for (a, b) in p1.iter().zip(p2.iter()) {
                    if !self.unify_types(a, b, mapping) {
                        return false;
                    }
                }
                match (r1, r2) {
                    (Some(x), Some(y)) => self.unify_types(x, y, mapping),
                    (None, None) => true,
                    _ => false,
                }
            }

            // 其他情况不匹配
            _ => false,
        }
    }

    /// 尝试计算编译时常量表达式
    /// 如果计算成功，返回 Some(u64)
    /// 如果包含无法在编译期确定的内容（如函数调用、变量），返回 None
    fn eval_constant_expr(&mut self, expr: &Expression) -> Option<u64> {
        match &expr.kind {
            ExpressionKind::Literal(Literal::Integer(val)) => Some(*val),

            ExpressionKind::Path(path) => {
                let def_id = self.ctx.path_resolutions.get(&path.id)?;

                self.ctx.constants.get(def_id).cloned()
            }

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

                    BinaryOperator::ShiftLeft => Some(l << r),
                    BinaryOperator::ShiftRight => Some(l >> r),
                    BinaryOperator::BitwiseAnd => Some(l & r),
                    BinaryOperator::BitwiseOr => Some(l | r),
                    BinaryOperator::BitwiseXor => Some(l ^ r),

                    _ => None,
                }
            }

            ExpressionKind::Unary {
                op: UnaryOperator::Negate,
                operand,
            } => {
                let val = self.eval_constant_expr(operand)?;

                Some((-(val as i64)) as u64)
            }

            ExpressionKind::SizeOf(target_type) => {
                let key = if let Some(k) = self.ctx.types.get(&target_type.id) {
                    k.clone()
                } else {
                    self.resolve_ast_type(target_type)
                };

                self.get_type_static_size(&key)
            }

            ExpressionKind::Cast { expr: src_expr, .. } => {
                let val = self.eval_constant_expr(src_expr)?;

                if let Some(target_key) = self.ctx.types.get(&expr.id) {
                    match target_key {
                        TypeKey::Primitive(p) => match p {
                            PrimitiveType::U8 | PrimitiveType::I8 => Some(val & 0xFF),
                            PrimitiveType::U16 | PrimitiveType::I16 => Some(val & 0xFFFF),
                            PrimitiveType::U32 | PrimitiveType::I32 => Some(val & 0xFFFFFFFF),

                            _ => Some(val),
                        },

                        TypeKey::Pointer(..) => Some(val),

                        _ => Some(val),
                    }
                } else {
                    Some(val)
                }
            }

            ExpressionKind::Literal(Literal::Boolean(b)) => Some(if *b { 1 } else { 0 }),

            ExpressionKind::AlignOf(target_type) => {
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
        self.ctx.get_type_layout(key).map(|l| l.size)
    }

    /// 检查类型转换是否合法
    /// 规则参考 Rust/C:
    /// 1. 整数 <-> 整数 (包含 Bool, Char)
    /// 2. 整数 <-> 浮点
    /// 3. 浮点 <-> 浮点
    /// 4. 指针 <-> 整数 (size_t)
    /// 5. 指针 <-> 指针
    /// 6. 数组 -> 指针 (Array Decay)
    /// 7. Enum -> 整数
    fn validate_cast(&self, src: &TypeKey, target: &TypeKey) -> bool {
        match (src, target) {
            (t1, t2) if t1 == t2 => true,

            (TypeKey::Primitive(p1), TypeKey::Primitive(p2)) => {
                self.is_numeric_or_bool_char(p1) && self.is_numeric_or_bool_char(p2)
            }

            (TypeKey::IntegerLiteral(_), TypeKey::Primitive(p)) => self.is_numeric_or_bool_char(p),
            (TypeKey::FloatLiteral(_), TypeKey::Primitive(p)) => self.is_numeric_or_bool_char(p),

            (TypeKey::Pointer(..), TypeKey::Primitive(p)) => self.is_integer_type(p),
            (TypeKey::Primitive(p), TypeKey::Pointer(..)) => self.is_integer_type(p),

            (TypeKey::IntegerLiteral(_), TypeKey::Pointer(..)) => true,

            (TypeKey::Pointer(..), TypeKey::Pointer(..)) => true,

            (TypeKey::Array(..), TypeKey::Pointer(..)) => true,

            (TypeKey::Instantiated { def_id, .. }, TypeKey::Primitive(p)) => {
                if self.ctx.def_kind.get(def_id) == Some(&DefKind::Enum) {
                    self.is_integer_type(p)
                } else {
                    false
                }
            }

            _ => false,
        }
    }

    fn get_type_static_align(&self, key: &TypeKey) -> Option<u64> {
        self.ctx.get_type_layout(key).map(|l| l.align)
    }

    fn is_numeric_or_bool_char(&self, p: &PrimitiveType) -> bool {
        self.is_numeric_type(p) || matches!(p, PrimitiveType::Bool)
    }

    fn is_generic_param(&self, id: DefId) -> bool {
        self.ctx.generic_param_defs.contains(&id)
    }

    fn dump_current_scopes(&self) {
        eprintln!("[DEBUG] --- Current Scopes Dump ---");
        for (i, scope) in self.scopes.iter().enumerate().rev() {
            eprintln!("  Scope depth {}: Kind = {:?}", i, scope.kind);
            for (name, (id, _)) in &scope.symbols {
                eprintln!("    Symbol '{}' -> ID {:?}", name, id);
            }
        }
        eprintln!("[DEBUG] ---------------------------");
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

struct PathResolutionResult {
    def_id: DefId,

    all_generic_args: Vec<TypeKey>,

    struct_context: Option<(DefId, Vec<TypeKey>)>,
}
