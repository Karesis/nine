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

use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::types::{BasicType, BasicTypeEnum, FunctionType, StructType};
use inkwell::values::{
    BasicValue, BasicValueEnum, FloatValue, FunctionValue, IntValue, PointerValue,
};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use std::collections::HashMap;

use crate::analyzer::{AnalysisContext, DefId, TypeKey, MethodInfo};
use crate::ast::*;

macro_rules! trace {
    ($($arg:tt)*) => {
        eprintln!("[Codegen] {}", format!($($arg)*));
    };
}

/// 代码生成器结构体
pub struct CodeGen<'a, 'ctx> {
    pub context: &'ctx Context,
    pub module: &'a Module<'ctx>,
    pub builder: &'a Builder<'ctx>,

    /// 语义分析的结果 (只读)
    pub analysis: &'a AnalysisContext,

    // 局部变量表：DefId -> PointerValue
    pub variables: HashMap<DefId, PointerValue<'ctx>>,

    // 全局变量 (Global Variables)
    pub globals: HashMap<DefId, PointerValue<'ctx>>,

    // 全局函数表：Mangled Name -> LLVM Function
    pub functions: HashMap<String, FunctionValue<'ctx>>,

    // 结构体类型表：Mangled Name -> LLVM StructType
    pub struct_types: HashMap<String, StructType<'ctx>>,

    // 结构体字段索引：Mangled Name -> (Field Name -> Index)
    pub struct_field_indices: HashMap<String, HashMap<String, u32>>,

    pub loop_stack: Vec<(BasicBlock<'ctx>, BasicBlock<'ctx>)>,
    current_fn: Option<FunctionValue<'ctx>>,

    // 当前正在编译的泛型上下文
    // 当编译泛型函数实例时，需要知道 T 对应什么，以便替换函数体内的局部变量类型
    // (DefId, Args)
    generic_context: Option<(DefId, Vec<TypeKey>)>,
    // 函数定义索引
    // 解决了 "去哪找函数体" 的问题，避免了 find_function_definition 的递归查找
    // 生命周期 'a 绑定到 Program 上
    pub function_index: HashMap<DefId, &'a FunctionDefinition>,
}

impl<'a, 'ctx> CodeGen<'a, 'ctx> {
    pub fn new(
        context: &'ctx Context,
        module: &'a Module<'ctx>,
        builder: &'a Builder<'ctx>,
        analysis: &'a AnalysisContext,
        program: &'a Program, // 【修改】new 的时候传入 Program，用于构建索引
    ) -> Self {
        // 1. 构建索引
        let mut function_index = HashMap::new();
        Self::build_function_index(&program.items, &mut function_index);

        Self {
            context,
            module,
            builder,
            analysis,
            variables: HashMap::new(),
            globals: HashMap::new(),
            functions: HashMap::new(),
            struct_types: HashMap::new(),
            struct_field_indices: HashMap::new(),
            loop_stack: Vec::new(),
            current_fn: None,
            function_index, // 存入
            generic_context: None,
        }
    }

    // 递归遍历 AST，把所有 FunctionDefinition 的引用存入 Map
    fn build_function_index(
        items: &'a [Item],
        index: &mut HashMap<DefId, &'a FunctionDefinition>,
    ) {
        for item in items {
            match &item.kind {
                ItemKind::FunctionDecl(f) => {
                    index.insert(f.id, f);
                }
                ItemKind::StructDecl(s) => {
                    for m in &s.static_methods {
                        index.insert(m.id, m);
                    }
                }
                ItemKind::EnumDecl(e) => {
                    for m in &e.static_methods {
                        index.insert(m.id, m);
                    }
                }
                ItemKind::Implementation { methods, .. } => {
                    for m in methods {
                        index.insert(m.id, m);
                    }
                }
                ItemKind::ModuleDecl { items: Some(subs), .. } => {
                    Self::build_function_index(subs, index);
                }
                _ => {}
            }
        }
    }

    // ========================================================================
    // 1. 类型转换系统 (TypeKey -> LLVM Type)
    // ========================================================================

    pub fn compile_type(&self, key: &TypeKey) -> Option<BasicTypeEnum<'ctx>> {
        match key {
            TypeKey::Primitive(prim) => Some(self.compile_primitive_type(prim)),

            // 【修改】处理实例化类型 (Struct<i32>)
            TypeKey::Instantiated { .. } => {
                // 1. 获取修饰名
                let mangled_name = self.analysis.get_mangling_type_name(key);
                
                // 2. 查表 (Pass 1 应该已经生成了 Opaque 类型)
                if let Some(st) = self.struct_types.get(&mangled_name) {
                    Some(st.as_basic_type_enum())
                } else {
                    panic!("ICE: Struct type '{}' not pre-generated.", mangled_name);
                }
            }

            // 指针在 LLVM IR (Opaque Pointers) 中统一为 ptr
            TypeKey::Pointer(_, _) => Some(
                self.context
                    .ptr_type(AddressSpace::default())
                    .as_basic_type_enum(),
            ),

            // 数组 [N x T]
            TypeKey::Array(inner, size) => {
                let inner_ty = self.compile_type(inner)?;
                Some(inner_ty.array_type(*size as u32).as_basic_type_enum())
            }

            TypeKey::Function { .. } => Some(
                self.context
                    .ptr_type(AddressSpace::default())
                    .as_basic_type_enum(),
            ),

            // 这里的 Panic 是为了捕获 Analyzer 的 Bug。
            // 所有的字面量在进入 Codegen 之前，都必须在 Analyzer 阶段被 coerce_literal_type 固化为 Primitive。
            TypeKey::IntegerLiteral(v) => {
                panic!(
                    "ICE (Internal Compiler Error): Analyzer failed to resolve IntegerLiteral({}) to a concrete type. This is a bug in the compiler.",
                    v
                );
            }
            TypeKey::FloatLiteral(bits) => {
                let v = f64::from_bits(*bits);
                panic!(
                    "ICE (Internal Compiler Error): Analyzer failed to resolve FloatLiteral({}) to a concrete type. This is a bug in the compiler.",
                    v
                );
            }

            // 【新增】处理泛型参数 T
            // 在 Codegen 阶段，T 应该已经被 get_resolved_type 替换掉了。
            // 如果还能遇到，说明逻辑有漏。
            TypeKey::GenericParam(id) => {
                panic!("ICE: GenericParam({:?}) reached Codegen layer!", id);
            }

            // Error 类型也不应该传到 Codegen，Analyzer 应该早就拦截并返回 Err 了
            TypeKey::Error => panic!(
                "ICE: TypeKey::Error reached Codegen. Compilation should have failed in Analyzer phase."
            ),
        }
    }

    fn compile_primitive_type(&self, prim: &PrimitiveType) -> BasicTypeEnum<'ctx> {
        match prim {
            PrimitiveType::I8 | PrimitiveType::U8 => self.context.i8_type().as_basic_type_enum(),
            PrimitiveType::I16 | PrimitiveType::U16 => self.context.i16_type().as_basic_type_enum(),
            PrimitiveType::I32 | PrimitiveType::U32 => self.context.i32_type().as_basic_type_enum(),
            PrimitiveType::I64 | PrimitiveType::U64 => self.context.i64_type().as_basic_type_enum(),
            PrimitiveType::ISize | PrimitiveType::USize => {
                // 根据 Target 的指针宽度决定生成 i32 还是 i64
                if self.analysis.target.ptr_byte_width == 4 {
                    self.context.i32_type().as_basic_type_enum()
                } else {
                    self.context.i64_type().as_basic_type_enum()
                }
            }
            PrimitiveType::F32 => self.context.f32_type().as_basic_type_enum(),
            PrimitiveType::F64 => self.context.f64_type().as_basic_type_enum(),
            PrimitiveType::Bool => self.context.bool_type().as_basic_type_enum(),
            // Unit 暂时用空结构体占位，具体根据上下文可能是 void
            //? 更详细的处理？
            PrimitiveType::Unit => self.context.struct_type(&[], false).as_basic_type_enum(),
        }
    }

    // ========================================================================
    // 2. 主入口
    // ========================================================================

    pub fn compile_program(&mut self, program: &Program) {
        // Step 1: 声明所有结构体 (Opaque)
        // 这一步是为了让递归结构体（如链表）能引用自己
        self.declare_concrete_structs();

        // Step 2: 填充结构体 Body
        // 这一步根据 Analyzer 算好的具体类型，填入字段
        self.fill_concrete_struct_bodies();

        // Step 3: 函数声明 (非泛型 + 泛型实例)
        // ... (接下来的逻辑，稍后我们处理函数时会写) ...
        
        // 3.1 声明普通函数
        for &def_id in &self.analysis.non_generic_functions {
            if let Some(func_def) = self.function_index.get(&def_id) {
                self.declare_function(func_def);
            }
        }

        // 3.2 声明泛型实例
        let generic_funcs: Vec<_> = self.analysis.concrete_functions.iter().cloned().collect();
        for (def_id, args) in &generic_funcs {
            self.declare_monomorphized_function(*def_id, args);
        }

        // Step 4: 全局变量
        self.compile_globals(&program.items);

        // Step 5: 函数体编译
        // 5.1 编译普通函数
        // 【修复】先 collect 到一个临时 Vec，断开对 self.analysis 的借用
        let non_generic_ids: Vec<_> = self.analysis.non_generic_functions.iter().cloned().collect();
        
        for def_id in non_generic_ids {
            // 注意：self.function_index.get 返回的是 &&FunctionDefinition
            // 我们需要 copier() 或者 * 解引用拿到 &'a FunctionDefinition
            // 这样就不会锁住 self.function_index 了
            if let Some(&func_def) = self.function_index.get(&def_id) {
                self.generic_context = None;
                
                // 现在可以放心调用 mut self 的方法了
                if let Err(e) = self.compile_function(func_def) {
                    panic!("Failed to compile function '{}': {}", func_def.name.name, e);
                }
            }
        }

        // 5.2 编译泛型实例
        // 【修复】同理，先 collect
        let generic_funcs: Vec<_> = self.analysis.concrete_functions.iter().cloned().collect();
        
        for (def_id, args) in generic_funcs {
            // 这里的 panic 也是必要的，generic_funcs 编译失败也应该炸
            self.compile_monomorphized_function(def_id, &args); 
        }
    }

    // 【Step 1 实现】只负责创建空壳
    fn declare_concrete_structs(&mut self) {
        // 遍历 Analyzer 计算出的所有具体结构体 (e.g. "Box_i32", "List_f64")
        for (mangled_name, _) in &self.analysis.instantiated_structs {
            let st = self.context.opaque_struct_type(mangled_name);
            // 存入表：Key 是修饰名 (String)
            self.struct_types.insert(mangled_name.clone(), st);
        }
    }

    // 【Step 2 实现】负责填肉
    fn fill_concrete_struct_bodies(&mut self) {
        // 克隆一份数据以避免借用冲突
        let structs: Vec<_> = self.analysis.instantiated_structs.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        for (mangled_name, fields) in structs {
            let st = *self.struct_types.get(&mangled_name).expect("Struct type missing");

            let mut llvm_field_types = Vec::new();
            let mut indices = HashMap::new();

            // 这里的 fields 已经是 [(字段名, 具体类型), ...]
            // 不需要再去查 Analyzer 的 struct_fields 表了，Analyzer 已经把饭喂到嘴边了
            for (i, (field_name, field_type_key)) in fields.iter().enumerate() {
                let llvm_ty = self.compile_type(field_type_key).expect("Field type compile failed");
                
                llvm_field_types.push(llvm_ty);
                indices.insert(field_name.clone(), i as u32);
            }

            st.set_body(&llvm_field_types, false); // packed = false
            self.struct_field_indices.insert(mangled_name, indices);
        }
    }

    fn compile_struct_body(&mut self, mangled_name: &str, fields: &[(String, TypeKey)]) {
        let st = *self.struct_types.get(mangled_name).unwrap();
        
        let mut llvm_fields = Vec::new();
        let mut indices = HashMap::new();

        for (i, (field_name, field_ty)) in fields.iter().enumerate() {
            // 这里的 field_ty 已经是 Analyzer 替换好的具体类型了 (i32)，直接编译！
            let llvm_ty = self.compile_type(field_ty).expect("Field type compile failed");
            llvm_fields.push(llvm_ty);
            indices.insert(field_name.clone(), i as u32);
        }

        st.set_body(&llvm_fields, false);
        self.struct_field_indices.insert(mangled_name.to_string(), indices);
    }

    fn declare_monomorphized_function(&mut self, def_id: DefId, args: &[TypeKey]) {
        let fn_mangled_name = self.analysis.get_mangled_function_name(def_id, args);

        // 如果已经声明过，跳过
        if self.functions.contains_key(&fn_mangled_name) {
            return;
        }

        // 2. 获取原始函数类型 (包含 T)
        let raw_fn_type = self.get_resolved_type(def_id);
        
        // 3. 【关键】执行替换 (T -> i32)
        // 使用刚刚下沉到 analysis 的方法
        let actual_fn_type_key = self.analysis.substitute_generics(&raw_fn_type, def_id, args);

        // 4. 编译为 LLVM Function Type
        let llvm_fn_type = self.compile_function_type_signature(&actual_fn_type_key).unwrap();

        // 5. 添加到 Module
        let val = self.module.add_function(&fn_mangled_name, llvm_fn_type, None);
        self.functions.insert(fn_mangled_name, val);
    }

    fn compile_monomorphized_function(&mut self, def_id: DefId, args: &[TypeKey]) {
        // 1. 直接查索引拿到 AST
        let func_def = *self.function_index.get(&def_id).expect("Generic function def missing in index");

       let fn_mangled_name = self.analysis.get_mangled_function_name(def_id, args);

        // 3. 获取 FunctionValue
        let function = *self.functions.get(&fn_mangled_name).expect("Function proto not declared");
        
        // 4. 设置上下文
        self.generic_context = Some((def_id, args.to_vec()));
        self.current_fn = Some(function);

        // 5. 创建 Entry Block
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        // 6. 编译参数 (Alloca + Store)
        self.variables.clear();
        for (i, param) in func_def.params.iter().enumerate() {
            let arg_val = function.get_nth_param(i as u32).unwrap();
            arg_val.set_name(&param.name.name);

            // 使用 get_resolved_type 自动替换 T -> i32
            let param_ty_key = self.get_resolved_type(param.id);
            let llvm_ty = self.compile_type(&param_ty_key).unwrap();

            let alloca = self.builder.build_alloca(llvm_ty, &param.name.name).unwrap();
            self.builder.build_store(alloca, arg_val).ok();
            self.variables.insert(param.id, alloca);
        }

        // 7. 编译 Body
        if let Some(body) = &func_def.body {
            self.compile_block(body).ok();
        }

        // 8. 补全 Void Return
        if !self.block_terminated(func_def.body.as_ref().unwrap()) {
             if func_def.return_type.is_none() {
                 self.builder.build_return(None).ok();
             }
        }

        // 9. 清理
        self.generic_context = None;
        self.current_fn = None;
    }

    fn compile_items(&mut self, items: &[Item]) {
        for item in items {
            if let ItemKind::GlobalVariable(def) = &item.kind {
                self.compile_global_variable(item.id, def);
            }
        }

        for item in items {
            match &item.kind {
                ItemKind::FunctionDecl(func) => {
                    self.compile_function(func).ok();
                }
                ItemKind::Implementation { methods, .. } => {
                    for m in methods {
                        self.compile_function(m).ok();
                    }
                }
                ItemKind::ModuleDecl {
                    items: Some(subs), ..
                } => self.compile_items(subs),
                ItemKind::StructDecl(def) => {
                    for m in &def.static_methods {
                        if let Err(e) = self.compile_function(m) {
                            println!(
                                "Codegen Error in static method {}::{}: {}",
                                def.name.name, m.name.name, e
                            );
                        }
                    }
                }
                ItemKind::EnumDecl(def) => {
                    for m in &def.static_methods {
                        if let Err(e) = self.compile_function(m) {
                            println!(
                                "Codegen Error in static method {}::{}: {}",
                                def.name.name, m.name.name, e
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // 递归遍历 AST，只编译 GlobalVariable
    fn compile_globals(&mut self, items: &[Item]) {
        for item in items {
            match &item.kind {
                // 1. 遇到全局变量 -> 编译
                ItemKind::GlobalVariable(def) => {
                    self.compile_global_variable(item.id, def);
                }
                
                // 2. 遇到模块 -> 递归进入
                ItemKind::ModuleDecl { items: Some(subs), .. } => {
                    self.compile_globals(subs);
                }
                
                // 其他类型 (函数、结构体等) 跳过，因为它们在 compile_program 的其他步骤处理了
                _ => {}
            }
        }
    }

    // Key 变为 mangled_name (String)
    fn get_field_index(&self, mangled_name: &str, field_name: &str) -> Option<u32> {
        self.struct_field_indices
            .get(mangled_name)
            .and_then(|indices| indices.get(field_name).cloned())
    }

    // ========================================================================
    // 4. 函数编译
    // ========================================================================

    pub fn register_function_prototypes(&mut self, items: &[Item]) {
        for item in items {
            match &item.kind {
                ItemKind::FunctionDecl(func) => self.declare_function(func),
                ItemKind::Implementation { methods, .. } => {
                    for m in methods {
                        self.declare_function(m);
                    }
                }
                ItemKind::StructDecl(def) => {
                    for m in &def.static_methods {
                        self.declare_function(m);
                    }
                }
                ItemKind::EnumDecl(def) => {
                    for m in &def.static_methods {
                        self.declare_function(m);
                    }
                }
                ItemKind::ModuleDecl {
                    items: Some(subs), ..
                } => self.register_function_prototypes(subs),
                _ => {}
            }
        }
    }

    // 声明非泛型函数
    fn declare_function(&mut self, func: &FunctionDefinition) {
        let param_types = func
            .params
            .iter()
            .map(|p| {
                // 使用 get_resolved_type 保持统一，虽然对于非泛型函数 types.get 也行
                let key = self.get_resolved_type(p.id);
                self.compile_type(&key).unwrap().into()
            })
            .collect::<Vec<_>>();

        let ret_key = self.get_resolved_type(func.id);

        // 解析 FunctionType
        let fn_type = if let TypeKey::Function { ret: Some(r), .. } = ret_key {
            if let TypeKey::Primitive(PrimitiveType::Unit) = *r {
                self.context
                    .void_type()
                    .fn_type(&param_types, func.is_variadic)
            } else {
                let ret_ty = self.compile_type(&r).unwrap();
                ret_ty.fn_type(&param_types, func.is_variadic)
            }
        } else {
            self.context
                .void_type()
                .fn_type(&param_types, func.is_variadic)
        };

        // 【修正】获取 Mangled Name (String)
        let fn_name = self
            .analysis
            .mangled_names
            .get(&func.id)
            .cloned()
            .unwrap_or(func.name.name.clone());

        // 这里的 fn_name 已经是 String 了
        let val = self.module.add_function(&fn_name, fn_type, None);
        
        // 【修正】使用 String 作为 Key
        self.functions.insert(fn_name, val);
    }

    pub fn compile_function(&mut self, func: &FunctionDefinition) -> Result<(), String> {
        trace!("Compiling function: {}", func.name.name);
        if func.body.is_none() {
            return Ok(());
        }
        let body_ref = func.body.as_ref().unwrap();

        // 1. 【修正】通过 Mangled Name 查找函数
        // 因为这是 compile_function（只用于非泛型），所以泛型参数列表为空 &[]
        let fn_mangled_name = self.analysis.get_mangled_function_name(func.id, &[]);
        
        let function = *self.functions.get(&fn_mangled_name)
            .ok_or_else(|| format!("Function proto '{}' missing", fn_mangled_name))?;
            
        self.current_fn = Some(function);

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        // 2. 处理参数
        // 建议先清空 variables，防止上一此编译的残留（虽然这里 map 是覆盖，但清空更安全）
        self.variables.clear(); 
        
        for (i, param) in func.params.iter().enumerate() {
            let arg_val = function.get_nth_param(i as u32).unwrap();
            arg_val.set_name(&param.name.name);

            // 【修正】统一使用 get_resolved_type
            // 尽管对于非泛型函数，types.get 也能拿到正确结果，但保持一致性更好
            let param_ty_key = self.get_resolved_type(param.id);
            let llvm_ty = self.compile_type(&param_ty_key).unwrap();

            let alloca = self
                .builder
                .build_alloca(llvm_ty, &param.name.name)
                .map_err(|_| "Alloca failed")?;
            self.builder.build_store(alloca, arg_val).ok();

            self.variables.insert(param.id, alloca);
        }

        // 3. 编译函数体
        self.compile_block(body_ref)?;

        // 4. 处理 Void 返回
        if !self.block_terminated(body_ref) {
            if func.return_type.is_none() {
                self.builder.build_return(None).ok();
            } else {
                // 如果 Analyzer 做了控制流分析，这里理论上不可达。
                // 但为了防止 Codegen Crash，给个友好的错误比 unreachable! 更好
                return Err(format!("Function '{}' missing return statement", func.name.name));
            }
        }
        
        // 5. 清理状态
        self.current_fn = None;

        Ok(())
    }

    // ========================================================================
    // 5. 语句编译
    // ========================================================================

    fn get_current_function(&self) -> FunctionValue<'ctx> {
        self.current_fn.expect("Outside fn")
    }

    fn block_terminated(&self, block: &Block) -> bool {
        if let Some(last) = block.stmts.last() {
            matches!(
                last.kind,
                StatementKind::Return(_) | StatementKind::Break | StatementKind::Continue
            )
        } else {
            false
        }
    }

    pub fn compile_stmt(&mut self, stmt: &Statement) -> Result<(), String> {
        trace!("  Compiling stmt: {:?}", stmt.kind);
        match &stmt.kind {
            StatementKind::VariableDeclaration {
                name,
                type_annotation: _,
                initializer,
                ..
            } => {
                // 1. 获取准确类型 (Analyzer 已推导)
                //？ 支持更多的推导？
                //？ 新的语法设计？（auto?)
                let type_key = self.get_resolved_type(stmt.id);
                let llvm_ty = self.compile_type(&type_key).unwrap();

                // 2. 栈分配
                let ptr = self
                    .builder
                    .build_alloca(llvm_ty, &name.name)
                    .map_err(|_| "Alloca failed")?;
                self.variables.insert(stmt.id, ptr);

                // 3. 初始化
                if let Some(init) = initializer {
                    let val = self.compile_expr(init)?;
                    self.builder
                        .build_store(ptr, val)
                        .map_err(|_| "Store failed")?;
                }
                Ok(())
            }

            StatementKind::Assignment { lhs, rhs } => {
                let val = self.compile_expr(rhs)?;
                let ptr = self.compile_expr_ptr(lhs)?;
                self.builder
                    .build_store(ptr, val)
                    .map_err(|_| "Assign failed")?;
                Ok(())
            }

            StatementKind::ExpressionStatement(expr) => {
                self.compile_expr(expr)?;
                Ok(())
            }

            StatementKind::Block(block) => self.compile_block(block),

            StatementKind::If {
                condition,
                then_block,
                else_branch,
            } => {
                let parent = self.get_current_function();

                // 1. 创建基本块
                let then_bb = self.context.append_basic_block(parent, "if_then");
                let else_bb = self.context.append_basic_block(parent, "if_else");
                let merge_bb = self.context.append_basic_block(parent, "if_merge");

                // 2. 编译条件跳转
                let cond_val = self.compile_expr(condition)?.into_int_value();

                // 如果有 else 分支，跳 else_bb；否则跳 merge_bb
                let actual_else = if else_branch.is_some() {
                    else_bb
                } else {
                    merge_bb
                };

                self.builder
                    .build_conditional_branch(cond_val, then_bb, actual_else)
                    .ok();

                // 3. 编译 Then 分支
                self.builder.position_at_end(then_bb);
                self.compile_block(then_block)?;
                // 检查当前 block (可能是 then_block 里的最后一个 block) 是否已经结束
                // ompile_block 可能会产生新的 basic block，所以我们要检查 builder 当前所在的 block
                if self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_none()
                {
                    self.builder.build_unconditional_branch(merge_bb).ok();
                }

                // 4. 编译 Else 分支
                if let Some(else_stmt) = else_branch {
                    self.builder.position_at_end(else_bb);
                    self.compile_stmt(else_stmt)?;

                    // 检查 Else 分支编译完后，当前所在的 block 是否有终结指令
                    if self
                        .builder
                        .get_insert_block()
                        .unwrap()
                        .get_terminator()
                        .is_none()
                    {
                        self.builder.build_unconditional_branch(merge_bb).ok();
                    }
                }

                // 5. 移动到 Merge 继续
                self.builder.position_at_end(merge_bb);
                Ok(())
            }

            StatementKind::While {
                condition,
                init_statement,
                body,
            } => {
                let parent = self.get_current_function();
                if let Some(init) = init_statement {
                    self.compile_stmt(init)?;
                }

                let cond_bb = self.context.append_basic_block(parent, "cond");
                let body_bb = self.context.append_basic_block(parent, "body");
                let end_bb = self.context.append_basic_block(parent, "end");

                self.loop_stack.push((end_bb, cond_bb));

                self.builder.build_unconditional_branch(cond_bb).ok();

                // Cond
                self.builder.position_at_end(cond_bb);
                let c = self.compile_expr(condition)?.into_int_value();
                self.builder
                    .build_conditional_branch(c, body_bb, end_bb)
                    .ok();

                // Body
                self.builder.position_at_end(body_bb);
                self.compile_block(body)?;

                self.loop_stack.pop();

                if !self.block_terminated(body) {
                    self.builder.build_unconditional_branch(cond_bb).ok();
                }

                self.builder.position_at_end(end_bb);
                Ok(())
            }

            StatementKind::Return(opt) => {
                if let Some(e) = opt {
                    let v = self.compile_expr(e)?;
                    self.builder.build_return(Some(&v)).ok();
                } else {
                    self.builder.build_return(None).ok();
                }
                Ok(())
            }

            StatementKind::Break => {
                let (target, _) = self.loop_stack.last().ok_or("Break error")?;
                self.builder.build_unconditional_branch(*target).ok();
                Ok(())
            }
            StatementKind::Continue => {
                let (_, target) = self.loop_stack.last().ok_or("Continue error")?;
                self.builder.build_unconditional_branch(*target).ok();
                Ok(())
            }

            StatementKind::Switch {
                target,
                cases,
                default_case,
            } => {
                let parent = self.get_current_function();

                // 1. 编译 switch 的目标值
                let target_val = self.compile_expr(target)?;
                if !target_val.is_int_value() {
                    return Err("Switch on non-integer types not implemented yet".into());
                }
                let target_int = target_val.into_int_value();

                // 2. 准备 Basic Blocks (Merge & Default)
                let merge_bb = self.context.append_basic_block(parent, "switch_merge");
                let default_bb = self.context.append_basic_block(parent, "switch_default");

                // --- 预创建 Blocks 并收集 Cases 列表 ---
                let mut collected_cases = Vec::new();

                // Vec<(BasicBlock, &Block)>
                let mut case_blocks_to_compile = Vec::new();

                for case in cases {
                    // 为每个 AST Case 创建一个 BasicBlock
                    let case_bb = self.context.append_basic_block(parent, "switch_case");

                    // 记录下来，稍后编译体
                    case_blocks_to_compile.push((case_bb, &case.body));

                    // 遍历该 Case 的所有Pattern，将它们都指向这个 case_bb
                    for pattern in &case.patterns {
                        // 编译 Pattern 表达式 (必须是常量整数)
                        let pattern_val = self.compile_expr(pattern)?;
                        if !pattern_val.is_int_value() {
                            return Err("Switch case pattern must be integer".into());
                        }

                        // 收集到列表中: (IntValue, BasicBlock)
                        collected_cases.push((pattern_val.into_int_value(), case_bb));
                    }
                }

                // --- 一次性生成 Switch 指令 ---
                self.builder
                    .build_switch(target_int, default_bb, &collected_cases)
                    .map_err(|_| "Build switch failed")?;

                // --- 填充各个 Case Block 的代码 ---
                for (bb, body) in case_blocks_to_compile {
                    self.builder.position_at_end(bb);
                    self.compile_block(body)?;

                    // 自动 Break 处理
                    if !self.block_terminated(body) {
                        self.builder.build_unconditional_branch(merge_bb).ok();
                    }
                }

                // --- 填充 Default Block ---
                self.builder.position_at_end(default_bb);
                if let Some(block) = default_case {
                    self.compile_block(block)?;
                    if !self.block_terminated(block) {
                        self.builder.build_unconditional_branch(merge_bb).ok();
                    }
                } else {
                    self.builder.build_unconditional_branch(merge_bb).ok();
                }

                // ---结束 ---
                self.builder.position_at_end(merge_bb);
                Ok(())
            }

            _ => Ok(()),
        }
    }

    fn compile_block(&mut self, block: &Block) -> Result<(), String> {
        for s in &block.stmts {
            self.compile_stmt(s)?;
        }
        Ok(())
    }

    // ========================================================================
    // 6. 表达式编译
    // ========================================================================

    pub fn compile_expr(&mut self, expr: &Expression) -> Result<BasicValueEnum<'ctx>, String> {
        trace!("    Compiling expr: {:?}", expr.kind);
        match &expr.kind {
            ExpressionKind::Literal(lit) => {
                let type_key = self.get_resolved_type(expr.id);
                self.compile_literal(lit, &type_key)
            }

            ExpressionKind::Binary { lhs, op, rhs } => self.compile_binary(lhs, *op, rhs),

            ExpressionKind::Unary { op, operand } => {
                match op {
                    // 1. 取地址 (x&)
                    UnaryOperator::AddressOf => {
                        let ptr = self.compile_expr_ptr(operand)?;
                        Ok(ptr.as_basic_value_enum())
                    }

                    // 2. 解引用 (ptr^)
                    UnaryOperator::Dereference => {
                        let ptr = self.compile_expr(operand)?.into_pointer_value();

                        let res_type_key = self
                            .analysis
                            .types
                            .get(&expr.id)
                            .ok_or("Dereference result type missing in analyzer")?;

                        let llvm_ty = self
                            .compile_type(res_type_key)
                            .ok_or("Failed to compile deref type")?;

                        let val = self
                            .builder
                            .build_load(llvm_ty, ptr, "deref")
                            .map_err(|_| "Load failed")?;
                        Ok(val)
                    }

                    // 3. 取负 (-x)
                    // 支持整数和浮点数
                    UnaryOperator::Negate => {
                        let val = self.compile_expr(operand)?;
                        if val.is_float_value() {
                            Ok(self
                                .builder
                                .build_float_neg(val.into_float_value(), "fneg")
                                .unwrap()
                                .into())
                        } else if val.is_int_value() {
                            Ok(self
                                .builder
                                .build_int_neg(val.into_int_value(), "neg")
                                .unwrap()
                                .into())
                        } else {
                            Err("Negate operand must be int or float".into())
                        }
                    }

                    // 4. 非 (!x)
                    UnaryOperator::Not => {
                        let val = self.compile_expr(operand)?;
                        if val.is_int_value() {
                            Ok(self
                                .builder
                                .build_not(val.into_int_value(), "not")
                                .unwrap()
                                .into())
                        } else {
                            //? Analyzer 应该拦截了非 Int/Bool 类型
                            //? 更严谨的检查和报错
                            Err("Not operand must be integer or boolean".into())
                        }
                    }
                }
            }

            // L-Value to R-Value conversion (Load)
            ExpressionKind::Index { .. }
            | ExpressionKind::FieldAccess { .. } => {
                trace!("    [Load] 1. Compiling ptr...");
                let ptr = self.compile_expr_ptr(expr)?;
                trace!("    [Load] 1. Ptr compiled. Is Null? {}", ptr.is_null());
                
                trace!("    [Load] 2. Getting type key for Expr ID {:?}...", expr.id);
                let type_key = self.get_resolved_type(expr.id);
                trace!("    [Load] 2. Type key resolved: {:?}", type_key);
                
                trace!("    [Load] 3. Compiling LLVM type...");
                let llvm_ty = self.compile_type(&type_key).unwrap();
                trace!("    [Load] 3. LLVM Type compiled: {:?}", llvm_ty);
                
                trace!("    [Load] 4. Building Load instruction...");
                let val = self
                    .builder
                    .build_load(llvm_ty, ptr, "load")
                    .map_err(|_| "Load failed")?;
                trace!("    [Load] 5. Load successful.");
                
                Ok(val)
            }

            ExpressionKind::Path(path) => {
                // 1. 获取类型
                let type_key = self.get_resolved_type(expr.id);
                
                // 2. 获取 DefId
                // 如果解析失败，说明有问题，直接 panic
                let def_id = *self.analysis.path_resolutions.get(&expr.id).expect("Path not resolved");

                // 3. 【核心修正】检查是否是全局函数定义
                // 只有当 Path 指向的是真正的函数定义时，我们才去 functions 表里查
                // 如果是指向变量（参数/局部变量），即使类型是 Function，也要走 Load 逻辑
                
                let is_global_function = self.function_index.contains_key(&def_id);

                if is_global_function {
                    // --- Case A: 引用全局函数 (作为值) ---
                    
                    // B. 查 Analyzer 获取泛型实参
                    let generic_args = self.analysis.node_generic_args
                        .get(&expr.id)
                        .cloned()
                        .unwrap_or_default();

                    // C. 生成修饰名
                    let fn_mangled_name = self.analysis.get_mangled_function_name(def_id, &generic_args);

                    // D. 查表获取 FunctionValue
                    let fn_val = *self.functions.get(&fn_mangled_name)
                        .unwrap_or_else(|| panic!("Function '{}' not found/compiled when used as value", fn_mangled_name));
                    
                    // E. 返回函数指针
                    return Ok(fn_val.as_global_value().as_pointer_value().as_basic_value_enum());
                }

                // --- Case B: 普通变量 / 函数指针变量 ---
                // 走正常的 Load 逻辑
                let ptr = self.compile_expr_ptr(expr)?;
                let llvm_ty = self.compile_type(&type_key).unwrap();
                let val = self
                    .builder
                    .build_load(llvm_ty, ptr, "load")
                    .map_err(|_| "Load failed")?;
                Ok(val)
            }

            ExpressionKind::Call { callee, arguments } => self.compile_call(callee, arguments),

            ExpressionKind::Cast {
                expr: src_expr,
                target_type,
            } => self.compile_cast(src_expr, target_type, expr.id),

            ExpressionKind::StructLiteral { type_name, fields } => {
                // 1. 获取该表达式在当前上下文中的具体类型 (Analyzer 已计算好)
                // 例如：Instantiated { def_id: Box, args: [i32] }
                let struct_key = self.get_resolved_type(expr.id);

                // 2. 获取 Codegen 用的唯一修饰名 (例如 "Box_i32")
                let mangled_name = self.analysis.get_mangling_type_name(&struct_key);

                // 3. 查表获取 LLVM 结构体类型
                // 这里我们要解引用 (*) 拿到 StructType (它是 Copy 的)，
                // 这样就不再借用 self.struct_types 了
                let struct_type = *self
                    .struct_types
                    .get(&mangled_name)
                    .unwrap_or_else(|| panic!("Struct type '{}' not found in codegen", mangled_name));

                // 4. 【核心修正】查表获取字段索引映射，并 CLONE
                // .clone() 会把 HashMap 复制一份。
                // 这样 index_map 就变成了 owned data，不再借用 self。
                let index_map = self
                    .struct_field_indices
                    .get(&mangled_name)
                    .unwrap_or_else(|| panic!("Indices for '{}' not found", mangled_name))
                    .clone(); // <--- 关键！切断借用链

                // 此时 self 身上没有任何借用，完全自由！

                // 5. 构建结构体值
                let mut struct_val = struct_type.get_undef();

                for field in fields {
                    // 现在可以放心地 mut borrow self 了
                    let val = self.compile_expr(&field.value)?;
                    
                    // index_map 是我们拥有的局部变量，随便用
                    let idx = *index_map
                        .get(&field.field_name.name)
                        .ok_or_else(|| format!("Field '{}' not found", field.field_name.name))?;

                    eprintln!("[Codegen Debug] Inserting field '{}' at index {} into struct '{}'", field.field_name.name, idx, mangled_name);
                    eprintln!("[Codegen Debug]   Value type: {:?}", val.get_type());
                    eprintln!("[Codegen Debug]   Struct type: {:?}", struct_type);

                    struct_val = self
                        .builder
                        .build_insert_value(struct_val, val, idx, "insert")
                        .map_err(|_| "Insert value failed")?
                        .into_struct_value();
                }

                Ok(struct_val.as_basic_value_enum())
            }

            ExpressionKind::MethodCall { receiver, method_name, arguments } => {
                self.compile_method_call_dispatch(expr.id, receiver, method_name, arguments)
            }

            ExpressionKind::StaticAccess { target: _, member: _ } => {
                // 1. 使用 get_resolved_type 获取具体类型
                let ty = self.get_resolved_type(expr.id);

                // 2. 匹配 TypeKey
                if let TypeKey::IntegerLiteral(val) = ty {
                    // 默认生成 i64
                    Ok(self
                        .context
                        .i64_type()
                        .const_int(val, false)
                        .as_basic_value_enum())
                } else {
                    // 可能是读取静态变量，或者 Enum 不是简单的整数
                    Err("Static access (non-enum/non-const) not implemented".into())
                }
            }

            // @sizeof
            ExpressionKind::SizeOf(target_type) => self.compile_sizeof(target_type),

            // @alignof
            ExpressionKind::AlignOf(target_type) => {
                // 1. 获取 TypeKey
                let type_key = self
                    .analysis
                    .types
                    .get(&target_type.id)
                    .ok_or("AlignOf target type not resolved")?;

                // 2. 转为 LLVM Type
                let llvm_ty = self
                    .compile_type(type_key)
                    .ok_or("Cannot compile type for alignof")?;

                // 3. 生成 alignof 指令
                Ok(llvm_ty.get_alignment().as_basic_value_enum())
            }
        }
    }

    pub fn compile_expr_ptr(&mut self, expr: &Expression) -> Result<PointerValue<'ctx>, String> {
        match &expr.kind {
            ExpressionKind::Path(path) => {
                let def_id = self.analysis.path_resolutions.get(&expr.id).unwrap();
                self.get_variable_ptr(*def_id).ok_or("Var missing".into())
            }

            ExpressionKind::FieldAccess {
                receiver,
                field_name,
            } => {
                // 1. 编译 Receiver 的指针
                let ptr = self.compile_expr_ptr(receiver)?;
                
                // 2. 获取 Receiver 的具体类型 (关键：使用 get_resolved_type 替换泛型 T -> i32)
                let recv_type_key = self.get_resolved_type(receiver.id);

                // 3. 确保是实例化类型 (Struct<T> 或 Struct)
                if let TypeKey::Instantiated { .. } = &recv_type_key {
                    // 4. 获取 Mangled Name (e.g. "Box_i32")
                    let mangled_name = self.analysis.get_mangling_type_name(&recv_type_key);

                    // 5. 使用 Mangled Name 查索引
                    let idx = self
                        .get_field_index(&mangled_name, &field_name.name)
                        .ok_or_else(|| format!("Field '{}' missing in struct '{}'", field_name.name, mangled_name))?;

                    // 6. 使用 Mangled Name 查 LLVM Type
                    let st_ty = *self
                        .struct_types
                        .get(&mangled_name)
                        .expect("Struct type missing in codegen");

                    // 7. 生成 GEP
                    let field_ptr = unsafe {
                        self.builder
                            .build_struct_gep(st_ty, ptr, idx, "gep")
                            .map_err(|_| "GEP failed")?
                    };
                    Ok(field_ptr)
                } else {
                    return Err(format!("Field access on non-struct type: {:?}", recv_type_key));
                }
            }

            ExpressionKind::Index { target, index } => {
                let ptr = self.compile_expr_ptr(target)?;
                let idx = self.compile_expr(index)?.into_int_value();
                
                // 1. 获取类型 (Owned)
                let target_key = self.get_resolved_type(target.id);

                // 2. 【关键】使用引用进行匹配 (&target_key)
                // 这样 inner 就变成了 &Box<TypeKey> (引用)，而不是 Box<TypeKey> (所有权)
                match &target_key {
                    TypeKey::Array(_inner, _) => {
                        // 这里我们再次使用 &target_key，它是完整的，因为上面只是借用
                        // 对于 GEP Array，我们需要数组本身的类型 [10 x i32]
                        let arr_ty = self.compile_type(&target_key).unwrap();
                        
                        let p = unsafe {
                            self.builder
                                .build_gep(
                                    arr_ty,
                                    ptr,
                                    &[self.context.i64_type().const_zero(), idx],
                                    "arr_gep",
                                )
                                .unwrap()
                        };
                        Ok(p) // 记得转 Enum
                    }
                    TypeKey::Pointer(inner, _) => {
                        // inner 是 &Box<TypeKey>
                        // compile_type 需要 &TypeKey
                        // Box 实现了 Deref，所以直接传 inner 即可自动解引用
                        // 或者显式写 &**inner
                        let inner_ty = self.compile_type(inner).unwrap();
                        
                        let p = unsafe {
                            self.builder
                                .build_gep(inner_ty, ptr, &[idx], "ptr_gep")
                                .unwrap()
                        };
                        Ok(p)
                    }
                    _ => Err("Index error: expected Array or Pointer".into()),
                }
            }

            ExpressionKind::Unary {
                op: UnaryOperator::Dereference,
                operand,
            } => {
                let val = self.compile_expr(operand)?;
                Ok(val.into_pointer_value())
            }

            _ => Err("Not an L-Value".into()),
        }
    }

    // ========================================================================
    // 7. 辅助函数
    // ========================================================================

    // ==========================================
    // Method Call 编译逻辑拆分
    // ==========================================

    fn compile_method_call_dispatch(
        &mut self,
        expr_id: NodeId,
        receiver: &Expression,
        method_name: &Identifier,
        arguments: &[Expression]
    ) -> Result<BasicValueEnum<'ctx>, String> {
        // 1. 获取 Receiver 类型
        let receiver_ty_key = self.get_resolved_type(receiver.id);

        // 2. 尝试判断是否为标准方法
        // 逻辑：如果 analyzer.method_registry 里能查到，就是标准方法
        // 我们复用 Analyzer 的 find_method_in_registry 逻辑 (假设 analysis 是 pub 的)
        // 这里的逻辑有点重复，但为了 Codegen 独立性，我们再查一次
        
        // 我们利用 Analyzer 提供的 Helper 来判断
        // (需要在 AnalysisContext 里把 find_method_in_registry 变成 pub，或者手写查找)
        let is_standard_method = self.find_method_info(&receiver_ty_key, &method_name.name).is_some();

        if is_standard_method {
            return self.compile_standard_method_call(expr_id, receiver, method_name, arguments, &receiver_ty_key);
        } else {
            // 否则尝试当做字段函数指针调用
            return self.compile_field_fn_ptr_call(receiver, method_name, arguments, &receiver_ty_key);
        }
    }

    // 辅助：在 Codegen 里查找 MethodInfo (复用 Analyzer 数据)
    fn find_method_info(&self, receiver_ty: &TypeKey, name: &str) -> Option<&MethodInfo> {
        // 1. 精确
        if let Some(methods) = self.analysis.method_registry.get(receiver_ty) {
            if let Some(info) = methods.get(name) { return Some(info); }
        }
        // 2. 模糊
        if let TypeKey::Instantiated { def_id, .. } = receiver_ty {
            for (key, methods) in &self.analysis.method_registry {
                if let TypeKey::Instantiated { def_id: k_id, .. } = key {
                    if k_id == def_id {
                         if let Some(info) = methods.get(name) { return Some(info); }
                    }
                }
            }
        }
        None
    }

    /// 路径 1: 编译标准方法调用
    fn compile_standard_method_call(
        &mut self,
        expr_id: NodeId,
        receiver: &Expression,
        method_name: &Identifier,
        arguments: &[Expression],
        receiver_ty: &TypeKey
    ) -> Result<BasicValueEnum<'ctx>, String> {
        // 1. 查找定义信息
        let method_info = self.find_method_info(receiver_ty, &method_name.name)
            .ok_or("Method info missing in codegen")?;

        // 2. 获取泛型参数 (查表)
        let method_args = self.analysis.node_generic_args.get(&expr_id).cloned().unwrap_or_default();

        // 3. 重建名字并获取函数
        let fn_name = self.analysis.get_mangled_function_name(method_info.def_id, &method_args);
        let fn_val = *self.functions.get(&fn_name)
            .ok_or_else(|| format!("Function '{}' not compiled", fn_name))?;

        // 4. 准备参数
        let mut compiled_args = Vec::new();
        compiled_args.push(self.compile_expr(receiver)?.into()); // self
        for arg in arguments {
            compiled_args.push(self.compile_expr(arg)?.into());
        }

        // 5. 调用
        let call = self.builder.build_call(fn_val, &compiled_args, "call").map_err(|_| "Call failed")?;
        self.handle_call_return(call)
    }

    /// 路径 2: 编译字段函数指针调用
    fn compile_field_fn_ptr_call(
        &mut self,
        receiver: &Expression,
        method_name: &Identifier,
        arguments: &[Expression],
        receiver_ty: &TypeKey
    ) -> Result<BasicValueEnum<'ctx>, String> {
        // [Logic from previous answer]
        // 1. 获取 Struct Key (穿透指针)
        let struct_key = match receiver_ty {
            TypeKey::Pointer(inner, _) => *inner.clone(),
            _ => receiver_ty.clone(),
        };
        let mangled_struct_name = self.analysis.get_mangling_type_name(&struct_key);

        // 2. 查字段索引
        let idx = self.get_field_index(&mangled_struct_name, &method_name.name)
            .ok_or_else(|| format!("Field '{}' not found in struct '{}'", method_name.name, mangled_struct_name))?;

        // 3. 获取字段地址
        let struct_ptr = self.compile_expr_ptr(receiver)?;
        let st_ty = *self.struct_types.get(&mangled_struct_name).expect("Struct type missing");
        
        let field_ptr = unsafe {
            self.builder.build_struct_gep(st_ty, struct_ptr, idx, "fn_field_gep").map_err(|_| "GEP failed")?
        };

        // 4. Load 函数指针
        // 为了 Load，我们需要知道字段的具体类型。
        // 从 instantiated_structs 里查
        let fields_list = self.analysis.instantiated_structs.get(&mangled_struct_name).expect("Instantiated struct missing");
        let (_, field_type_key) = &fields_list[idx as usize];
        let llvm_field_type = self.compile_type(field_type_key).unwrap();

        let fn_ptr_val = self.builder.build_load(llvm_field_type, field_ptr, "fn_ptr_load")
            .map_err(|_| "Load failed")?
            .into_pointer_value();

        // 5. 准备参数 (没有 self)
        let mut compiled_args = Vec::new();
        for arg in arguments {
            compiled_args.push(self.compile_expr(arg)?.into());
        }

        // 6. 编译签名并间接调用
        let fn_type = self.compile_function_type_signature(field_type_key)?;
        let call = self.builder.build_indirect_call(fn_type, fn_ptr_val, &compiled_args, "indirect_call")
            .map_err(|_| "Indirect call failed")?;

        self.handle_call_return(call)
    }

    fn compile_literal(
        &self,
        lit: &Literal,
        type_key: &TypeKey,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        match lit {
            Literal::Integer(val) => {
                match type_key {
                    TypeKey::Primitive(p) => {
                        let ty = self.compile_primitive_type(p).into_int_type();
                        Ok(ty.const_int(*val, false).as_basic_value_enum())
                    }
                    // Fallback
                    _ => Ok(self
                        .context
                        .i64_type()
                        .const_int(*val, false)
                        .as_basic_value_enum()),
                }
            }
            Literal::Float(val) => Ok(self
                .context
                .f64_type()
                .const_float(*val)
                .as_basic_value_enum()),
            Literal::Boolean(b) => Ok(self
                .context
                .bool_type()
                .const_int(if *b { 1 } else { 0 }, false)
                .as_basic_value_enum()),
            Literal::String(s) => {
                let str_val = self.context.const_string(s.as_bytes(), true);
                let global = self.module.add_global(
                    str_val.get_type(),
                    Some(AddressSpace::default()),
                    "str_lit",
                );
                global.set_initializer(&str_val);
                global.set_constant(true);
                global.set_linkage(Linkage::Internal);
                Ok(global.as_pointer_value().as_basic_value_enum())
            }
            Literal::Char(c) => Ok(self
                .context
                .i32_type()
                .const_int(*c as u64, false)
                .as_basic_value_enum()),
        }
    }

    fn compile_binary(
        &mut self,
        lhs: &Expression,
        op: BinaryOperator,
        rhs: &Expression,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        // 如果是逻辑运算，进入短路求值逻辑
        if matches!(op, BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr) {
            return self.compile_short_circuit_binary(lhs, op, rhs);
        }

        let lhs_val = self.compile_expr(lhs)?;
        let rhs_val = self.compile_expr(rhs)?;

        // 1. 浮点数处理
        if lhs_val.is_float_value() {
            return self.compile_float_binary(
                lhs_val.into_float_value(),
                op,
                rhs_val.into_float_value(),
            );
        }

        // 2. 【关键修复】指针处理
        // 如果直接对指针调用 into_int_value() 会导致 Panic
        if lhs_val.is_pointer_value() {
            // 获取 LHS 的类型信息 (Analyzer 必须保证类型存在)
            let lhs_type_key = self
                .analysis
                .types
                .get(&lhs.id)
                .ok_or("Type missing for pointer op")?;

            return self.compile_pointer_binary(
                lhs_val.into_pointer_value(),
                op,
                rhs_val,
                lhs_type_key,
            );
        }

        // 3. 整数处理 (默认)
        if lhs_val.is_int_value() && rhs_val.is_int_value() {
            let l = lhs_val.into_int_value();
            let r = rhs_val.into_int_value();

            // 获取类型以判断符号 (Signed/Unsigned)
            let type_key = self
                .analysis
                .types
                .get(&lhs.id)
                .ok_or("Type missing for binary op")?;
            let is_signed = self.is_signed(type_key);

            //? TODO: 检测溢出？
            return match op {
                BinaryOperator::Add => Ok(self.builder.build_int_add(l, r, "add").unwrap().into()),
                BinaryOperator::Subtract => {
                    Ok(self.builder.build_int_sub(l, r, "sub").unwrap().into())
                }
                BinaryOperator::Multiply => {
                    Ok(self.builder.build_int_mul(l, r, "mul").unwrap().into())
                }

                // 位运算
                BinaryOperator::BitwiseAnd => {
                    Ok(self.builder.build_and(l, r, "and").unwrap().into())
                }
                BinaryOperator::BitwiseOr => Ok(self.builder.build_or(l, r, "or").unwrap().into()),
                BinaryOperator::BitwiseXor => {
                    Ok(self.builder.build_xor(l, r, "xor").unwrap().into())
                }

                // 除法 & 取模
                BinaryOperator::Divide => {
                    if is_signed {
                        Ok(self
                            .builder
                            .build_int_signed_div(l, r, "sdiv")
                            .unwrap()
                            .into())
                    } else {
                        Ok(self
                            .builder
                            .build_int_unsigned_div(l, r, "udiv")
                            .unwrap()
                            .into())
                    }
                }
                BinaryOperator::Modulo => {
                    if is_signed {
                        Ok(self
                            .builder
                            .build_int_signed_rem(l, r, "srem")
                            .unwrap()
                            .into())
                    } else {
                        Ok(self
                            .builder
                            .build_int_unsigned_rem(l, r, "urem")
                            .unwrap()
                            .into())
                    }
                }

                // 移位
                BinaryOperator::ShiftLeft => {
                    Ok(self.builder.build_left_shift(l, r, "shl").unwrap().into())
                }
                BinaryOperator::ShiftRight => Ok(self
                    .builder
                    .build_right_shift(l, r, is_signed, "shr")
                    .unwrap()
                    .into()),

                // 比较
                BinaryOperator::Equal => Ok(self
                    .builder
                    .build_int_compare(IntPredicate::EQ, l, r, "eq")
                    .unwrap()
                    .into()),
                BinaryOperator::NotEqual => Ok(self
                    .builder
                    .build_int_compare(IntPredicate::NE, l, r, "ne")
                    .unwrap()
                    .into()),
                BinaryOperator::Less => {
                    let pred = if is_signed {
                        IntPredicate::SLT
                    } else {
                        IntPredicate::ULT
                    };
                    Ok(self
                        .builder
                        .build_int_compare(pred, l, r, "lt")
                        .unwrap()
                        .into())
                }
                BinaryOperator::LessEqual => {
                    let pred = if is_signed {
                        IntPredicate::SLE
                    } else {
                        IntPredicate::ULE
                    };
                    Ok(self
                        .builder
                        .build_int_compare(pred, l, r, "le")
                        .unwrap()
                        .into())
                }
                BinaryOperator::Greater => {
                    let pred = if is_signed {
                        IntPredicate::SGT
                    } else {
                        IntPredicate::UGT
                    };
                    Ok(self
                        .builder
                        .build_int_compare(pred, l, r, "gt")
                        .unwrap()
                        .into())
                }
                BinaryOperator::GreaterEqual => {
                    let pred = if is_signed {
                        IntPredicate::SGE
                    } else {
                        IntPredicate::UGE
                    };
                    Ok(self
                        .builder
                        .build_int_compare(pred, l, r, "ge")
                        .unwrap()
                        .into())
                }

                _ => Err(format!("Binary op {:?} not supported for integer", op)),
            };
        }

        Err("Operands type mismatch or unsupported type for binary op".into())
    }

    fn compile_short_circuit_binary(
        &mut self,
        lhs: &Expression,
        op: BinaryOperator,
        rhs: &Expression,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let parent = self.get_current_function();
        let lhs_val = self.compile_expr(lhs)?;

        // 确保 LHS 是 bool (i1)
        if !lhs_val.is_int_value() {
            return Err("Logical op requires boolean/integer operands".into());
        }
        let lhs_int = lhs_val.into_int_value();

        // 如果不是 1 位宽 (i1)，需要跟 0 比较转成 i1
        let i1_type = self.context.bool_type(); // i1
        let zero = self.context.i64_type().const_zero();

        let lhs_bool = if lhs_int.get_type().get_bit_width() > 1 {
            // val != 0
            self.builder
                .build_int_compare(IntPredicate::NE, lhs_int, zero, "tobool")
                .unwrap()
        } else {
            lhs_int
        };

        // 创建基本块
        let rhs_bb = self.context.append_basic_block(parent, "logic_rhs");
        let merge_bb = self.context.append_basic_block(parent, "logic_merge");

        // 根据运算类型决定短路逻辑
        // And: LHS true -> check RHS; LHS false -> merge (false)
        // Or:  LHS true -> merge (true); LHS false -> check RHS
        match op {
            BinaryOperator::LogicalAnd => {
                self.builder
                    .build_conditional_branch(lhs_bool, rhs_bb, merge_bb)
                    .ok();
            }
            BinaryOperator::LogicalOr => {
                self.builder
                    .build_conditional_branch(lhs_bool, merge_bb, rhs_bb)
                    .ok();
            }
            _ => unreachable!(),
        }

        // 记录 LHS 结束时的 Block (用于 PHI)
        let lhs_end_bb = self.builder.get_insert_block().unwrap();

        // --- 编译 RHS 块 ---
        self.builder.position_at_end(rhs_bb);
        let rhs_val = self.compile_expr(rhs)?;
        if !rhs_val.is_int_value() {
            return Err("RHS must be int/bool".into());
        }

        let rhs_int = rhs_val.into_int_value();
        let rhs_bool = if rhs_int.get_type().get_bit_width() > 1 {
            self.builder
                .build_int_compare(IntPredicate::NE, rhs_int, zero, "tobool")
                .unwrap()
        } else {
            rhs_int
        };

        self.builder.build_unconditional_branch(merge_bb).ok();
        let rhs_end_bb = self.builder.get_insert_block().unwrap();

        // --- 编译 Merge 块 (PHI Node) ---
        self.builder.position_at_end(merge_bb);

        let phi = self.builder.build_phi(i1_type, "logic_res").unwrap();

        // 构建 PHI 入口
        let true_val = i1_type.const_int(1, false);
        let false_val = i1_type.const_int(0, false);

        match op {
            BinaryOperator::LogicalAnd => {
                // 如果来自 LHS (说明 LHS 为 false)，结果为 false
                // 如果来自 RHS，结果为 RHS 的值
                phi.add_incoming(&[(&false_val, lhs_end_bb), (&rhs_bool, rhs_end_bb)]);
            }
            BinaryOperator::LogicalOr => {
                // 如果来自 LHS (说明 LHS 为 true)，结果为 true
                // 如果来自 RHS，结果为 RHS 的值
                phi.add_incoming(&[(&true_val, lhs_end_bb), (&rhs_bool, rhs_end_bb)]);
            }
            _ => unreachable!(),
        }

        Ok(phi.as_basic_value().into())
    }

    fn compile_float_binary(
        &self,
        l: FloatValue<'ctx>,
        op: BinaryOperator,
        r: FloatValue<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        match op {
            BinaryOperator::Add => Ok(self.builder.build_float_add(l, r, "fadd").unwrap().into()),
            BinaryOperator::Subtract => {
                Ok(self.builder.build_float_sub(l, r, "fsub").unwrap().into())
            }
            BinaryOperator::Multiply => {
                Ok(self.builder.build_float_mul(l, r, "fmul").unwrap().into())
            }
            BinaryOperator::Divide => {
                Ok(self.builder.build_float_div(l, r, "fdiv").unwrap().into())
            }
            BinaryOperator::Modulo => {
                Ok(self.builder.build_float_rem(l, r, "frem").unwrap().into())
            }

            // 9是OEQ (Ordered Equal), FloatRredicate 都不为 NaN 且相等
            BinaryOperator::Equal => Ok(self
                .builder
                .build_float_compare(FloatPredicate::OEQ, l, r, "feq")
                .unwrap()
                .into()),
            BinaryOperator::NotEqual => Ok(self
                .builder
                .build_float_compare(FloatPredicate::ONE, l, r, "fne")
                .unwrap()
                .into()),
            BinaryOperator::Less => Ok(self
                .builder
                .build_float_compare(FloatPredicate::OLT, l, r, "flt")
                .unwrap()
                .into()),
            BinaryOperator::LessEqual => Ok(self
                .builder
                .build_float_compare(FloatPredicate::OLE, l, r, "fle")
                .unwrap()
                .into()),
            BinaryOperator::Greater => Ok(self
                .builder
                .build_float_compare(FloatPredicate::OGT, l, r, "fgt")
                .unwrap()
                .into()),
            BinaryOperator::GreaterEqual => Ok(self
                .builder
                .build_float_compare(FloatPredicate::OGE, l, r, "fge")
                .unwrap()
                .into()),

            _ => Err(format!("Binary op {:?} not supported for float", op)),
        }
    }

    fn compile_call(
        &mut self,
        callee: &Expression,
        args: &[Expression],
    ) -> Result<BasicValueEnum<'ctx>, String> {
        trace!("      Compiling call...");
        // 1. 预先编译所有参数
        let mut compiled_args = Vec::new();
        for arg in args {
            compiled_args.push(self.compile_expr(arg)?.into());
        }

        // =========================================================
        // Case 1: 尝试作为直接函数调用 (Direct Call)
        // =========================================================
        let direct_call_result = if let ExpressionKind::Path(path) = &callee.kind {
            if let Some(&def_id) = self.analysis.path_resolutions.get(&callee.id) {
                let generic_args = self.analysis.node_generic_args
                    .get(&callee.id)
                    .cloned()
                    .unwrap_or_default();
                let fn_mangled_name = self.analysis.get_mangled_function_name(def_id, &generic_args);
                
                // 【关键修改】只在找到函数时才返回，否则继续往下走
                if let Some(fn_val) = self.functions.get(&fn_mangled_name) {
                    Some((fn_val, compiled_args.clone())) // 找到了！
                } else {
                    None // 没找到，可能是局部变量
                }
            } else { None }
        } else { None };

        if let Some((fn_val, args)) = direct_call_result {
             let call_site = self.builder.build_call(*fn_val, &args, "direct_call").unwrap();
             return self.handle_call_return(call_site);
        }

        // =========================================================
        // Case 2: 静态方法调用 (Static Method Call)
        // e.g. Box::new(1) 或 MyStruct#<i32>::method()
        // =========================================================
        if let ExpressionKind::StaticAccess { .. } = &callee.kind {
            if let Some(&def_id) = self.analysis.path_resolutions.get(&callee.id) {
                // A. 查表获取泛型实参
                // 对于 Box<i32>::new，Analyzer 应该已经把 [i32] 填入 node_generic_args 了
                let generic_args = self.analysis.node_generic_args
                    .get(&callee.id)
                    .cloned()
                    .unwrap_or_default();

                // B. 重建修饰名
                let fn_mangled_name = self.analysis.get_mangled_function_name(def_id, &generic_args);

                // C. 查表
                if let Some(fn_val) = self.functions.get(&fn_mangled_name) {
                    trace!("        Build call to '{}'", fn_mangled_name);
                    let call_site = self
                        .builder
                        .build_call(*fn_val, &compiled_args, "static_call")
                        .map_err(|_| "Static call failed")?;
                    return self.handle_call_return(call_site);
                } else {
                    panic!("ICE: Static method '{}' resolved but not compiled", fn_mangled_name);
                }
            }
        }

        // =========================================================
        // Case 3: 间接调用 (Indirect Call / Function Pointer)
        // e.g. let f = foo; f();
        // =========================================================
        
        // A. 编译表达式得到函数指针 (PointerValue)
        let fn_ptr_val = self.compile_expr(callee)?.into_pointer_value();

        // B. 获取函数类型签名
        // 【关键修正】必须使用 get_resolved_type 而不是直接查 types
        // 因为如果这是一个泛型函数里的间接调用 (let f: fn(T) -> T)，我们需要把 T 换成具体类型
        let callee_type_key = self.get_resolved_type(callee.id);

        // C. 将 TypeKey 转换为 LLVM FunctionType
        let fn_type = self.compile_function_type_signature(&callee_type_key)?;

        // D. 生成间接调用指令
        let call_site = self
            .builder
            .build_indirect_call(fn_type, fn_ptr_val, &compiled_args, "indirect_call")
            .map_err(|_| "Indirect call failed")?;

        self.handle_call_return(call_site)
    }

    // 辅助函数: 统一处理返回值
    fn handle_call_return(
        &self,
        call_site: inkwell::values::CallSiteValue<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        use inkwell::values::ValueKind;
        match call_site.try_as_basic_value() {
            ValueKind::Basic(val) => Ok(val),
            ValueKind::Instruction(_) => {
                // Void 返回，生成 Unit 占位
                Ok(self
                    .context
                    .struct_type(&[], false)
                    .const_zero()
                    .as_basic_value_enum())
            }
        }
    }

    // 辅助函数: 提取函数签名
    fn compile_function_type_signature(&self, key: &TypeKey) -> Result<FunctionType<'ctx>, String> {
        match key {
            // 解构加上 is_variadic
            TypeKey::Function {
                params,
                ret,
                is_variadic,
            } => {
                // 1. 转换参数类型
                let mut param_types = Vec::new();
                for param in params {
                    let llvm_ty = self
                        .compile_type(param)
                        .ok_or("Failed to compile param type")?;
                    param_types.push(llvm_ty.into());
                }

                // 2. 转换返回值类型，并传入 is_variadic
                if let Some(ret_key) = ret {
                    if let TypeKey::Primitive(PrimitiveType::Unit) = **ret_key {
                        Ok(self.context.void_type().fn_type(&param_types, *is_variadic))
                    } else {
                        let ret_llvm_ty = self
                            .compile_type(ret_key)
                            .ok_or("Failed to compile ret type")?;
                        Ok(ret_llvm_ty.fn_type(&param_types, *is_variadic))
                    }
                } else {
                    Ok(self.context.void_type().fn_type(&param_types, *is_variadic))
                }
            }

            TypeKey::Pointer(inner, _) => self.compile_function_type_signature(inner),

            _ => Err(format!(
                "Expected function type for indirect call, got {:?}",
                key
            )),
        }
    }

    fn compile_unary(
        &mut self,
        op: UnaryOperator,
        operand: &Expression,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let val = self.compile_expr(operand)?;
        match op {
            UnaryOperator::Negate => {
                if val.is_int_value() {
                    Ok(self
                        .builder
                        .build_int_neg(val.into_int_value(), "neg")
                        .unwrap()
                        .into())
                } else {
                    Ok(self
                        .builder
                        .build_float_neg(val.into_float_value(), "neg")
                        .unwrap()
                        .into())
                }
            }
            UnaryOperator::Not => Ok(self
                .builder
                .build_not(val.into_int_value(), "not")
                .unwrap()
                .into()),
            _ => Err("Should be handled elsewhere".into()),
        }
    }

    /// 辅助函数：从符号表中查找变量地址
    fn get_variable_ptr(&self, def_id: DefId) -> Option<PointerValue<'ctx>> {
        self.variables.get(&def_id)
            .or_else(|| self.globals.get(&def_id)) // 查全局表
            .cloned()
    }

    // 辅助：检查类型是否有符号
    fn is_signed(&self, key: &TypeKey) -> bool {
        match key {
            TypeKey::Primitive(p) => matches!(
                p,
                PrimitiveType::I8
                    | PrimitiveType::I16
                    | PrimitiveType::I32
                    | PrimitiveType::I64
                    | PrimitiveType::ISize
            ),
            // 字面量如果没有被 Analyzer 固化为具体类型，
            // 默认行为通常视作有符号i64
            TypeKey::IntegerLiteral(_) => true,

            // 指针、Char、Bool、数组等视为无符号用于比较
            _ => false,
        }
    }

    fn compile_cast(
        &mut self,
        src_expr: &Expression,
        target_ast_type: &Type,
        cast_expr_id: NodeId,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        // 1. 编译源表达式
        let src_val = self.compile_expr(src_expr)?;

        // 2. 获取类型信息
        // src_key: 源表达式的语义类型
        // target_key: Cast 表达式本身的语义类型
        let src_key = self
            .analysis
            .types
            .get(&src_expr.id)
            .ok_or("Source type missing")?;
        let target_key = self
            .analysis
            .types
            .get(&cast_expr_id)
            .ok_or("Target type missing")?;

        // 3. 获取 LLVM 目标类型
        let target_llvm_ty = self
            .compile_type(target_key)
            .ok_or("Compile target type failed")?;

        // 4. 分类讨论生成指令
        match (src_key, target_key) {
            // === Case A: 整数 -> 整数 (包含 Bool, Char) ===
            (TypeKey::Primitive(src_p), TypeKey::Primitive(target_p))
                if self.is_integer_kind(&src_p) && self.is_integer_kind(&target_p) =>
            {
                let src_int = src_val.into_int_value();
                let target_int_ty = target_llvm_ty.into_int_type();
                let src_width = src_int.get_type().get_bit_width();
                let target_width = target_int_ty.get_bit_width();

                if src_width > target_width {
                    // 截断: i32 -> i8
                    Ok(self
                        .builder
                        .build_int_truncate(src_int, target_int_ty, "trunc")
                        .unwrap()
                        .into())
                } else if src_width < target_width {
                    // 扩展: i8 -> i32
                    // 需要检查源类型是否有符号
                    if self.is_signed(src_key) {
                        Ok(self
                            .builder
                            .build_int_s_extend(src_int, target_int_ty, "sext")
                            .unwrap()
                            .into())
                    } else {
                        Ok(self
                            .builder
                            .build_int_z_extend(src_int, target_int_ty, "zext")
                            .unwrap()
                            .into())
                    }
                } else {
                    // 位宽相同: i32 -> u32, No-op
                    Ok(src_int.into())
                }
            }

            // === Case B: 浮点 -> 浮点 ===
            (TypeKey::Primitive(src_p), TypeKey::Primitive(target_p))
                if self.is_float_kind(&src_p) && self.is_float_kind(&target_p) =>
            {
                let src_float = src_val.into_float_value();
                let target_float_ty = target_llvm_ty.into_float_type();
                Ok(self
                    .builder
                    .build_float_cast(src_float, target_float_ty, "fpcast")
                    .unwrap()
                    .into())
            }

            // === Case C: 整数 -> 浮点 ===
            (TypeKey::Primitive(src_p), TypeKey::Primitive(target_p))
                if self.is_integer_kind(src_p) && self.is_float_kind(target_p) =>
            {
                let src_int = src_val.into_int_value();
                let target_float_ty = target_llvm_ty.into_float_type();

                if self.is_signed(src_key) {
                    Ok(self
                        .builder
                        .build_signed_int_to_float(src_int, target_float_ty, "sitofp")
                        .unwrap()
                        .into())
                } else {
                    Ok(self
                        .builder
                        .build_unsigned_int_to_float(src_int, target_float_ty, "uitofp")
                        .unwrap()
                        .into())
                }
            }

            // === Case D: 浮点 -> 整数 ===
            (TypeKey::Primitive(src_p), TypeKey::Primitive(target_p))
                if self.is_float_kind(&src_p) && self.is_integer_kind(&target_p) =>
            {
                let src_float = src_val.into_float_value();
                let target_int_ty = target_llvm_ty.into_int_type();

                if self.is_signed(target_key) {
                    Ok(self
                        .builder
                        .build_float_to_signed_int(src_float, target_int_ty, "fptosi")
                        .unwrap()
                        .into())
                } else {
                    Ok(self
                        .builder
                        .build_float_to_unsigned_int(src_float, target_int_ty, "fptoui")
                        .unwrap()
                        .into())
                }
            }

            // === Case E: 指针 -> 整数 ===
            (TypeKey::Pointer(..), TypeKey::Primitive(p)) if self.is_integer_kind(p) => {
                let src_ptr = src_val.into_pointer_value();
                let target_int_ty = target_llvm_ty.into_int_type();
                Ok(self
                    .builder
                    .build_ptr_to_int(src_ptr, target_int_ty, "ptr2int")
                    .unwrap()
                    .into())
            }

            // === Case F: 整数 -> 指针 ===
            (TypeKey::Primitive(p), TypeKey::Pointer(..)) if self.is_integer_kind(p) => {
                let src_int = src_val.into_int_value();
                let target_ptr_ty = target_llvm_ty.into_pointer_type();
                Ok(self
                    .builder
                    .build_int_to_ptr(src_int, target_ptr_ty, "int2ptr")
                    .unwrap()
                    .into())
            }

            // === Case G: 指针 -> 指针 (Bitcast) ===
            (TypeKey::Pointer(..), TypeKey::Pointer(..)) => {
                // Opaque Pointers 下通常是 No-op，但如果涉及 AddressSpace 转换需要 addrspacecast
                // 暂时bitcast
                //? LLVM 会自动优化?
                let src_ptr = src_val.into_pointer_value();
                let target_ptr_ty = target_llvm_ty.into_pointer_type();
                Ok(self
                    .builder
                    .build_pointer_cast(src_ptr, target_ptr_ty, "bitcast")
                    .unwrap()
                    .into())
            }

            // === Case H: 数组 -> 指针 (Array Decay) ===
            (TypeKey::Array(..), TypeKey::Pointer(..)) => {
                // 如果 src_val 已经是 Pointer，直接 Bitcast
                if src_val.is_pointer_value() {
                    let src_ptr = src_val.into_pointer_value();
                    let target_ptr_ty = target_llvm_ty.into_pointer_type();
                    Ok(self
                        .builder
                        .build_pointer_cast(src_ptr, target_ptr_ty, "array_decay")
                        .unwrap()
                        .into())
                } else {
                    Err("Cannot cast array value to pointer directly".into())
                }
            }

            // 默认错误
            _ => Err(format!(
                "Unsupported cast from {:?} to {:?}",
                src_key, target_key
            )),
        }
    }

    // 全局变量编译逻辑
    fn compile_global_variable(&mut self, id: DefId, def: &GlobalDefinition) {
        // 1. 获取类型
        let ty_key = self.get_resolved_type(id);
        let llvm_ty = self.compile_type(&ty_key).unwrap();

        // 2. 获取修饰名
        // 如果是 extern，不做 mangle，且不设置 initializer
        let name = if def.is_extern {
            def.name.name.clone()
        } else {
            self.analysis
                .mangled_names
                .get(&id)
                .cloned()
                .unwrap_or(def.name.name.clone())
        };

        // 3. 创建全局变量
        let global = self
            .module
            .add_global(llvm_ty, Some(AddressSpace::default()), &name);

        if def.is_extern {
            // Extern 变量：声明 linkage 为 External，不设置 initializer
            global.set_linkage(Linkage::External);
            // 不设置 initializer
        } else {
            // 4. 设置可变性 (LLVM IR 里的 constant 意味着只读)
            global.set_constant(def.modifier == Mutability::Constant);

            // 5. 设置初始化值
            if let Some(init) = &def.initializer {
                // A: 如果是简单字面量 (String, Float, Bool)，直接编译
                if let ExpressionKind::Literal(lit) = &init.kind {
                    let lit_ty = self.get_resolved_type(init.id);
                    let val = self
                        .compile_literal(lit, &lit_ty)
                        .expect("Literal compile failed");
                    global.set_initializer(&val);
                }
                // B: 如果 Analyzer 已经计算出了整数值 (Int 运算和 指针强转)
                else if let Some(&const_val) = self.analysis.constants.get(&id) {
                    if llvm_ty.is_int_type() {
                        // 情况 1: 整数类型 (i32, u64...)
                        let int_ty = llvm_ty.into_int_type();
                        // 视为无符号处理raw bits
                        let val = int_ty.const_int(const_val, false);
                        global.set_initializer(&val.as_basic_value_enum());
                    } else if llvm_ty.is_pointer_type() {
                        // 情况 2: 指针类型 (e.g., 0xB8000 as *u16)
                        let i64_ty = self.context.i64_type();
                        let int_val = i64_ty.const_int(const_val, false);

                        let ptr_val = int_val.const_to_pointer(llvm_ty.into_pointer_type());
                        global.set_initializer(&ptr_val.as_basic_value_enum());
                    } else {
                        // 其他类型暂不支持计算初始化 (比如 Struct 常量，需要 const struct builder)
                        //?: TODO: 支持更多的计算初始化
                        panic!(
                            "Global variable has computed value but type {:?} is not supported for auto-init yet",
                            llvm_ty
                        );
                    }
                } else {
                    panic!(
                        "Global initializer is too complex (not a literal, and analyzer couldn't eval it)."
                    );
                }
            } else {
                // 如果没有初始化值，设置为 Zero Initializer (.bss)
                global.set_initializer(&llvm_ty.const_zero());
            }
        }

        // 6. 注册到符号表
        self.globals.insert(id, global.as_pointer_value());
    }

    // 辅助函数：编译 sizeof
    fn compile_sizeof(&self, target_type: &Type) -> Result<BasicValueEnum<'ctx>, String> {
        // 1. 从 Analyzer 获取类型键
        let type_key = self.get_resolved_type(target_type.id);

        // 2. 编译为 LLVM Type
        let llvm_ty = self
            .compile_type(&type_key)
            .ok_or_else(|| format!("Cannot compile type {:?} for sizeof", type_key))?;

        // 3. 获取大小 (LLVM ConstantExpr)
        // size_of() 返回的是一个 IntValue (通常是 i64)，代表字节数
        match llvm_ty.size_of() {
            Some(size_val) => Ok(size_val.as_basic_value_enum()),
            None => Err("Type does not have a determinable size (e.g. void)".into()),
        }
    }

    fn compile_pointer_binary(
        &self,
        l_ptr: PointerValue<'ctx>,
        op: BinaryOperator,
        rhs_val: BasicValueEnum<'ctx>,
        lhs_type_key: &TypeKey, // 需要类型信息来做指针加减
    ) -> Result<BasicValueEnum<'ctx>, String> {
        match op {
            // === 指针比较 (Ptr == Ptr) ===
            // 支持 curr != NULL_NODE
            BinaryOperator::Equal | BinaryOperator::NotEqual => {
                if !rhs_val.is_pointer_value() {
                    return Err("Cannot compare pointer with non-pointer".into());
                }
                let r_ptr = rhs_val.into_pointer_value();

                // 将指针转为整数 (i64/usize) 进行比较
                let i64_type = self.context.i64_type();
                let l_int = self
                    .builder
                    .build_ptr_to_int(l_ptr, i64_type, "lhs_p2i")
                    .unwrap();
                let r_int = self
                    .builder
                    .build_ptr_to_int(r_ptr, i64_type, "rhs_p2i")
                    .unwrap();

                let pred = match op {
                    BinaryOperator::Equal => IntPredicate::EQ,
                    BinaryOperator::NotEqual => IntPredicate::NE,
                    _ => unreachable!(),
                };

                Ok(self
                    .builder
                    .build_int_compare(pred, l_int, r_int, "ptr_cmp")
                    .unwrap()
                    .into())
            }

            // === 指针加法 (Ptr + Int) ===
            // 支持 ptr = ptr + 1 (移动 sizeof(T))
            BinaryOperator::Add => {
                if !rhs_val.is_int_value() {
                    return Err("Pointer arithmetic requires integer RHS".into());
                }
                let r_int = rhs_val.into_int_value();

                // 1. 获取指针指向的类型 (Pointee Type)
                let pointee_type = match lhs_type_key {
                    TypeKey::Pointer(inner, _) => self.compile_type(inner),
                    TypeKey::Array(inner, _) => self.compile_type(inner),
                    _ => {
                        return Err(format!(
                            "LHS is not a pointer/array type: {:?}",
                            lhs_type_key
                        ));
                    }
                }
                .ok_or("Failed to compile pointee type for arithmetic")?;

                // 2. 生成 GEP (GetElementPtr)
                // LLVM 的 GEP 强类型：ptr + N 意味着内存地址增加 N * sizeof(pointee_type)
                let new_ptr = unsafe {
                    self.builder
                        .build_gep(pointee_type, l_ptr, &[r_int], "ptr_add")
                        .map_err(|_| "GEP failed")?
                };

                Ok(new_ptr.as_basic_value_enum())
            }

            // === 指针减法 ===
            BinaryOperator::Subtract => {
                if rhs_val.is_int_value() {
                    // Case A: Ptr - Int => Ptr + (-Int)
                    let r_int = rhs_val.into_int_value();
                    let neg_r = self.builder.build_int_neg(r_int, "neg_offset").unwrap();

                    let pointee_type = match lhs_type_key {
                        TypeKey::Pointer(inner, _) => self.compile_type(inner),
                        TypeKey::Array(inner, _) => self.compile_type(inner),
                        _ => return Err("Not a pointer".into()),
                    }
                    .unwrap();

                    let new_ptr = unsafe {
                        self.builder
                            .build_gep(pointee_type, l_ptr, &[neg_r], "ptr_sub")
                            .unwrap()
                    };
                    Ok(new_ptr.as_basic_value_enum())
                } else if rhs_val.is_pointer_value() {
                    // Case B: Ptr - Ptr -> Int (两个指针的距离)
                    // 返回的是元素个数 (i64)，不是字节数
                    let r_ptr = rhs_val.into_pointer_value();

                    let pointee_type = match lhs_type_key {
                        TypeKey::Pointer(inner, _) => self.compile_type(inner),
                        _ => return Err("Ptr diff need pointer".into()),
                    }
                    .unwrap();

                    let diff = self
                        .builder
                        .build_ptr_diff(pointee_type, l_ptr, r_ptr, "diff")
                        .unwrap();
                    Ok(diff.into())
                } else {
                    Err("Invalid type for pointer sub".into())
                }
            }

            _ => Err(format!("Binary op {:?} not supported for pointers", op)),
        }
    }

    // 判断是否是整数类 (包括 Bool, Char, Unit)
    fn is_integer_kind(&self, p: &PrimitiveType) -> bool {
        match p {
            PrimitiveType::F32 | PrimitiveType::F64 => false,
            _ => true,
        }
    }

    // 判断是否是浮点类
    fn is_float_kind(&self, p: &PrimitiveType) -> bool {
        match p {
            PrimitiveType::F32 | PrimitiveType::F64 => true,
            _ => false,
        }
    }

    // 获取 AST 节点的类型，并根据当前泛型上下文进行替换
    fn get_resolved_type(&self, id: NodeId) -> TypeKey {
        let raw_ty = self.analysis.types.get(&id).expect("Type missing in analyzer");
        
        if let Some((def_id, args)) = &self.generic_context {
            // 如果当前在泛型函数里，执行替换 (T -> i32)
            self.analysis.substitute_generics(raw_ty, *def_id, args)
        } else {
            raw_ty.clone()
        }
    }
}

trait BasicTypeEnumExt<'ctx> {
    fn get_alignment(&self) -> IntValue<'ctx>;
}

impl<'ctx> BasicTypeEnumExt<'ctx> for BasicTypeEnum<'ctx> {
    fn get_alignment(&self) -> IntValue<'ctx> {
        match self {
            BasicTypeEnum::ArrayType(t) => t.get_alignment(),
            BasicTypeEnum::FloatType(t) => t.get_alignment(),
            BasicTypeEnum::IntType(t) => t.get_alignment(),
            BasicTypeEnum::PointerType(t) => t.get_alignment(),
            BasicTypeEnum::StructType(t) => t.get_alignment(),
            BasicTypeEnum::VectorType(t) => t.get_alignment(),
            BasicTypeEnum::ScalableVectorType(t) => t.get_alignment(),
        }
    }
}
