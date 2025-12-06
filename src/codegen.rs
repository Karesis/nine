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

use crate::analyzer::{AnalysisContext, DefId, MethodInfo, TypeKey};
use crate::ast::*;
use crate::analyzer::DefKind;

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

    pub variables: HashMap<DefId, PointerValue<'ctx>>,

    pub globals: HashMap<DefId, PointerValue<'ctx>>,

    pub functions: HashMap<String, FunctionValue<'ctx>>,

    pub struct_types: HashMap<String, StructType<'ctx>>,

    pub struct_field_indices: HashMap<String, HashMap<String, u32>>,

    pub loop_stack: Vec<(BasicBlock<'ctx>, BasicBlock<'ctx>)>,
    current_fn: Option<FunctionValue<'ctx>>,

    generic_context: Option<(DefId, Vec<TypeKey>)>,

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
            function_index,
            generic_context: None,
        }
    }

    fn build_function_index(items: &'a [Item], index: &mut HashMap<DefId, &'a FunctionDefinition>) {
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
                ItemKind::ModuleDecl {
                    items: Some(subs), ..
                } => {
                    Self::build_function_index(subs, index);
                }
                _ => {}
            }
        }
    }

    pub fn compile_type(&self, key: &TypeKey) -> Option<BasicTypeEnum<'ctx>> {
        match key {
            TypeKey::Primitive(prim) => Some(self.compile_primitive_type(prim)),

            TypeKey::Instantiated { def_id, .. } => {
                // 1. 查 Analyzer 获取定义种类
                let kind = self.analysis.def_kind.get(def_id).cloned().unwrap_or(DefKind::Struct);

                match kind {
                    // Case A: Enum 是整数
                    DefKind::Enum => {
                        let underlying = self.analysis.enum_underlying_types.get(def_id)
                            .cloned().unwrap_or(PrimitiveType::I32);
                        Some(self.compile_primitive_type(&underlying))
                    }
                    
                    // Case B: Struct 和 Union 都是 LLVM Struct
                    DefKind::Struct | DefKind::Union => {
                        let mangled_name = self.analysis.get_mangling_type_name(key);
                        if let Some(st) = self.struct_types.get(&mangled_name) {
                            Some(st.as_basic_type_enum())
                        } else {
                            panic!("ICE: Struct/Union type '{}' not pre-generated.", mangled_name);
                        }
                    }
                }
            }

            TypeKey::Pointer(_, _) => Some(
                self.context
                    .ptr_type(AddressSpace::default())
                    .as_basic_type_enum(),
            ),

            TypeKey::Array(inner, size) => {
                let inner_ty = self.compile_type(inner)?;
                Some(inner_ty.array_type(*size as u32).as_basic_type_enum())
            }

            TypeKey::Function { .. } => Some(
                self.context
                    .ptr_type(AddressSpace::default())
                    .as_basic_type_enum(),
            ),

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

            TypeKey::GenericParam(id) => {
                panic!("ICE: GenericParam({:?}) reached Codegen layer!", id);
            }

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
                if self.analysis.target.ptr_byte_width == 4 {
                    self.context.i32_type().as_basic_type_enum()
                } else {
                    self.context.i64_type().as_basic_type_enum()
                }
            }
            PrimitiveType::F32 => self.context.f32_type().as_basic_type_enum(),
            PrimitiveType::F64 => self.context.f64_type().as_basic_type_enum(),
            PrimitiveType::Bool => self.context.bool_type().as_basic_type_enum(),

            PrimitiveType::Unit => self.context.struct_type(&[], false).as_basic_type_enum(),
        }
    }

    pub fn compile_program(&mut self, program: &Program) {
        self.declare_concrete_structs();

        self.fill_concrete_struct_bodies();

        for &def_id in &self.analysis.non_generic_functions {
            if let Some(func_def) = self.function_index.get(&def_id) {
                self.declare_function(func_def);
            }
        }

        let generic_funcs: Vec<_> = self.analysis.concrete_functions.iter().cloned().collect();
        for (def_id, args) in &generic_funcs {
            self.declare_monomorphized_function(*def_id, args);
        }

        self.compile_globals(&program.items);

        let non_generic_ids: Vec<_> = self
            .analysis
            .non_generic_functions
            .iter()
            .cloned()
            .collect();

        for def_id in non_generic_ids {
            if let Some(&func_def) = self.function_index.get(&def_id) {
                self.generic_context = None;

                if let Err(e) = self.compile_function(func_def) {
                    panic!("Failed to compile function '{}': {}", func_def.name.name, e);
                }
            }
        }

        let generic_funcs: Vec<_> = self.analysis.concrete_functions.iter().cloned().collect();

        for (def_id, args) in generic_funcs {
            self.compile_monomorphized_function(def_id, &args);
        }
    }

    fn declare_concrete_structs(&mut self) {
        for key in &self.analysis.concrete_structs {
            if let TypeKey::Instantiated { def_id, .. } = key {
                let kind = self.analysis.def_kind.get(def_id).cloned().unwrap_or(DefKind::Struct);
                
                if matches!(kind, DefKind::Struct | DefKind::Union) {
                    let mangled_name = self.analysis.get_mangling_type_name(key);
                    
                    // 避免重复创建
                    if !self.struct_types.contains_key(&mangled_name) {
                        let st = self.context.opaque_struct_type(&mangled_name);
                        self.struct_types.insert(mangled_name, st);
                    }
                }
            }
        }
    }

    fn fill_concrete_struct_bodies(&mut self) {
        // 还是遍历 concrete_structs 比较稳妥，因为我们需要 DefId 来判断 Kind
        let structs: Vec<_> = self.analysis.concrete_structs.iter().cloned().collect();

        for key in structs {
            if let TypeKey::Instantiated { def_id, args } = &key {
                let kind = self.analysis.def_kind.get(def_id).cloned().unwrap_or(DefKind::Struct);
                
                // Enum 跳过
                if kind == DefKind::Enum { continue; }

                let mangled_name = self.analysis.get_mangling_type_name(&key);
                let st = *self.struct_types.get(&mangled_name).expect("Struct type missing");

                // 检查是否已经填过 Body (防止重复)
                if !st.is_opaque() { continue; }

                match kind {
                    // === Struct ===
                    DefKind::Struct => {
                        // 查 instantiated_structs 拿字段列表
                        if let Some(fields) = self.analysis.instantiated_structs.get(&mangled_name) {
                            let mut llvm_field_types = Vec::new();
                            let mut indices = HashMap::new();

                            for (i, (field_name, field_type_key)) in fields.iter().enumerate() {
                                let llvm_ty = self.compile_type(field_type_key).expect("Field type compile failed");
                                llvm_field_types.push(llvm_ty);
                                indices.insert(field_name.clone(), i as u32);
                            }

                            st.set_body(&llvm_field_types, false);
                            self.struct_field_indices.insert(mangled_name, indices);
                        }
                    }
                    
                    // === Union ===
                    DefKind::Union => {
                        // Union 在 LLVM 中通常表示为一个字节数组，大小等于最大对齐后的大小
                        // 我们利用 Analyzer 算好的 Layout
                        if let Some(layout) = self.analysis.get_type_layout(&key) {
                            let size = layout.size;
                            // 生成 [size x i8]
                            let byte_type = self.context.i8_type();
                            let array_type = byte_type.array_type(size as u32);
                            
                            st.set_body(&[array_type.as_basic_type_enum()], false); // packed=false
                            
                            // Union 不需要 field indices，因为所有字段偏移量都是 0
                            // 我们存一个空的 Map 或者根本不存
                            self.struct_field_indices.insert(mangled_name, HashMap::new());
                        }
                    }
                    
                    _ => {}
                }
            }
        }
    }

    fn compile_struct_body(&mut self, mangled_name: &str, fields: &[(String, TypeKey)]) {
        let st = *self.struct_types.get(mangled_name).unwrap();

        let mut llvm_fields = Vec::new();
        let mut indices = HashMap::new();

        for (i, (field_name, field_ty)) in fields.iter().enumerate() {
            let llvm_ty = self
                .compile_type(field_ty)
                .expect("Field type compile failed");
            llvm_fields.push(llvm_ty);
            indices.insert(field_name.clone(), i as u32);
        }

        st.set_body(&llvm_fields, false);
        self.struct_field_indices
            .insert(mangled_name.to_string(), indices);
    }

    fn declare_monomorphized_function(&mut self, def_id: DefId, args: &[TypeKey]) {
        let fn_mangled_name = self.analysis.get_mangled_function_name(def_id, args);

        if self.functions.contains_key(&fn_mangled_name) {
            return;
        }

        let raw_fn_type = self.get_resolved_type(def_id);

        let actual_fn_type_key = self
            .analysis
            .substitute_generics(&raw_fn_type, def_id, args);

        let llvm_fn_type = self
            .compile_function_type_signature(&actual_fn_type_key)
            .unwrap();

        let val = self
            .module
            .add_function(&fn_mangled_name, llvm_fn_type, None);
        self.functions.insert(fn_mangled_name, val);
    }

    fn compile_monomorphized_function(&mut self, def_id: DefId, args: &[TypeKey]) {
        let func_def = *self
            .function_index
            .get(&def_id)
            .expect("Generic function def missing in index");

        let fn_mangled_name = self.analysis.get_mangled_function_name(def_id, args);

        let function = *self
            .functions
            .get(&fn_mangled_name)
            .expect("Function proto not declared");

        self.generic_context = Some((def_id, args.to_vec()));
        self.current_fn = Some(function);

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        self.variables.clear();
        for (i, param) in func_def.params.iter().enumerate() {
            let arg_val = function.get_nth_param(i as u32).unwrap();
            arg_val.set_name(&param.name.name);

            let param_ty_key = self.get_resolved_type(param.id);
            let llvm_ty = self.compile_type(&param_ty_key).unwrap();

            let alloca = self
                .builder
                .build_alloca(llvm_ty, &param.name.name)
                .unwrap();
            self.builder.build_store(alloca, arg_val).ok();
            self.variables.insert(param.id, alloca);
        }

        if let Some(body) = &func_def.body {
            self.compile_block(body).ok();
        }

        if !self.block_terminated(func_def.body.as_ref().unwrap()) {
            if func_def.return_type.is_none() {
                self.builder.build_return(None).ok();
            }
        }

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

    fn compile_globals(&mut self, items: &[Item]) {
        for item in items {
            match &item.kind {
                ItemKind::GlobalVariable(def) => {
                    self.compile_global_variable(item.id, def);
                }

                ItemKind::ModuleDecl {
                    items: Some(subs), ..
                } => {
                    self.compile_globals(subs);
                }

                _ => {}
            }
        }
    }

    fn get_field_index(&self, mangled_name: &str, field_name: &str) -> Option<u32> {
        self.struct_field_indices
            .get(mangled_name)
            .and_then(|indices| indices.get(field_name).cloned())
    }

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

    fn declare_function(&mut self, func: &FunctionDefinition) {
        let param_types = func
            .params
            .iter()
            .map(|p| {
                let key = self.get_resolved_type(p.id);
                self.compile_type(&key).unwrap().into()
            })
            .collect::<Vec<_>>();

        let ret_key = self.get_resolved_type(func.id);

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

        let fn_name = self
            .analysis
            .mangled_names
            .get(&func.id)
            .cloned()
            .unwrap_or(func.name.name.clone());

        let val = self.module.add_function(&fn_name, fn_type, None);

        self.functions.insert(fn_name, val);
    }

    pub fn compile_function(&mut self, func: &FunctionDefinition) -> Result<(), String> {
        trace!("Compiling function: {}", func.name.name);
        if func.body.is_none() {
            return Ok(());
        }
        let body_ref = func.body.as_ref().unwrap();

        let fn_mangled_name = self.analysis.get_mangled_function_name(func.id, &[]);

        let function = *self
            .functions
            .get(&fn_mangled_name)
            .ok_or_else(|| format!("Function proto '{}' missing", fn_mangled_name))?;

        self.current_fn = Some(function);

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        self.variables.clear();

        for (i, param) in func.params.iter().enumerate() {
            let arg_val = function.get_nth_param(i as u32).unwrap();
            arg_val.set_name(&param.name.name);

            let param_ty_key = self.get_resolved_type(param.id);
            let llvm_ty = self.compile_type(&param_ty_key).unwrap();

            let alloca = self
                .builder
                .build_alloca(llvm_ty, &param.name.name)
                .map_err(|_| "Alloca failed")?;
            self.builder.build_store(alloca, arg_val).ok();

            self.variables.insert(param.id, alloca);
        }

        self.compile_block(body_ref)?;

        if !self.block_terminated(body_ref) {
            if func.return_type.is_none() {
                self.builder.build_return(None).ok();
            } else {
                return Err(format!(
                    "Function '{}' missing return statement",
                    func.name.name
                ));
            }
        }

        self.current_fn = None;

        Ok(())
    }

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
                let type_key = self.get_resolved_type(stmt.id);
                let llvm_ty = self.compile_type(&type_key).unwrap();

                let ptr = self
                    .builder
                    .build_alloca(llvm_ty, &name.name)
                    .map_err(|_| "Alloca failed")?;
                self.variables.insert(stmt.id, ptr);

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

                let then_bb = self.context.append_basic_block(parent, "if_then");
                let else_bb = self.context.append_basic_block(parent, "if_else");
                let merge_bb = self.context.append_basic_block(parent, "if_merge");

                let cond_val = self.compile_expr(condition)?.into_int_value();

                let actual_else = if else_branch.is_some() {
                    else_bb
                } else {
                    merge_bb
                };

                self.builder
                    .build_conditional_branch(cond_val, then_bb, actual_else)
                    .ok();

                self.builder.position_at_end(then_bb);
                self.compile_block(then_block)?;

                if self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_none()
                {
                    self.builder.build_unconditional_branch(merge_bb).ok();
                }

                if let Some(else_stmt) = else_branch {
                    self.builder.position_at_end(else_bb);
                    self.compile_stmt(else_stmt)?;

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

                self.builder.position_at_end(cond_bb);
                let c = self.compile_expr(condition)?.into_int_value();
                self.builder
                    .build_conditional_branch(c, body_bb, end_bb)
                    .ok();

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

                let target_val = self.compile_expr(target)?;
                if !target_val.is_int_value() {
                    return Err("Switch on non-integer types not implemented yet".into());
                }
                let target_int = target_val.into_int_value();

                let merge_bb = self.context.append_basic_block(parent, "switch_merge");
                let default_bb = self.context.append_basic_block(parent, "switch_default");

                let mut collected_cases = Vec::new();

                let mut case_blocks_to_compile = Vec::new();

                for case in cases {
                    let case_bb = self.context.append_basic_block(parent, "switch_case");

                    case_blocks_to_compile.push((case_bb, &case.body));

                    for pattern in &case.patterns {
                        let pattern_val = self.compile_expr(pattern)?;
                        if !pattern_val.is_int_value() {
                            return Err("Switch case pattern must be integer".into());
                        }

                        collected_cases.push((pattern_val.into_int_value(), case_bb));
                    }
                }

                self.builder
                    .build_switch(target_int, default_bb, &collected_cases)
                    .map_err(|_| "Build switch failed")?;

                for (bb, body) in case_blocks_to_compile {
                    self.builder.position_at_end(bb);
                    self.compile_block(body)?;

                    if !self.block_terminated(body) {
                        self.builder.build_unconditional_branch(merge_bb).ok();
                    }
                }

                self.builder.position_at_end(default_bb);
                if let Some(block) = default_case {
                    self.compile_block(block)?;
                    if !self.block_terminated(block) {
                        self.builder.build_unconditional_branch(merge_bb).ok();
                    }
                } else {
                    self.builder.build_unconditional_branch(merge_bb).ok();
                }

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

    pub fn compile_expr(&mut self, expr: &Expression) -> Result<BasicValueEnum<'ctx>, String> {
        trace!("    Compiling expr: {:?}", expr.kind);
        match &expr.kind {
            ExpressionKind::Literal(lit) => {
                let type_key = self.get_resolved_type(expr.id);
                self.compile_literal(lit, &type_key)
            }

            ExpressionKind::Binary { lhs, op, rhs } => self.compile_binary(lhs, *op, rhs),

            ExpressionKind::Unary { op, operand } => match op {
                UnaryOperator::AddressOf => {
                    let ptr = self.compile_expr_ptr(operand)?;
                    Ok(ptr.as_basic_value_enum())
                }

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

                UnaryOperator::Not => {
                    let val = self.compile_expr(operand)?;
                    if val.is_int_value() {
                        Ok(self
                            .builder
                            .build_not(val.into_int_value(), "not")
                            .unwrap()
                            .into())
                    } else {
                        Err("Not operand must be integer or boolean".into())
                    }
                }
            },

            ExpressionKind::Index { .. } | ExpressionKind::FieldAccess { .. } => {
                trace!("    [Load] 1. Compiling ptr...");
                let ptr = self.compile_expr_ptr(expr)?;
                trace!("    [Load] 1. Ptr compiled. Is Null? {}", ptr.is_null());

                trace!(
                    "    [Load] 2. Getting type key for Expr ID {:?}...",
                    expr.id
                );
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
                let type_key = self.get_resolved_type(expr.id);

                let def_id = *self
                    .analysis
                    .path_resolutions
                    .get(&expr.id)
                    .expect("Path not resolved");

                let is_global_function = self.function_index.contains_key(&def_id);

                if is_global_function {
                    let generic_args = self
                        .analysis
                        .node_generic_args
                        .get(&expr.id)
                        .cloned()
                        .unwrap_or_default();

                    let fn_mangled_name = self
                        .analysis
                        .get_mangled_function_name(def_id, &generic_args);

                    let fn_val = *self.functions.get(&fn_mangled_name).unwrap_or_else(|| {
                        panic!(
                            "Function '{}' not found/compiled when used as value",
                            fn_mangled_name
                        )
                    });

                    return Ok(fn_val
                        .as_global_value()
                        .as_pointer_value()
                        .as_basic_value_enum());
                }

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
                let struct_key = self.get_resolved_type(expr.id);

                let mangled_name = self.analysis.get_mangling_type_name(&struct_key);

                let struct_type = *self.struct_types.get(&mangled_name).unwrap_or_else(|| {
                    panic!("Struct type '{}' not found in codegen", mangled_name)
                });

                let index_map = self
                    .struct_field_indices
                    .get(&mangled_name)
                    .unwrap_or_else(|| panic!("Indices for '{}' not found", mangled_name))
                    .clone();

                let mut struct_val = struct_type.get_undef();

                for field in fields {
                    let val = self.compile_expr(&field.value)?;

                    let idx = *index_map
                        .get(&field.field_name.name)
                        .ok_or_else(|| format!("Field '{}' not found", field.field_name.name))?;

                    eprintln!(
                        "[Codegen Debug] Inserting field '{}' at index {} into struct '{}'",
                        field.field_name.name, idx, mangled_name
                    );
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

            ExpressionKind::MethodCall {
                receiver,
                method_name,
                arguments,
            } => self.compile_method_call_dispatch(expr.id, receiver, method_name, arguments),

            ExpressionKind::StaticAccess {
                target: _,
                member: _,
            } => {
                let ty = self.get_resolved_type(expr.id);

                if let TypeKey::IntegerLiteral(val) = ty {
                    Ok(self
                        .context
                        .i64_type()
                        .const_int(val, false)
                        .as_basic_value_enum())
                } else {
                    Err("Static access (non-enum/non-const) not implemented".into())
                }
            }

            ExpressionKind::SizeOf(target_type) => self.compile_sizeof(target_type),

            ExpressionKind::AlignOf(target_type) => {
                let type_key = self
                    .analysis
                    .types
                    .get(&target_type.id)
                    .ok_or("AlignOf target type not resolved")?;

                let llvm_ty = self
                    .compile_type(type_key)
                    .ok_or("Cannot compile type for alignof")?;

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

            ExpressionKind::FieldAccess { receiver, field_name } => {
                // 编译 Receiver 指针
                let ptr = self.compile_expr_ptr(receiver)?;
                let recv_type_key = self.get_resolved_type(receiver.id);

                if let TypeKey::Instantiated { def_id, .. } = &recv_type_key {
                    let kind = self.analysis.def_kind.get(def_id).cloned().unwrap_or(DefKind::Struct);
                    
                    match kind {
                        // === Case A: Struct (GEP) ===
                        DefKind::Struct => {
                            let mangled_name = self.analysis.get_mangling_type_name(&recv_type_key);
                            let idx = self.get_field_index(&mangled_name, &field_name.name)
                                .ok_or_else(|| format!("Field not found"))?;
                            let st_ty = *self.struct_types.get(&mangled_name).unwrap();

                            let field_ptr = self.builder.build_struct_gep(st_ty, ptr, idx, "gep").map_err(|_| "GEP failed")?;
                            Ok(field_ptr)
                        }

                        // === Case B: Union ===
                        DefKind::Union => {
                            Ok(ptr) 
                        }
                        
                        DefKind::Enum => Err("Cannot access field of Enum".into()),
                    }
                } else {
                    Err(format!("Field access on non-composite type: {:?}", recv_type_key))
                }
            }

            ExpressionKind::Index { target, index } => {
                let ptr = self.compile_expr_ptr(target)?;
                let idx = self.compile_expr(index)?.into_int_value();

                let target_key = self.get_resolved_type(target.id);

                match &target_key {
                    TypeKey::Array(_inner, _) => {
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
                        Ok(p)
                    }
                    TypeKey::Pointer(inner, _) => {
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

    fn compile_method_call_dispatch(
        &mut self,
        expr_id: NodeId,
        receiver: &Expression,
        method_name: &Identifier,
        arguments: &[Expression],
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let receiver_ty_key = self.get_resolved_type(receiver.id);

        let is_standard_method = self
            .find_method_info(&receiver_ty_key, &method_name.name)
            .is_some();

        if is_standard_method {
            return self.compile_standard_method_call(
                expr_id,
                receiver,
                method_name,
                arguments,
                &receiver_ty_key,
            );
        } else {
            return self.compile_field_fn_ptr_call(
                receiver,
                method_name,
                arguments,
                &receiver_ty_key,
            );
        }
    }

    fn find_method_info(&self, receiver_ty: &TypeKey, name: &str) -> Option<&MethodInfo> {
        if let Some(methods) = self.analysis.method_registry.get(receiver_ty) {
            if let Some(info) = methods.get(name) {
                return Some(info);
            }
        }

        if let TypeKey::Instantiated { def_id, .. } = receiver_ty {
            for (key, methods) in &self.analysis.method_registry {
                if let TypeKey::Instantiated { def_id: k_id, .. } = key {
                    if k_id == def_id {
                        if let Some(info) = methods.get(name) {
                            return Some(info);
                        }
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
        receiver_ty: &TypeKey,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let method_info = self
            .find_method_info(receiver_ty, &method_name.name)
            .ok_or("Method info missing in codegen")?;

        let method_args = self
            .analysis
            .node_generic_args
            .get(&expr_id)
            .cloned()
            .unwrap_or_default();

        let fn_name = self
            .analysis
            .get_mangled_function_name(method_info.def_id, &method_args);
        let fn_val = *self
            .functions
            .get(&fn_name)
            .ok_or_else(|| format!("Function '{}' not compiled", fn_name))?;

        let mut compiled_args = Vec::new();
        compiled_args.push(self.compile_expr(receiver)?.into());
        for arg in arguments {
            compiled_args.push(self.compile_expr(arg)?.into());
        }

        let call = self
            .builder
            .build_call(fn_val, &compiled_args, "call")
            .map_err(|_| "Call failed")?;
        self.handle_call_return(call)
    }

    /// 路径 2: 编译字段函数指针调用
    fn compile_field_fn_ptr_call(
        &mut self,
        receiver: &Expression,
        method_name: &Identifier,
        arguments: &[Expression],
        receiver_ty: &TypeKey,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let struct_key = match receiver_ty {
            TypeKey::Pointer(inner, _) => *inner.clone(),
            _ => receiver_ty.clone(),
        };
        let mangled_struct_name = self.analysis.get_mangling_type_name(&struct_key);

        let idx = self
            .get_field_index(&mangled_struct_name, &method_name.name)
            .ok_or_else(|| {
                format!(
                    "Field '{}' not found in struct '{}'",
                    method_name.name, mangled_struct_name
                )
            })?;

        let struct_ptr = self.compile_expr_ptr(receiver)?;
        let st_ty = *self
            .struct_types
            .get(&mangled_struct_name)
            .expect("Struct type missing");

        let field_ptr = unsafe {
            self.builder
                .build_struct_gep(st_ty, struct_ptr, idx, "fn_field_gep")
                .map_err(|_| "GEP failed")?
        };

        let fields_list = self
            .analysis
            .instantiated_structs
            .get(&mangled_struct_name)
            .expect("Instantiated struct missing");
        let (_, field_type_key) = &fields_list[idx as usize];
        let llvm_field_type = self.compile_type(field_type_key).unwrap();

        let fn_ptr_val = self
            .builder
            .build_load(llvm_field_type, field_ptr, "fn_ptr_load")
            .map_err(|_| "Load failed")?
            .into_pointer_value();

        let mut compiled_args = Vec::new();
        for arg in arguments {
            compiled_args.push(self.compile_expr(arg)?.into());
        }

        let fn_type = self.compile_function_type_signature(field_type_key)?;
        let call = self
            .builder
            .build_indirect_call(fn_type, fn_ptr_val, &compiled_args, "indirect_call")
            .map_err(|_| "Indirect call failed")?;

        self.handle_call_return(call)
    }

    fn compile_literal(
        &self,
        lit: &Literal,
        type_key: &TypeKey,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        match lit {
            Literal::Integer(val) => match type_key {
                TypeKey::Primitive(p) => {
                    let ty = self.compile_primitive_type(p).into_int_type();
                    Ok(ty.const_int(*val, false).as_basic_value_enum())
                }

                _ => Ok(self
                    .context
                    .i64_type()
                    .const_int(*val, false)
                    .as_basic_value_enum()),
            },
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
        if matches!(op, BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr) {
            return self.compile_short_circuit_binary(lhs, op, rhs);
        }

        let lhs_val = self.compile_expr(lhs)?;
        let rhs_val = self.compile_expr(rhs)?;

        if lhs_val.is_float_value() {
            return self.compile_float_binary(
                lhs_val.into_float_value(),
                op,
                rhs_val.into_float_value(),
            );
        }

        if lhs_val.is_pointer_value() {
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

        if lhs_val.is_int_value() && rhs_val.is_int_value() {
            let l = lhs_val.into_int_value();
            let r = rhs_val.into_int_value();

            let type_key = self
                .analysis
                .types
                .get(&lhs.id)
                .ok_or("Type missing for binary op")?;
            let is_signed = self.is_signed(type_key);

            return match op {
                BinaryOperator::Add => Ok(self.builder.build_int_add(l, r, "add").unwrap().into()),
                BinaryOperator::Subtract => {
                    Ok(self.builder.build_int_sub(l, r, "sub").unwrap().into())
                }
                BinaryOperator::Multiply => {
                    Ok(self.builder.build_int_mul(l, r, "mul").unwrap().into())
                }

                BinaryOperator::BitwiseAnd => {
                    Ok(self.builder.build_and(l, r, "and").unwrap().into())
                }
                BinaryOperator::BitwiseOr => Ok(self.builder.build_or(l, r, "or").unwrap().into()),
                BinaryOperator::BitwiseXor => {
                    Ok(self.builder.build_xor(l, r, "xor").unwrap().into())
                }

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

                BinaryOperator::ShiftLeft => {
                    Ok(self.builder.build_left_shift(l, r, "shl").unwrap().into())
                }
                BinaryOperator::ShiftRight => Ok(self
                    .builder
                    .build_right_shift(l, r, is_signed, "shr")
                    .unwrap()
                    .into()),

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

        if !lhs_val.is_int_value() {
            return Err("Logical op requires boolean/integer operands".into());
        }
        let lhs_int = lhs_val.into_int_value();

        let i1_type = self.context.bool_type();
        let zero = self.context.i64_type().const_zero();

        let lhs_bool = if lhs_int.get_type().get_bit_width() > 1 {
            self.builder
                .build_int_compare(IntPredicate::NE, lhs_int, zero, "tobool")
                .unwrap()
        } else {
            lhs_int
        };

        let rhs_bb = self.context.append_basic_block(parent, "logic_rhs");
        let merge_bb = self.context.append_basic_block(parent, "logic_merge");

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

        let lhs_end_bb = self.builder.get_insert_block().unwrap();

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

        self.builder.position_at_end(merge_bb);

        let phi = self.builder.build_phi(i1_type, "logic_res").unwrap();

        let true_val = i1_type.const_int(1, false);
        let false_val = i1_type.const_int(0, false);

        match op {
            BinaryOperator::LogicalAnd => {
                phi.add_incoming(&[(&false_val, lhs_end_bb), (&rhs_bool, rhs_end_bb)]);
            }
            BinaryOperator::LogicalOr => {
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

        let mut compiled_args = Vec::new();
        for arg in args {
            compiled_args.push(self.compile_expr(arg)?.into());
        }

        let direct_call_result = if let ExpressionKind::Path(path) = &callee.kind {
            if let Some(&def_id) = self.analysis.path_resolutions.get(&callee.id) {
                let generic_args = self
                    .analysis
                    .node_generic_args
                    .get(&callee.id)
                    .cloned()
                    .unwrap_or_default();
                let fn_mangled_name = self
                    .analysis
                    .get_mangled_function_name(def_id, &generic_args);

                if let Some(fn_val) = self.functions.get(&fn_mangled_name) {
                    Some((fn_val, compiled_args.clone()))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some((fn_val, args)) = direct_call_result {
            let call_site = self
                .builder
                .build_call(*fn_val, &args, "direct_call")
                .unwrap();
            return self.handle_call_return(call_site);
        }

        if let ExpressionKind::StaticAccess { .. } = &callee.kind {
            if let Some(&def_id) = self.analysis.path_resolutions.get(&callee.id) {
                let generic_args = self
                    .analysis
                    .node_generic_args
                    .get(&callee.id)
                    .cloned()
                    .unwrap_or_default();

                let fn_mangled_name = self
                    .analysis
                    .get_mangled_function_name(def_id, &generic_args);

                if let Some(fn_val) = self.functions.get(&fn_mangled_name) {
                    trace!("        Build call to '{}'", fn_mangled_name);
                    let call_site = self
                        .builder
                        .build_call(*fn_val, &compiled_args, "static_call")
                        .map_err(|_| "Static call failed")?;
                    return self.handle_call_return(call_site);
                } else {
                    panic!(
                        "ICE: Static method '{}' resolved but not compiled",
                        fn_mangled_name
                    );
                }
            }
        }

        let fn_ptr_val = self.compile_expr(callee)?.into_pointer_value();

        let callee_type_key = self.get_resolved_type(callee.id);

        let fn_type = self.compile_function_type_signature(&callee_type_key)?;

        let call_site = self
            .builder
            .build_indirect_call(fn_type, fn_ptr_val, &compiled_args, "indirect_call")
            .map_err(|_| "Indirect call failed")?;

        self.handle_call_return(call_site)
    }

    fn handle_call_return(
        &self,
        call_site: inkwell::values::CallSiteValue<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        use inkwell::values::ValueKind;
        match call_site.try_as_basic_value() {
            ValueKind::Basic(val) => Ok(val),
            ValueKind::Instruction(_) => Ok(self
                .context
                .struct_type(&[], false)
                .const_zero()
                .as_basic_value_enum()),
        }
    }

    fn compile_function_type_signature(&self, key: &TypeKey) -> Result<FunctionType<'ctx>, String> {
        match key {
            TypeKey::Function {
                params,
                ret,
                is_variadic,
            } => {
                let mut param_types = Vec::new();
                for param in params {
                    let llvm_ty = self
                        .compile_type(param)
                        .ok_or("Failed to compile param type")?;
                    param_types.push(llvm_ty.into());
                }

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
        self.variables
            .get(&def_id)
            .or_else(|| self.globals.get(&def_id))
            .cloned()
    }

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

            TypeKey::IntegerLiteral(_) => true,

            _ => false,
        }
    }

    fn compile_cast(
        &mut self,
        src_expr: &Expression,
        target_ast_type: &Type,
        cast_expr_id: NodeId,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let src_val = self.compile_expr(src_expr)?;

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

        let target_llvm_ty = self
            .compile_type(target_key)
            .ok_or("Compile target type failed")?;

        match (src_key, target_key) {
            (TypeKey::Primitive(src_p), TypeKey::Primitive(target_p))
                if self.is_integer_kind(&src_p) && self.is_integer_kind(&target_p) =>
            {
                let src_int = src_val.into_int_value();
                let target_int_ty = target_llvm_ty.into_int_type();
                let src_width = src_int.get_type().get_bit_width();
                let target_width = target_int_ty.get_bit_width();

                if src_width > target_width {
                    Ok(self
                        .builder
                        .build_int_truncate(src_int, target_int_ty, "trunc")
                        .unwrap()
                        .into())
                } else if src_width < target_width {
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
                    Ok(src_int.into())
                }
            }

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

            (TypeKey::Pointer(..), TypeKey::Primitive(p)) if self.is_integer_kind(p) => {
                let src_ptr = src_val.into_pointer_value();
                let target_int_ty = target_llvm_ty.into_int_type();
                Ok(self
                    .builder
                    .build_ptr_to_int(src_ptr, target_int_ty, "ptr2int")
                    .unwrap()
                    .into())
            }

            (TypeKey::Primitive(p), TypeKey::Pointer(..)) if self.is_integer_kind(p) => {
                let src_int = src_val.into_int_value();
                let target_ptr_ty = target_llvm_ty.into_pointer_type();
                Ok(self
                    .builder
                    .build_int_to_ptr(src_int, target_ptr_ty, "int2ptr")
                    .unwrap()
                    .into())
            }

            (TypeKey::Pointer(..), TypeKey::Pointer(..)) => {
                let src_ptr = src_val.into_pointer_value();
                let target_ptr_ty = target_llvm_ty.into_pointer_type();
                Ok(self
                    .builder
                    .build_pointer_cast(src_ptr, target_ptr_ty, "bitcast")
                    .unwrap()
                    .into())
            }

            (TypeKey::Array(..), TypeKey::Pointer(..)) => {
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

            _ => Err(format!(
                "Unsupported cast from {:?} to {:?}",
                src_key, target_key
            )),
        }
    }

    fn compile_global_variable(&mut self, id: DefId, def: &GlobalDefinition) {
        let ty_key = self.get_resolved_type(id);
        let llvm_ty = self.compile_type(&ty_key).unwrap();

        let name = if def.is_extern {
            def.name.name.clone()
        } else {
            self.analysis
                .mangled_names
                .get(&id)
                .cloned()
                .unwrap_or(def.name.name.clone())
        };

        let global = self
            .module
            .add_global(llvm_ty, Some(AddressSpace::default()), &name);

        if def.is_extern {
            global.set_linkage(Linkage::External);
        } else {
            global.set_constant(def.modifier == Mutability::Constant);

            if let Some(init) = &def.initializer {
                if let ExpressionKind::Literal(lit) = &init.kind {
                    let lit_ty = self.get_resolved_type(init.id);
                    let val = self
                        .compile_literal(lit, &lit_ty)
                        .expect("Literal compile failed");
                    global.set_initializer(&val);
                } else if let Some(&const_val) = self.analysis.constants.get(&id) {
                    if llvm_ty.is_int_type() {
                        let int_ty = llvm_ty.into_int_type();

                        let val = int_ty.const_int(const_val, false);
                        global.set_initializer(&val.as_basic_value_enum());
                    } else if llvm_ty.is_pointer_type() {
                        let i64_ty = self.context.i64_type();
                        let int_val = i64_ty.const_int(const_val, false);

                        let ptr_val = int_val.const_to_pointer(llvm_ty.into_pointer_type());
                        global.set_initializer(&ptr_val.as_basic_value_enum());
                    } else {
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
                global.set_initializer(&llvm_ty.const_zero());
            }
        }

        self.globals.insert(id, global.as_pointer_value());
    }

    fn compile_sizeof(&self, target_type: &Type) -> Result<BasicValueEnum<'ctx>, String> {
        let type_key = self.get_resolved_type(target_type.id);

        let llvm_ty = self
            .compile_type(&type_key)
            .ok_or_else(|| format!("Cannot compile type {:?} for sizeof", type_key))?;

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
        lhs_type_key: &TypeKey,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        match op {
            BinaryOperator::Equal | BinaryOperator::NotEqual => {
                if !rhs_val.is_pointer_value() {
                    return Err("Cannot compare pointer with non-pointer".into());
                }
                let r_ptr = rhs_val.into_pointer_value();

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

            BinaryOperator::Add => {
                if !rhs_val.is_int_value() {
                    return Err("Pointer arithmetic requires integer RHS".into());
                }
                let r_int = rhs_val.into_int_value();

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

                let new_ptr = unsafe {
                    self.builder
                        .build_gep(pointee_type, l_ptr, &[r_int], "ptr_add")
                        .map_err(|_| "GEP failed")?
                };

                Ok(new_ptr.as_basic_value_enum())
            }

            BinaryOperator::Subtract => {
                if rhs_val.is_int_value() {
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

    fn is_integer_kind(&self, p: &PrimitiveType) -> bool {
        match p {
            PrimitiveType::F32 | PrimitiveType::F64 => false,
            _ => true,
        }
    }

    fn is_float_kind(&self, p: &PrimitiveType) -> bool {
        match p {
            PrimitiveType::F32 | PrimitiveType::F64 => true,
            _ => false,
        }
    }

    fn get_resolved_type(&self, id: NodeId) -> TypeKey {
        let raw_ty = self
            .analysis
            .types
            .get(&id)
            .expect("Type missing in analyzer");

        if let Some((def_id, args)) = &self.generic_context {
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
