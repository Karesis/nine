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
use inkwell::values::{BasicValue, BasicValueEnum, FloatValue, FunctionValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use std::collections::HashMap;

use crate::analyzer::{AnalysisContext, DefId, TypeKey};
use crate::ast::*;

/// 代码生成器结构体
pub struct CodeGen<'a, 'ctx> {
    pub context: &'ctx Context,
    pub module: &'a Module<'ctx>,
    pub builder: &'a Builder<'ctx>,

    /// 语义分析的结果 (只读)
    pub analyzer: &'a AnalysisContext,

    /// 符号表：DefId (AST中的ID) -> LLVM 的内存地址 (Alloca出来的栈变量 或 全局变量)
    pub variables: HashMap<DefId, PointerValue<'ctx>>,

    /// 函数表：DefId -> LLVM 函数对象
    pub functions: HashMap<DefId, FunctionValue<'ctx>>,

    /// 结构体类型表：DefId -> LLVM 结构体类型
    pub struct_types: HashMap<DefId, StructType<'ctx>>,

    /// 结构体字段索引映射：DefId -> (Field Name -> Index)
    /// 解决 AST 定义顺序与 LLVM GEP 索引一致性的问题
    pub struct_field_indices: HashMap<DefId, HashMap<String, u32>>,

    /// 循环上下文栈: (break_target, continue_target)
    pub loop_stack: Vec<(BasicBlock<'ctx>, BasicBlock<'ctx>)>,

    /// 当前正在编译的函数 (用于处理 return)
    current_fn: Option<FunctionValue<'ctx>>,
}

impl<'a, 'ctx> CodeGen<'a, 'ctx> {
    pub fn new(
        context: &'ctx Context,
        module: &'a Module<'ctx>,
        builder: &'a Builder<'ctx>,
        analyzer: &'a AnalysisContext,
    ) -> Self {
        Self {
            context,
            module,
            builder,
            analyzer,
            variables: HashMap::new(),
            functions: HashMap::new(),
            struct_types: HashMap::new(),
            struct_field_indices: HashMap::new(),
            loop_stack: Vec::new(),
            current_fn: None,
        }
    }

    // ========================================================================
    // 1. 类型转换系统 (TypeKey -> LLVM Type)
    // ========================================================================

    pub fn compile_type(&self, key: &TypeKey) -> Option<BasicTypeEnum<'ctx>> {
        match key {
            TypeKey::Primitive(prim) => Some(self.compile_primitive_type(prim)),

            TypeKey::Named(def_id) => self
                .struct_types
                .get(def_id)
                .map(|st| st.as_basic_type_enum()),

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

            // 回退逻辑：如果 Analyzer 没能回写类型 (Type Coercion)，只能默认处理
            // 但理想情况下 analyzer 应该已经处理为 Primitive
            //? TODO: 更健壮的报错?
            TypeKey::IntegerLiteral(_) => Some(self.context.i64_type().as_basic_type_enum()),
            TypeKey::FloatLiteral(_) => Some(self.context.f64_type().as_basic_type_enum()),

            TypeKey::Error => None,
        }
    }

    fn compile_primitive_type(&self, prim: &PrimitiveType) -> BasicTypeEnum<'ctx> {
        match prim {
            PrimitiveType::I8 | PrimitiveType::U8 => self.context.i8_type().as_basic_type_enum(),
            PrimitiveType::I16 | PrimitiveType::U16 => self.context.i16_type().as_basic_type_enum(),
            PrimitiveType::I32 | PrimitiveType::U32 => self.context.i32_type().as_basic_type_enum(),
            PrimitiveType::I64 | PrimitiveType::U64 => self.context.i64_type().as_basic_type_enum(),
            PrimitiveType::ISize | PrimitiveType::USize => {
                self.context.i64_type().as_basic_type_enum()
            } //? TODO: 辅助函数根据目标机器选择(暂时假定 64 位)
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
        // Pass 1: 注册结构体名称
        self.register_struct_declarations(&program.items);

        // Pass 2: 填充结构体 Body
        self.fill_struct_bodies(&program.items);

        // Pass 3: 声明所有函数原型
        self.register_function_prototypes(&program.items);

        // Pass 4: 编译函数体
        self.compile_items(&program.items);
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

    // ========================================================================
    // 3. 结构体与字段管理
    // ========================================================================

    pub fn register_struct_declarations(&mut self, items: &[Item]) {
        for item in items {
            match &item.kind {
                ItemKind::StructDecl(def) => {
                    let st = self.context.opaque_struct_type(&def.name.name);
                    self.struct_types.insert(item.id, st);
                }
                ItemKind::ModuleDecl {
                    items: Some(subs), ..
                } => self.register_struct_declarations(subs),
                _ => {}
            }
        }
    }

    pub fn fill_struct_bodies(&mut self, items: &[Item]) {
        for item in items {
            match &item.kind {
                ItemKind::StructDecl(def) => {
                    let st = *self.struct_types.get(&item.id).unwrap();
                    let mut field_types = Vec::new();
                    let mut indices = HashMap::new();
                    for (i, field) in def.fields.iter().enumerate() {
                        // 从 analyzer 查类型 (Analyzer 存的是 Map<Name, TypeKey>)
                        let field_map = self
                            .analyzer
                            .struct_fields
                            .get(&item.id)
                            .expect("Analyzer missed struct");
                        let type_key = field_map
                            .get(&field.name.name)
                            .expect("Analyzer missed field");

                        let llvm_ty = self.compile_type(type_key).expect("Compile type failed");
                        field_types.push(llvm_ty);
                        indices.insert(field.name.name.clone(), i as u32);
                    }

                    st.set_body(&field_types, false); // packed = false
                    self.struct_field_indices.insert(item.id, indices);
                }
                ItemKind::ModuleDecl {
                    items: Some(subs), ..
                } => self.fill_struct_bodies(subs),
                _ => {}
            }
        }
    }

    fn get_field_index(&self, struct_id: DefId, field_name: &str) -> Option<u32> {
        self.struct_field_indices
            .get(&struct_id)
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

    fn declare_function(&mut self, func: &FunctionDefinition) {
        let param_types = func
            .params
            .iter()
            .map(|p| {
                let key = self.analyzer.types.get(&p.id).unwrap();
                self.compile_type(key).unwrap().into()
            })
            .collect::<Vec<_>>();

        let ret_key = self.analyzer.types.get(&func.id).unwrap();

        // 解析 FunctionType
        let fn_type = if let TypeKey::Function { ret: Some(r), .. } = ret_key {
            if let TypeKey::Primitive(PrimitiveType::Unit) = **r {
                self.context
                    .void_type()
                    .fn_type(&param_types, func.is_variadic)
            } else {
                let ret_ty = self.compile_type(r).unwrap();
                ret_ty.fn_type(&param_types, func.is_variadic)
            }
        } else {
            self.context
                .void_type()
                .fn_type(&param_types, func.is_variadic)
        };

        // 优先查 mangled_names，如果查不到（理论上不应发生），回退到原始名
        //? TODO: 更严谨的处理和更多的报错
        let fn_name = self
            .analyzer
            .mangled_names
            .get(&func.id)
            .cloned()
            .unwrap_or(func.name.name.clone());

        // 使用 fn_name 而不是 func.name.name
        let val = self.module.add_function(&fn_name, fn_type, None);
        self.functions.insert(func.id, val);
    }

    pub fn compile_function(&mut self, func: &FunctionDefinition) -> Result<(), String> {
        if func.body.is_none() {
            return Ok(());
        }

        // 既然上面检查过了，这里直接as_ref
        let body_ref = func.body.as_ref().unwrap();

        let function = *self.functions.get(&func.id).ok_or("Proto missing")?;
        self.current_fn = Some(function);

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        // 处理参数：Alloca + Store
        for (i, param) in func.params.iter().enumerate() {
            let arg_val = function.get_nth_param(i as u32).unwrap();
            arg_val.set_name(&param.name.name);

            let param_ty_key = self.analyzer.types.get(&param.id).unwrap();
            let llvm_ty = self.compile_type(param_ty_key).unwrap();

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
                unreachable!();
            }
        }

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
                let type_key = self.analyzer.types.get(&stmt.id).unwrap();
                let llvm_ty = self.compile_type(type_key).unwrap();

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
                let then_bb = self.context.append_basic_block(parent, "then");
                let else_bb = self.context.append_basic_block(parent, "else");
                let merge_bb = self.context.append_basic_block(parent, "merge");

                let cond_val = self.compile_expr(condition)?.into_int_value();
                let actual_else = if else_branch.is_some() {
                    else_bb
                } else {
                    merge_bb
                };

                self.builder
                    .build_conditional_branch(cond_val, then_bb, actual_else)
                    .ok();

                // Then
                self.builder.position_at_end(then_bb);
                self.compile_block(then_block)?;
                if !self.block_terminated(then_block) {
                    self.builder.build_unconditional_branch(merge_bb).ok();
                }

                // Else
                if let Some(else_stmt) = else_branch {
                    self.builder.position_at_end(else_bb);
                    self.compile_stmt(else_stmt)?;
                    // 如果当前块没结束指令就跳 merge
                    //? 完整的terminator检查？
                    if then_bb.get_terminator().is_none() {
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
        match &expr.kind {
            ExpressionKind::Literal(lit) => {
                // 查 Analyzer 的回写结果，精确控制位宽
                let type_key = self
                    .analyzer
                    .types
                    .get(&expr.id)
                    .ok_or("Literal Type Missing")?;
                self.compile_literal(lit, type_key)
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
                            .analyzer
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
            ExpressionKind::Path(_)
            | ExpressionKind::Index { .. }
            | ExpressionKind::FieldAccess { .. } => {
                let ptr = self.compile_expr_ptr(expr)?;
                let type_key = self.analyzer.types.get(&expr.id).unwrap();
                let llvm_ty = self.compile_type(type_key).unwrap();
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
                let def_id = self.analyzer.path_resolutions.get(&type_name.id).unwrap();
                let struct_type = *self
                    .struct_types
                    .get(def_id)
                    .expect("Struct type not found");

                let mut struct_val = struct_type.get_undef();

                let index_map = self.struct_field_indices.get(def_id).unwrap().clone();
                for field in fields {
                    let val = self.compile_expr(&field.value)?;
                    let idx = *index_map.get(&field.field_name.name).unwrap();

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
            } => {
                // 1. 获取 Receiver 的类型
                let receiver_ty_key = self.analyzer.types.get(&receiver.id).unwrap();

                // 2. 查 Analyzer 的 method_registry 找到对应的函数定义 ID
                let methods = self
                    .analyzer
                    .method_registry
                    .get(receiver_ty_key)
                    .ok_or(format!("No methods found for type {:?}", receiver_ty_key))?;

                let method_info = methods
                    .get(&method_name.name)
                    .ok_or(format!("Method '{}' not found", method_name.name))?;

                let fn_val = *self
                    .functions
                    .get(&method_info.def_id)
                    .expect("Function not compiled");

                // 3. 准备参数列表：[Receiver, ...Args]
                let mut compiled_args = Vec::new();

                // 处理 `self` 参数
                compiled_args.push(self.compile_expr(receiver)?.into());

                for arg in arguments {
                    compiled_args.push(self.compile_expr(arg)?.into());
                }

                // 4. 生成 Call
                let call_site = self
                    .builder
                    .build_call(fn_val, &compiled_args, "method_call")
                    .map_err(|_| "Method call failed")?;

                // 5. 处理返回值
                use inkwell::values::ValueKind;
                match call_site.try_as_basic_value() {
                    ValueKind::Basic(val) => Ok(val),
                    ValueKind::Instruction(_) => Ok(self
                        .context
                        .struct_type(&[], false)
                        .const_zero()
                        .as_basic_value_enum()),
                    _ => Err("Invalid call return".into()),
                }
            }

            ExpressionKind::StaticAccess {
                target: _,
                member: _,
            } => {
                if let Some(TypeKey::IntegerLiteral(val)) = self.analyzer.types.get(&expr.id) {
                    // 默认生成 i64
                    //? TODO: 查 Enum的underlying type生成具体的跳转
                    Ok(self
                        .context
                        .i64_type()
                        .const_int(*val, false)
                        .as_basic_value_enum())
                } else {
                    // 可能是读取静态变量 (Static Var)，暂时报错
                    //? TODO: Enum 读取静态变量?
                    Err("Static access (non-enum) not implemented".into())
                }
            }
        }
    }

    pub fn compile_expr_ptr(&mut self, expr: &Expression) -> Result<PointerValue<'ctx>, String> {
        match &expr.kind {
            ExpressionKind::Path(path) => {
                let def_id = self.analyzer.path_resolutions.get(&path.id).unwrap();
                self.get_variable_ptr(*def_id).ok_or("Var missing".into())
            }

            ExpressionKind::FieldAccess {
                receiver,
                field_name,
            } => {
                let ptr = self.compile_expr_ptr(receiver)?;
                // 获取 Receiver 的 Struct DefId
                let recv_key = self.analyzer.types.get(&receiver.id).unwrap();
                let struct_id = if let TypeKey::Named(id) = recv_key {
                    *id
                } else {
                    return Err("Not a struct".into());
                };

                let idx = self
                    .get_field_index(struct_id, &field_name.name)
                    .ok_or("Field missing")?;
                let st_ty = *self.struct_types.get(&struct_id).unwrap();

                let field_ptr = unsafe {
                    self.builder
                        .build_struct_gep(st_ty, ptr, idx, "gep")
                        .map_err(|_| "GEP failed")?
                };
                Ok(field_ptr)
            }

            ExpressionKind::Index { target, index } => {
                let ptr = self.compile_expr_ptr(target)?;
                let idx = self.compile_expr(index)?.into_int_value();
                let target_key = self.analyzer.types.get(&target.id).unwrap();

                match target_key {
                    TypeKey::Array(inner, _) => {
                        let arr_ty = self.compile_type(target_key).unwrap();
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
                    _ => Err("Index error".into()),
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
        let lhs_val = self.compile_expr(lhs)?;
        let rhs_val = self.compile_expr(rhs)?;

        // 获取 LHS 的类型信息 (Analyzer 保证了 LHS 和 RHS 类型兼容)
        let type_key = self
            .analyzer
            .types
            .get(&lhs.id)
            .ok_or("Type missing for binary op")?;
        let is_signed = self.is_signed(type_key);
        // 如果是 Float，需要一套完全不同的 build_float_* 指令
        if lhs_val.is_float_value() {
            return self.compile_float_binary(
                lhs_val.into_float_value(),
                op,
                rhs_val.into_float_value(),
            );
        }

        let l = lhs_val.into_int_value();
        let r = rhs_val.into_int_value();

        match op {
            // 加减乘：LLVM 底层指令对于有符号/无符号是一样的
            //? TODO：检测溢出 (nsw/nuw)
            BinaryOperator::Add => Ok(self.builder.build_int_add(l, r, "add").unwrap().into()),
            BinaryOperator::Subtract => Ok(self.builder.build_int_sub(l, r, "sub").unwrap().into()),
            BinaryOperator::Multiply => Ok(self.builder.build_int_mul(l, r, "mul").unwrap().into()),

            // 位运算
            BinaryOperator::BitwiseAnd => Ok(self.builder.build_and(l, r, "and").unwrap().into()),
            BinaryOperator::BitwiseOr => Ok(self.builder.build_or(l, r, "or").unwrap().into()),
            BinaryOperator::BitwiseXor => Ok(self.builder.build_xor(l, r, "xor").unwrap().into()),

            // 除法：区分符号
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

            // 取模：区分符号
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

            // 比较运算：区分 Predicate
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
        }
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
        // 1. 预先编译所有参数
        let mut compiled_args = Vec::new();
        for arg in args {
            compiled_args.push(self.compile_expr(arg)?.into());
        }

        // 2. 尝试Direct Call
        // 如果 callee 是一个直接的 Path，且该 Path 指向已知的函数定义
        if let ExpressionKind::Path(path) = &callee.kind {
            if let Some(def_id) = self.analyzer.path_resolutions.get(&path.id) {
                // 在 functions 表里找，如果找到了，说明是全局函数
                if let Some(fn_val) = self.functions.get(def_id) {
                    let call_site = self
                        .builder
                        .build_call(*fn_val, &compiled_args, "direct_call")
                        .map_err(|_| "Direct call failed")?;
                    return self.handle_call_return(call_site);
                }
            }
        }

        // 静态方法调用 (Struct::method)
        if let ExpressionKind::StaticAccess { target: _, member } = &callee.kind {
            if let Some(def_id) = self.analyzer.path_resolutions.get(&callee.id) {
                if let Some(fn_val) = self.functions.get(def_id) {
                    let call_site = self
                        .builder
                        .build_call(*fn_val, &compiled_args, "static_call")
                        .map_err(|_| "Static call failed")?;
                    return self.handle_call_return(call_site);
                }
            }
        }

        // 3. “间接调用” (Indirect Call) - 函数指针
        // 如果走到这里，说明 callee 是一个表达式（比如变量、数组元素、返回函数指针的函数调用等）
        //? TODO: 设计语法支持间接调用?
        //? TODO: 闭包?

        // A. 编译表达式得到函数指针 (PointerValue)
        let fn_ptr_val = self.compile_expr(callee)?.into_pointer_value();

        // B. 获取函数类型签名
        let callee_type_key = self
            .analyzer
            .types
            .get(&callee.id)
            .ok_or("Callee type missing in analyzer")?;

        // C. 将 TypeKey 转换为 LLVM FunctionType
        let fn_type = self.compile_function_type_signature(callee_type_key)?;

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
            _ => Err("Invalid call return kind".into()),
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
        self.variables.get(&def_id).cloned()
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
            .analyzer
            .types
            .get(&src_expr.id)
            .ok_or("Source type missing")?;
        let target_key = self
            .analyzer
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
                //? Inkwell build_float_cast 会自动处理 Ext/Trunc?
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
        let ty_key = self.analyzer.types.get(&id).unwrap();
        let llvm_ty = self.compile_type(ty_key).unwrap();

        // 2. 获取修饰名
        let name = self
            .analyzer
            .mangled_names
            .get(&id)
            .cloned()
            .unwrap_or(def.name.name.clone());

        // 3. 创建全局变量
        let global = self
            .module
            .add_global(llvm_ty, Some(AddressSpace::default()), &name);

        // 4. 设置可变性 (LLVM IR 里的 constant 意味着只读)
        global.set_constant(def.modifier == Mutability::Constant);

        // 5. 设置初始化值
        if let Some(init) = &def.initializer {
            // A: 如果是简单字面量 (String, Float, Bool)，直接编译
            if let ExpressionKind::Literal(lit) = &init.kind {
                let lit_ty = self.analyzer.types.get(&init.id).unwrap();
                let val = self
                    .compile_literal(lit, lit_ty)
                    .expect("Literal compile failed");
                global.set_initializer(&val);
            }
            // B: 如果 Analyzer 已经计算出了整数值 (Int 运算和 指针强转)
            else if let Some(&const_val) = self.analyzer.constants.get(&id) {
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

        // 6. 注册到符号表
        self.variables.insert(id, global.as_pointer_value());
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
}
