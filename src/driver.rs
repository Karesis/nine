// src/driver.rs
use std::path::{Path, PathBuf};
use crate::source::{SourceManager, FileId};
use crate::parser::Parser;
use crate::lexer::Lexer;
use crate::ast::{ItemKind, Program, Item};

pub struct Driver {
    pub source_manager: SourceManager,
}

impl Driver {
    pub fn new() -> Self {
        Self { source_manager: SourceManager::new() }
    }

    /// 入口：编译项目
    pub fn compile_project(&mut self, entry_path: PathBuf) -> Result<Program, String> {
        let abs_entry = std::fs::canonicalize(&entry_path)
            .map_err(|e| format!("Invalid entry path: {}", e))?;

        let root_id = self.source_manager.load_file(&abs_entry)
            .map_err(|e| format!("Cannot load root file: {}", e))?;
        
        // 根文件的 is_root = true，这决定了它查找子模块的方式
        self.parse_module(root_id, true)
    }

    /// 递归解析模块
    /// is_root: 标记当前解析的文件是否是根文件（影响路径查找策略）
    fn parse_module(&mut self, file_id: FileId, is_root: bool) -> Result<Program, String> {
        // 1. 获取源码和 【base_offset】
        let file = self.source_manager.get_file(file_id);
        let src = file.src.clone();
        let base_offset = file.start_pos; // <--- 获取全局偏移
        
        // 2. Parse (传入 base_offset)
        let lexer = Lexer::new(&src);
        let mut parser = Parser::new(&src, lexer, base_offset);
        let mut program = parser.parse_program();

        if !parser.errors.is_empty() {
            // 简单报错，实际项目可能需要更漂亮的错误打印
            let file_name = &self.source_manager.get_file(file_id).name;
            return Err(format!("Parse error in {}: {:?}", file_name, parser.errors));
        }

        // 2. 计算子模块查找的基准目录
        let current_file_path = self.source_manager.get_file(file_id).path.clone();
        let current_dir = current_file_path.parent().unwrap();
        
        // 路径策略：
        // 如果是 root (main.9)，mod foo -> ./foo.9 (同级)
        // 如果是 module (foo.9)，mod bar -> ./foo/bar.9 (子级)
        let search_dir = if is_root {
            current_dir.to_path_buf()
        } else {
            // 获取文件名（无后缀），例如 "foo.9" -> "foo"
            let stem = current_file_path.file_stem().unwrap();
            current_dir.join(stem)
        };

        // 3. 遍历 AST，寻找并加载 ModuleDecl
        for item in &mut program.items {
            if let ItemKind::ModuleDecl { name, items, .. } = &mut item.kind {
                // 如果 items 已经是 Some，说明可能被预处理过（或者将来支持内联 mod { ... }）
                if items.is_some() { continue; }

                let mod_name = &name.name;
                
                // 构造期望的路径： search_dir / mod_name.9
                let mod_path = search_dir.join(format!("{}.9", mod_name));
                
                // 检查文件是否存在
                if !mod_path.exists() {
                     return Err(format!(
                         "Module not found: '{}'. Expected at {:?}", 
                         mod_name, mod_path
                     ));
                }

                // 加载并解析子模块
                // 注意：子模块 is_root = false
                let sub_id = self.source_manager.load_file(&mod_path)
                    .map_err(|e| format!("Failed to load module {}: {}", mod_name, e))?;
                
                let sub_program = self.parse_module(sub_id, false)?;

                // 核心步骤：将解析出的子 AST 挂载到当前节点的 items 字段
                *items = Some(sub_program.items);
                
                // 可选：打印调试信息
                println!("Loaded module '{}' from {:?}", mod_name, mod_path);
            }
        }

        Ok(program)
    }
}