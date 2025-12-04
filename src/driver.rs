// src/driver.rs
use std::path::{Path, PathBuf};
use crate::source::{SourceManager, FileId};
use crate::parser::{Parser, ParseError};
use crate::lexer::Lexer;
use crate::ast::{ItemKind, Program};

pub struct Driver {
    pub source_manager: SourceManager,
    pub global_node_id: u32,
    pub parse_errors: Vec<ParseError>,
}

impl Driver {
    pub fn new() -> Self {
        Self { source_manager: SourceManager::new(), global_node_id: 0, parse_errors: Vec::new(),}
    }

   /// 入口：编译项目
    pub fn compile_project(&mut self, entry_path: PathBuf) -> Result<Program, String> {
        let abs_entry = std::fs::canonicalize(&entry_path)
            .map_err(|e| format!("Invalid entry path: {}", e))?;

        let root_id = self.source_manager.load_file(&abs_entry)
            .map_err(|e| format!("Cannot load root file: {}", e))?;
        
        // 每次编译前清空错误
        self.parse_errors.clear();

        // 根文件的 is_root = true
        self.parse_module(root_id, true)
    }

    /// 递归解析模块
    fn parse_module(&mut self, file_id: FileId, is_root: bool) -> Result<Program, String> {
        let file = self.source_manager.get_file(file_id);
        let src = file.src.clone();
        let base_offset = file.start_pos;
        let current_file_path = file.path.clone();
        
        // 2. Parse
        let lexer = Lexer::new(&src);
        let mut parser = Parser::new(&src, lexer, base_offset, &mut self.global_node_id);
        let mut program = parser.parse_program();

        if !parser.errors.is_empty() {
            // 将 parser 里的错误转移到 driver 里
            self.parse_errors.extend(parser.errors);
        }

        // === 3. 计算子模块查找路径 (保持原逻辑) ===
        let parent_dir = current_file_path.parent().unwrap();
        let file_stem = current_file_path.file_stem().unwrap().to_str().unwrap();

        let search_dir = if is_root {
            parent_dir.to_path_buf()
        } else if file_stem == "entry" {
            parent_dir.to_path_buf()
        } else {
            parent_dir.join(file_stem)
        };

        // 4. 遍历 AST，加载 ModuleDecl
        for item in &mut program.items {
            if let ItemKind::ModuleDecl { name, items, .. } = &mut item.kind {
                if items.is_some() { continue; } 

                let mod_name = &name.name;
                
                // 查找策略
                let path_sibling = search_dir.join(format!("{}.9", mod_name));
                let path_entry = search_dir.join(mod_name).join("entry.9");

                let target_path = if path_sibling.exists() {
                    path_sibling
                } else if path_entry.exists() {
                    path_entry
                } else {
                    return Err(format!(
                        "Module '{}' not found at {:?} (span: {:?})", 
                        mod_name, search_dir, name.span
                    ));
                };

                let sub_id = self.source_manager.load_file(&target_path)
                    .map_err(|e| format!("Failed to load module {}: {}", mod_name, e))?;
                
                // 递归
                let sub_program = self.parse_module(sub_id, false)?;
                *items = Some(sub_program.items);
            }
        }

        Ok(program)
    }
}