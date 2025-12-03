// src/driver.rs
use std::path::{Path, PathBuf};
use crate::source::{SourceManager, FileId};
use crate::parser::Parser;
use crate::lexer::Lexer;
use crate::ast::{ItemKind, Program, Item};

pub struct Driver {
    pub source_manager: SourceManager,
    pub global_node_id: u32,
}

impl Driver {
    pub fn new() -> Self {
        Self { source_manager: SourceManager::new(), global_node_id: 0,}
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
    fn parse_module(&mut self, file_id: FileId, is_root: bool) -> Result<Program, String> {
        // 1. 获取源码和 base_offset
        let file = self.source_manager.get_file(file_id);
        let src = file.src.clone();
        let base_offset = file.start_pos;
        let current_file_path = file.path.clone(); // Clone path to avoid borrow issues
        
        // 2. Parse
        let lexer = Lexer::new(&src);
        let mut parser = Parser::new(&src, lexer, base_offset, &mut self.global_node_id);
        let mut program = parser.parse_program();

        if !parser.errors.is_empty() {
            let file_name = &self.source_manager.get_file(file_id).name;
            return Err(format!("Parse error in {}: {:?}", file_name, parser.errors));
        }

        // === 3. 计算【子模块查找基准目录】 (Search Base) ===
        let parent_dir = current_file_path.parent().unwrap();
        let file_stem = current_file_path.file_stem().unwrap().to_str().unwrap();

        // 这里的逻辑需要适配 entry.9 的情况
        let search_dir = if is_root {
            // 根文件所在目录就是基准
            parent_dir.to_path_buf()
        } else if file_stem == "entry" {
            // 如果当前文件是 "foo/entry.9"，它代表的是 "foo" 模块
            // 它的子模块（如 "bar"）应该在 "foo/bar.9" 或 "foo/bar/entry.9"
            // 所以基准目录就是当前 entry.9 所在的目录
            parent_dir.to_path_buf()
        } else {
            // 如果当前文件是 "foo.9" (Sibling style)
            // 它的子模块应该在 "foo/" 目录下
            // 所以基准目录是 "./foo/"
            parent_dir.join(file_stem)
        };

        // 4. 遍历 AST，加载 ModuleDecl
        for item in &mut program.items {
            if let ItemKind::ModuleDecl { name, items, .. } = &mut item.kind {
                if items.is_some() { continue; } 

                let mod_name = &name.name;
                
                // === 9-lang 模块查找策略 ===
                
                // 1. 优先查找同级文件: search_dir/mod_name.9
                // 例如: .../net.9
                let path_sibling = search_dir.join(format!("{}.9", mod_name));
                
                // 2. 其次查找文件夹入口: search_dir/mod_name/entry.9
                // 例如: .../net/entry.9
                let path_entry = search_dir.join(mod_name).join("entry.9");

                // 决策
                let target_path = if path_sibling.exists() {
                    path_sibling
                } else if path_entry.exists() {
                    path_entry
                } else {
                    return Err(format!(
                        "Module '{}' not found.\nSearched at:\n - {:?} (Sibling)\n - {:?} (Entry)", 
                        mod_name, path_sibling, path_entry
                    ));
                };

                // 加载子模块
                let sub_id = self.source_manager.load_file(&target_path)
                    .map_err(|e| format!("Failed to load module {}: {}", mod_name, e))?;
                
                let sub_program = self.parse_module(sub_id, false)?;

                *items = Some(sub_program.items);
            }
        }

        Ok(program)
    }
}