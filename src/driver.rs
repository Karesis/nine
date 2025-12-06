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

use crate::ast::{ItemKind, Program};
use crate::lexer::Lexer;
use crate::parser::{ParseError, Parser};
use crate::source::{FileId, SourceManager};
use std::path::{Path, PathBuf};

pub struct Driver {
    pub source_manager: SourceManager,
    pub global_node_id: u32,
    pub parse_errors: Vec<ParseError>,
}

impl Driver {
    pub fn new() -> Self {
        Self {
            source_manager: SourceManager::new(),
            global_node_id: 0,
            parse_errors: Vec::new(),
        }
    }

    /// 入口：编译项目
    pub fn compile_project(&mut self, entry_path: PathBuf) -> Result<Program, String> {
        let abs_entry =
            std::fs::canonicalize(&entry_path).map_err(|e| format!("Invalid entry path: {}", e))?;

        let root_id = self
            .source_manager
            .load_file(&abs_entry)
            .map_err(|e| format!("Cannot load root file: {}", e))?;

        self.parse_errors.clear();

        self.parse_module(root_id, true)
    }

    /// 递归解析模块
    fn parse_module(&mut self, file_id: FileId, is_root: bool) -> Result<Program, String> {
        let file = self.source_manager.get_file(file_id);
        let src = file.src.clone();
        let base_offset = file.start_pos;
        let current_file_path = file.path.clone();

        let lexer = Lexer::new(&src);
        let mut parser = Parser::new(&src, lexer, base_offset, &mut self.global_node_id);
        let mut program = parser.parse_program();

        if !parser.errors.is_empty() {
            self.parse_errors.extend(parser.errors);
        }

        let parent_dir = current_file_path.parent().unwrap();
        let file_stem = current_file_path.file_stem().unwrap().to_str().unwrap();

        let search_dir = if is_root {
            parent_dir.to_path_buf()
        } else if file_stem == "entry" {
            parent_dir.to_path_buf()
        } else {
            parent_dir.join(file_stem)
        };

        for item in &mut program.items {
            if let ItemKind::ModuleDecl { name, items, .. } = &mut item.kind {
                if items.is_some() {
                    continue;
                }

                let mod_name = &name.name;

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

                let sub_id = self
                    .source_manager
                    .load_file(&target_path)
                    .map_err(|e| format!("Failed to load module {}: {}", mod_name, e))?;

                let sub_program = self.parse_module(sub_id, false)?;
                *items = Some(sub_program.items);
            }
        }

        Ok(program)
    }
}
