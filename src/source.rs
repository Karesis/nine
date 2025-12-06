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

use std::ops::Range;
use std::path::PathBuf;

/// 简单的文件句柄，避免到处传 String
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileId(pub usize);

/// 源码区间：只存偏移量，不存内容，Copy 代价极小
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// 方便转换为 Range 用于切片： &src[span.as_range()]
    pub fn as_range(&self) -> Range<usize> {
        self.start..self.end
    }
}

/// ======================================================
/// 2. 单个文件结构: SourceFile
/// ======================================================

#[derive(Debug)]
pub struct SourceFile {
    /// 真实的文件路径 (新增!)
    pub path: PathBuf,
    /// 文件名 (用于报错显示)
    pub name: String,
    /// 完整源码
    pub src: String,
    /// 每一行开始的字节偏移量缓存
    /// lines[0] = 0, lines[1] = 第一行结束位置+1...
    pub line_starts: Vec<usize>,
    /// 该文件在虚拟全局空间中的起始偏移量
    pub start_pos: usize,
}

impl SourceFile {
    pub fn new(path: PathBuf, src: String, start_pos: usize) -> Self {
        let line_starts = std::iter::once(0)
            .chain(src.match_indices('\n').map(|(i, _)| i + 1))
            .collect();

        let name = path.to_string_lossy().to_string();

        Self {
            path,
            name,
            src,
            line_starts,
            start_pos,
        }
    }

    /// 核心算法：利用二分查找快速定位行号
    pub fn lookup_line(&self, offset: usize) -> usize {
        match self.line_starts.binary_search(&offset) {
            Ok(line) => line + 1,
            Err(line) => line,
        }
    }

    /// 获取 (Line, Column)
    /// 注意：编辑器通常行号列号都从 1 开始
    pub fn lookup_location(&self, offset: usize) -> (usize, usize) {
        let line = self.lookup_line(offset);
        let line_start = self.line_starts[line - 1];
        let col = offset - line_start + 1;
        (line, col)
    }
}

use std::fs;
/// ======================================================
/// 3. 总管：SourceManager
/// ======================================================
use std::io;
use std::path::Path;

#[derive(Debug)]
pub struct SourceManager {
    files: Vec<SourceFile>,
    /// 下一个文件应该从哪个全局偏移量开始
    next_offset: usize,
}

impl SourceManager {
    pub fn new() -> Self {
        Self {
            files: Vec::new(),
            next_offset: 0,
        }
    }

    /// 加载文件并返回 ID
    pub fn load_file<P: AsRef<Path>>(&mut self, path: P) -> io::Result<FileId> {
        let path = path.as_ref();
        let abs_path = fs::canonicalize(path)?;

        if let Some((id, _)) = self
            .files
            .iter()
            .enumerate()
            .find(|(_, f)| f.path == abs_path)
        {
            return Ok(FileId(id));
        }

        let src = fs::read_to_string(&abs_path)?;
        let len = src.len();

        let start_pos = self.next_offset;

        self.next_offset += len + 1;

        let file = SourceFile::new(abs_path, src, start_pos);

        let id = FileId(self.files.len());
        self.files.push(file);
        Ok(id)
    }

    /// 获取文件所在的目录 (核心功能：用于解析相对路径的 import)
    pub fn get_file_dir(&self, id: FileId) -> PathBuf {
        let file = &self.files[id.0];

        file.path.parent().unwrap_or(Path::new(".")).to_path_buf()
    }

    /// 直接添加字符串（用于测试或 REPL）
    pub fn add_file(&mut self, name: String, src: String) -> io::Result<FileId> {
        let len = src.len();
        let start_pos = self.next_offset;
        self.next_offset += len + 1;
        let file = SourceFile::new(PathBuf::from(name), src, start_pos);
        let id = FileId(self.files.len());
        self.files.push(file);
        Ok(id)
    }

    /// 获取文件引用
    pub fn get_file(&self, id: FileId) -> &SourceFile {
        &self.files[id.0]
    }

    /// 获取源码片段
    pub fn get_source(&self, id: FileId, span: Span) -> &str {
        &self.files[id.0].src[span.as_range()]
    }

    pub fn lookup_source(&self, span: Span) -> Option<(&SourceFile, usize, usize)> {
        for file in &self.files {
            let file_end = file.start_pos + file.src.len();
            if span.start >= file.start_pos && span.start < file_end {
                let local_start = span.start - file.start_pos;
                let (line, col) = file.lookup_location(local_start);
                return Some((file, line, col));
            }
        }
        None
    }
}
